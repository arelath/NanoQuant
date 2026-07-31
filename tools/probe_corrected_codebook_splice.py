"""Gate the best sparse-corrected sign codebook with paired held-out splice KL."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_sign_word_codebook import (
    PINNED_MODEL_REVISION,
    PROJECTION_PATHS,
    _load_profile,
    _logical_seed,
    _metrics,
)
from safetensors import safe_open

from nanoquant.application.kl_budget import paired_bootstrap_kl_delta
from nanoquant.config.codec import to_dict
from nanoquant.domain.factorization import (
    AdmmParameters,
    factorize_admm_with_parameters,
)
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.domain.sign_word_codebook import (
    codebook_index_metrics,
    corrected_asymmetric_codebook_bit_cost,
    factorize_sign_word_codebook_admm,
    maximum_corrected_asymmetric_rank_for_budget,
    mixed_right_corrected_codebook_bit_cost,
)
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

MODEL_SOURCE = "google/gemma-3-1b-it"


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("blocks must be unique non-negative integers")
    return result


def _parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(
        float(item.strip()) for item in value.split(",") if item.strip()
    )
    if (
        not result
        or len(result) != len(set(result))
        or any(item <= 0 for item in result)
    ):
        raise argparse.ArgumentTypeError(
            "selection thresholds must be unique positive fractions"
        )
    return result


def _parse_projections(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if (
        not result
        or len(result) != len(set(result))
        or any(item not in PROJECTION_PATHS for item in result)
    ):
        raise argparse.ArgumentTypeError(
            "projections must be unique known projection names"
        )
    return result


def _dtype(config: dict[str, object]) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(cast(str, config.get("torch_dtype")), torch.float32)


def _reconstruction_set(
    values: list[tuple[LayerId, torch.Tensor, float]],
) -> SpliceReconstructionSet:
    layers = []
    units = []
    errors = []
    for layer, reconstruction, weighted_rmse in values:
        unit = f"{layer.block.index}:{layer.path}"
        squared_error = weighted_rmse**2
        layers.append(
            SpliceReconstruction(
                layer,
                reconstruction.detach().to(device="cpu", dtype=torch.bfloat16),
                None,
                squared_error,
            )
        )
        units.append((unit, (layer,)))
        errors.append((unit, squared_error))
    return SpliceReconstructionSet(
        tuple(layers),
        tuple(units),
        tuple(errors),
    )


def _select_blocks(
    reconstructions: SpliceReconstructionSet,
    blocks: tuple[int, ...],
) -> SpliceReconstructionSet:
    selected = frozenset(blocks)
    layers = tuple(
        layer
        for layer in reconstructions.layers
        if layer.layer.block.index in selected
    )
    if {layer.layer.block.index for layer in layers} != selected:
        raise ValueError("selected splice blocks do not map to reconstruction layers")
    units = tuple(
        (unit, members)
        for unit, members in reconstructions.unit_members
        if members and all(member.block.index in selected for member in members)
    )
    errors = tuple(
        (unit, error)
        for unit, error in reconstructions.unit_weighted_normalized_squared_errors
        if any(candidate == unit for candidate, _members in units)
    )
    return SpliceReconstructionSet(layers, units, errors)


def _select_token_window(
    tokens: torch.Tensor,
    *,
    offset: int,
    samples: int,
) -> torch.Tensor:
    if tokens.ndim != 2 or offset < 0 or samples <= 0:
        raise ValueError("held-out token window is invalid")
    selected = tokens[offset : offset + samples]
    if selected.shape[0] != samples:
        raise ValueError("held-out token inventory is shorter than requested")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--projection", choices=tuple(PROJECTION_PATHS), default="down")
    parser.add_argument("--projections", type=_parse_projections)
    parser.add_argument("--transpose-matrix", action="store_true")
    parser.add_argument("--block", type=int, default=12)
    parser.add_argument("--blocks", type=_parse_ints)
    parser.add_argument("--selection-thresholds", type=_parse_floats)
    parser.add_argument("--baseline-rank", type=int, default=970)
    parser.add_argument("--candidate-rank", type=int)
    parser.add_argument("--right-free-rows", type=int, default=0)
    parser.add_argument("--index-width", type=int, default=10)
    parser.add_argument("--corrections-per-word", type=int, default=2)
    parser.add_argument("--correction-bits", type=int, default=9)
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--codebook-update-interval", type=int, default=10)
    parser.add_argument("--codebook-freeze-fraction", type=float, default=0.5)
    parser.add_argument("--assignment-batch-words", type=int, default=8192)
    parser.add_argument("--corrected-assignment-candidates", type=int, default=16)
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--wikitext-samples", type=int, default=12)
    parser.add_argument("--wikitext-offset", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.corrections_per_word not in {1, 2, 3}:
        raise ValueError("splice probe supports one to three corrections")
    if args.candidate_rank is not None and (
        args.candidate_rank <= 0
        or args.right_free_rows < 0
        or args.right_free_rows >= args.candidate_rank
    ):
        raise ValueError("candidate rank/free-row configuration is invalid")
    if (
        args.wikitext_samples <= 0
        or args.wikitext_offset < 0
        or args.sequence_length < 2
    ):
        raise ValueError("held-out token dimensions are invalid")
    config_payload = json.loads(
        (args.snapshot / "config.json").read_text(encoding="utf-8")
    )
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    adapter = adapter_for_config(config)
    projections = (
        (args.projection,)
        if args.projections is None
        else args.projections
    )
    projection_paths = tuple(PROJECTION_PATHS[item] for item in projections)
    blocks = (args.block,) if args.blocks is None else args.blocks
    if any(
        block >= adapter.decoder_block_count_from_config(config)
        for block in blocks
    ):
        raise ValueError("requested block is outside the model")
    parameters = AdmmParameters(
        outer_iterations=args.outer_iterations,
        inner_iterations=args.inner_iterations,
        regularization=args.regularization,
        penalty_schedule=args.penalty_schedule,
        convergence_check_interval=args.convergence_check_interval,
        transpose_wide=True,
    )
    with acquire_device_lease(args.device), safe_open(
        str(args.model),
        framework="pt",
        device="cpu",
    ) as handle:
        baseline_entries: list[tuple[LayerId, torch.Tensor, float]] = []
        candidate_entries: list[tuple[LayerId, torch.Tensor, float]] = []
        reconstruction_metrics: dict[str, dict[str, dict[str, float]]] = {}
        candidate_index_metrics: dict[
            str,
            dict[str, dict[str, float | int | bool]],
        ] = {}
        rank = 0
        matrix_shape = (0, 0)
        for block in blocks:
            for projection, projection_path in zip(
                projections,
                projection_paths,
                strict=True,
            ):
                tensor_name = f"model.layers.{block}.{projection_path}.weight"
                calibration_path = f"block.{block}.{projection_path}"
                input_cpu, output_cpu = _load_profile(
                    args.calibration_state,
                    calibration_path,
                    args.calibration_shrinkage,
                )
                weight = handle.get_tensor(tensor_name).to(args.device)
                input_importance = input_cpu.to(args.device).float()
                output_importance = output_cpu.to(args.device).float()
                if args.transpose_matrix:
                    weight = weight.mT.contiguous()
                    input_importance, output_importance = (
                        output_importance,
                        input_importance,
                    )
                current_shape = (int(weight.shape[0]), int(weight.shape[1]))
                if matrix_shape != (0, 0) and current_shape != matrix_shape:
                    raise ValueError(
                        "joint projection splices require one factorization shape"
                    )
                matrix_shape = current_shape
                unit_key = (
                    str(block)
                    if len(projections) == 1
                    else f"{block}:{projection}"
                )
                baseline_generator = torch.Generator(device=args.device).manual_seed(
                    _logical_seed(args.seed, "free-word-baseline")
                )
                baseline_factors = factorize_admm_with_parameters(
                    weight,
                    input_importance,
                    output_importance,
                    args.baseline_rank,
                    baseline_generator,
                    parameters,
                )
                baseline_fit = fit_scales(
                    weight,
                    baseline_factors.left_binary,
                    baseline_factors.right_binary,
                    baseline_factors.scale_pre,
                    baseline_factors.scale_mid,
                    baseline_factors.scale_post,
                    input_importance,
                    output_importance,
                    alternating_passes=args.scale_fit_passes,
                )
                baseline_metrics = _metrics(
                    weight,
                    baseline_fit.reconstruction,
                    input_importance,
                    output_importance,
                )
                target_bits = factor_bit_cost(
                    weight.shape[0],
                    weight.shape[1],
                    args.baseline_rank,
                    scale_bits=16,
                ).total
                rank = maximum_corrected_asymmetric_rank_for_budget(
                    weight.shape[0],
                    weight.shape[1],
                    target_bits,
                    left_index_width=None,
                    right_index_width=args.index_width,
                    right_flip_bits=args.correction_bits,
                    rank_multiple=32,
                    scale_width=16,
                )
                if args.candidate_rank is not None:
                    rank = args.candidate_rank
                candidate_generator = torch.Generator(device=args.device).manual_seed(
                    _logical_seed(
                        args.seed,
                        f"full-right-flip{args.corrections_per_word}-"
                        f"{args.index_width}-rank-{rank}",
                    )
                )
                candidate_factors = factorize_sign_word_codebook_admm(
                    weight,
                    input_importance,
                    output_importance,
                    rank,
                    candidate_generator,
                    index_bits=args.index_width,
                    outer_iterations=args.outer_iterations,
                    inner_iterations=args.inner_iterations,
                    regularization=args.regularization,
                    penalty_schedule=args.penalty_schedule,
                    convergence_check_interval=args.convergence_check_interval,
                    codebook_update_interval=args.codebook_update_interval,
                    codebook_freeze_fraction=args.codebook_freeze_fraction,
                    assignment_batch_words=args.assignment_batch_words,
                    corrected_assignment_candidates=(
                        args.corrected_assignment_candidates
                    ),
                    codebook_mode="full",
                    constrain_left=False,
                    right_flips_per_word=args.corrections_per_word,
                    right_free_rows=args.right_free_rows,
                )
                candidate_fit = fit_scales(
                    weight,
                    candidate_factors.factors.left_binary,
                    candidate_factors.factors.right_binary,
                    candidate_factors.factors.scale_pre,
                    candidate_factors.factors.scale_mid,
                    candidate_factors.factors.scale_post,
                    input_importance,
                    output_importance,
                    alternating_passes=args.scale_fit_passes,
                )
                candidate_metrics = _metrics(
                    weight,
                    candidate_fit.reconstruction,
                    input_importance,
                    output_importance,
                )
                candidate_index_metrics[unit_key] = codebook_index_metrics(
                    candidate_factors
                )
                layer = LayerId(BlockId(block), projection_path)
                baseline_reconstruction = baseline_fit.reconstruction
                candidate_reconstruction = candidate_fit.reconstruction
                if args.transpose_matrix:
                    baseline_reconstruction = baseline_reconstruction.mT.contiguous()
                    candidate_reconstruction = candidate_reconstruction.mT.contiguous()
                baseline_entries.append(
                    (
                        layer,
                        baseline_reconstruction,
                        float(baseline_metrics["weighted_normalized_rmse"]),
                    )
                )
                candidate_entries.append(
                    (
                        layer,
                        candidate_reconstruction,
                        float(candidate_metrics["weighted_normalized_rmse"]),
                    )
                )
                reconstruction_metrics[unit_key] = {
                    "free_words": baseline_metrics,
                    "corrected_codebook": candidate_metrics,
                }
                del (
                    weight,
                    input_importance,
                    output_importance,
                    baseline_factors,
                    baseline_fit,
                    candidate_fit,
                    candidate_factors,
                )
                gc.collect()
                torch.cuda.empty_cache()
        all_reconstruction_sets = {
            "free_words": _reconstruction_set(baseline_entries),
            "corrected_codebook": _reconstruction_set(candidate_entries),
        }
        selection_specs: list[tuple[str, float | None, tuple[int, ...]]] = []
        if args.selection_thresholds is None:
            selection_specs.append(("full", None, blocks))
        else:
            for threshold in args.selection_thresholds:
                selected = tuple(
                    block
                    for block in blocks
                    if all(
                        (
                            1
                            - reconstruction_metrics[
                                (
                                    str(block)
                                    if len(projections) == 1
                                    else f"{block}:{projection}"
                                )
                            ]["corrected_codebook"]["weighted_normalized_rmse"]
                            / reconstruction_metrics[
                                (
                                    str(block)
                                    if len(projections) == 1
                                    else f"{block}:{projection}"
                                )
                            ]["free_words"]["weighted_normalized_rmse"]
                        )
                        >= threshold
                        for projection in projections
                    )
                )
                if not selected:
                    raise ValueError(
                        f"selection threshold {threshold} selects no blocks"
                    )
                selection_specs.append(
                    (f"weighted_gain_ge_{threshold:.6g}", threshold, selected)
                )

        tokens, dataset_fingerprint, _bos = _wikitext_tokens(
            args.snapshot,
            samples=args.wikitext_offset + args.wikitext_samples,
            sequence_length=args.sequence_length,
            local_files_only=args.local_files_only,
        )
        tokens = _select_token_window(
            tokens,
            offset=args.wikitext_offset,
            samples=args.wikitext_samples,
        )
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        teacher_nll = 0.0
        teacher_cache: tuple[torch.Tensor, ...] = ()
        selection_results: dict[str, dict[str, Any]] = {}
        for name, threshold, selected_blocks in selection_specs:
            reconstruction_sets = {
                arm: _select_blocks(reconstructions, selected_blocks)
                for arm, reconstructions in all_reconstruction_sets.items()
            }
            baseline_evaluator = DenseKlSpliceEvaluator(
                teacher,
                reconstruction_sets["free_words"],
                tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            if not teacher_cache:
                teacher_nll, teacher_cache = (
                    baseline_evaluator.teacher_cache_state()
                )
            else:
                baseline_evaluator.install_teacher_cache(
                    teacher_nll,
                    teacher_cache,
                )
            baseline_kl = baseline_evaluator("full")
            candidate_evaluator = DenseKlSpliceEvaluator(
                teacher,
                reconstruction_sets["corrected_codebook"],
                tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            candidate_evaluator.install_teacher_cache(
                teacher_nll,
                teacher_cache,
            )
            candidate_kl = candidate_evaluator("full")
            interval = paired_bootstrap_kl_delta(
                baseline_kl,
                candidate_kl,
                seed=args.seed,
            )
            selection_results[name] = {
                "minimum_weighted_rmse_gain_fraction": threshold,
                "blocks": list(selected_blocks),
                "kl": {
                    "free_words": to_dict(baseline_kl),
                    "corrected_codebook": to_dict(candidate_kl),
                },
                "paired_candidate_minus_free_words": {
                    "point_delta": interval.point_delta,
                    "relative_delta": (
                        interval.point_delta
                        / baseline_kl.kl_nats_per_token
                    ),
                    "lower_delta": interval.lower_delta,
                    "upper_delta": interval.upper_delta,
                    "confidence": interval.confidence,
                    "resamples": interval.resamples,
                    "improved_with_confidence": (
                        interval.point_delta < 0 and interval.upper_delta < 0
                    ),
                },
            }
            del baseline_evaluator, candidate_evaluator
        del teacher

    cost = (
        mixed_right_corrected_codebook_bit_cost(
            matrix_shape[0],
            matrix_shape[1],
            rank,
            right_free_rows=args.right_free_rows,
            right_index_width=args.index_width,
            right_flip_bits=args.correction_bits,
            scale_width=16,
        )
        if args.right_free_rows
        else corrected_asymmetric_codebook_bit_cost(
            matrix_shape[0],
            matrix_shape[1],
            rank,
            left_index_width=None,
            right_index_width=args.index_width,
            right_flip_bits=args.correction_bits,
            scale_width=16,
        )
    )
    output: dict[str, Any] = {
        "schema_version": 5,
        "status": "completed",
        "role": "analysis-only corrected-codebook splice gate",
        "model_source": MODEL_SOURCE,
        "model_revision": args.model_revision,
        "blocks": list(blocks),
        "projection": projection_paths[0] if len(projection_paths) == 1 else None,
        "projections": list(projection_paths),
        "transposed_for_factorization": args.transpose_matrix,
        "factorization_shape": list(matrix_shape),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_slice_hash": _token_hash(tokens),
        "wikitext_samples": args.wikitext_samples,
        "wikitext_offset": args.wikitext_offset,
        "sequence_length": args.sequence_length,
        "teacher_baseline_nll": teacher_nll,
        "candidate": {
            "index_width": args.index_width,
            "corrections_per_word": args.corrections_per_word,
            "correction_bits": args.correction_bits,
            "rank": rank,
            "right_free_rows": args.right_free_rows,
            "corrected_assignment_candidates": (
                args.corrected_assignment_candidates
            ),
            "bit_cost": asdict(cost),
            "actual_bpw": cost.total / (matrix_shape[0] * matrix_shape[1]),
            "index_metrics_by_block": candidate_index_metrics,
        },
        "reconstruction_by_block": reconstruction_metrics,
        "selection_results": selection_results,
    }
    if args.selection_thresholds is None:
        full = selection_results["full"]
        output["kl"] = full["kl"]
        output["paired_candidate_minus_free_words"] = full[
            "paired_candidate_minus_free_words"
        ]
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                "reconstruction_by_block": output["reconstruction_by_block"],
                "selection_results": output["selection_results"],
            },
            indent=2,
        )
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
