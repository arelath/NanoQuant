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
from nanoquant.domain.mlp_operator_refit import (
    coupled_mlp_output_normalized_rmse,
    fit_coupled_mlp_output_scales,
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


def _module_at_path(block: torch.nn.Module, path: str) -> torch.nn.Module:
    current = block
    for part in path.split("."):
        child = (
            current[part]
            if isinstance(current, torch.nn.ModuleDict)
            else getattr(current, part, None)
        )
        if not isinstance(child, torch.nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current


def _decoder_blocks(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise TypeError("model does not expose a supported decoder block stack")
    return tuple(layers)


@torch.inference_mode()
def _capture_linear_inputs(
    model: torch.nn.Module,
    module: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    captured = []

    def hook(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("MLP projection input hook did not receive a tensor")
        value = inputs[0]
        captured.append(
            value.detach()
            .reshape(-1, value.shape[-1])
            .to(device="cpu", dtype=torch.bfloat16)
        )

    handle = module.register_forward_pre_hook(hook)
    try:
        for index in range(tokens.shape[0]):
            cast(Any, model)(
                input_ids=tokens[index : index + 1].to(device),
                use_cache=False,
            )
    finally:
        handle.remove()
    if len(captured) != tokens.shape[0]:
        raise ValueError("MLP input capture did not cover every requested sequence")
    return torch.cat(captured)


@torch.inference_mode()
def _linear_outputs(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    device: str,
    row_batch: int = 256,
) -> torch.Tensor:
    if inputs.ndim != 2 or weight.ndim != 2 or inputs.shape[1] != weight.shape[1]:
        raise ValueError("operator-refit linear dimensions do not match")
    if row_batch <= 0:
        raise ValueError("operator-refit row batch must be positive")
    weight_device = weight.to(device=device, dtype=torch.bfloat16)
    outputs = []
    for start in range(0, inputs.shape[0], row_batch):
        outputs.append(
            inputs[start : start + row_batch]
            .to(device=device, dtype=torch.bfloat16)
            .matmul(weight_device.mT)
        )
    return torch.cat(outputs)


def _replace_weights(
    reconstructions: SpliceReconstructionSet,
    replacements: dict[LayerId, torch.Tensor],
) -> SpliceReconstructionSet:
    layers = tuple(
        SpliceReconstruction(
            item.layer,
            replacements.get(item.layer, item.weight),
            item.bias,
            item.weighted_normalized_squared_error,
        )
        for item in reconstructions.layers
    )
    if set(replacements) != {
        item.layer for item in layers if item.layer in replacements
    }:
        raise ValueError("operator refit replacement inventory is incomplete")
    return SpliceReconstructionSet(
        layers,
        reconstructions.unit_members,
        reconstructions.unit_weighted_normalized_squared_errors,
    )


def _operator_refit_sets(
    teacher: torch.nn.Module,
    blocks: tuple[int, ...],
    fit_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    reconstruction_sets: dict[str, SpliceReconstructionSet],
    *,
    device: str,
    gate_grid_points: int,
    minimum_gate_multiplier: float,
    maximum_gate_multiplier: float,
    minimum_up_multiplier: float,
    maximum_up_multiplier: float,
) -> tuple[dict[str, SpliceReconstructionSet], dict[str, Any]]:
    decoder = _decoder_blocks(teacher)
    result = dict(reconstruction_sets)
    metrics: dict[str, Any] = {}
    for arm in ("free_words", "corrected_codebook"):
        replacements: dict[LayerId, torch.Tensor] = {}
        arm_metrics: dict[str, Any] = {}
        by_layer = {item.layer: item for item in reconstruction_sets[arm].layers}
        for block_index in blocks:
            gate_id = LayerId(BlockId(block_index), PROJECTION_PATHS["gate"])
            up_id = LayerId(BlockId(block_index), PROJECTION_PATHS["up"])
            gate_item = by_layer.get(gate_id)
            up_item = by_layer.get(up_id)
            if gate_item is None or up_item is None:
                raise ValueError("operator refit requires gate and up reconstructions")
            block = decoder[block_index]
            gate_module = _module_at_path(block, PROJECTION_PATHS["gate"])
            up_module = _module_at_path(block, PROJECTION_PATHS["up"])
            if not isinstance(gate_module, torch.nn.Linear) or not isinstance(
                up_module,
                torch.nn.Linear,
            ):
                raise TypeError("operator refit targets must be dense linear modules")
            fit_inputs = _capture_linear_inputs(
                teacher,
                gate_module,
                fit_tokens,
                device=device,
            )
            validation_inputs = _capture_linear_inputs(
                teacher,
                gate_module,
                validation_tokens,
                device=device,
            )
            teacher_fit_gate = _linear_outputs(
                fit_inputs,
                gate_module.weight,
                device=device,
            )
            teacher_fit_up = _linear_outputs(
                fit_inputs,
                up_module.weight,
                device=device,
            )
            candidate_fit_gate = _linear_outputs(
                fit_inputs,
                gate_item.weight,
                device=device,
            )
            candidate_fit_up = _linear_outputs(
                fit_inputs,
                up_item.weight,
                device=device,
            )
            fitted = fit_coupled_mlp_output_scales(
                teacher_fit_gate,
                teacher_fit_up,
                candidate_fit_gate,
                candidate_fit_up,
                gate_grid_points=gate_grid_points,
                minimum_gate_multiplier=minimum_gate_multiplier,
                maximum_gate_multiplier=maximum_gate_multiplier,
                minimum_up_multiplier=minimum_up_multiplier,
                maximum_up_multiplier=maximum_up_multiplier,
            )
            teacher_validation_gate = _linear_outputs(
                validation_inputs,
                gate_module.weight,
                device=device,
            )
            teacher_validation_up = _linear_outputs(
                validation_inputs,
                up_module.weight,
                device=device,
            )
            candidate_validation_gate = _linear_outputs(
                validation_inputs,
                gate_item.weight,
                device=device,
            )
            candidate_validation_up = _linear_outputs(
                validation_inputs,
                up_item.weight,
                device=device,
            )
            validation_before = coupled_mlp_output_normalized_rmse(
                teacher_validation_gate,
                teacher_validation_up,
                candidate_validation_gate,
                candidate_validation_up,
            )
            validation_after = coupled_mlp_output_normalized_rmse(
                teacher_validation_gate,
                teacher_validation_up,
                candidate_validation_gate,
                candidate_validation_up,
                gate_multiplier=fitted.gate_multiplier,
                up_multiplier=fitted.up_multiplier,
            )
            replacements[gate_id] = (
                gate_item.weight.float()
                * fitted.gate_multiplier.detach().cpu().reshape(-1, 1)
            ).to(torch.bfloat16)
            replacements[up_id] = (
                up_item.weight.float()
                * fitted.up_multiplier.detach().cpu().reshape(-1, 1)
            ).to(torch.bfloat16)
            arm_metrics[str(block_index)] = {
                "fit_before_normalized_rmse": fitted.before_normalized_rmse,
                "fit_after_normalized_rmse": fitted.after_normalized_rmse,
                "validation_before_normalized_rmse": validation_before,
                "validation_after_normalized_rmse": validation_after,
                "validation_change_fraction": (
                    validation_after / validation_before - 1
                ),
                "gate_multiplier_minimum": float(
                    fitted.gate_multiplier.min()
                ),
                "gate_multiplier_maximum": float(
                    fitted.gate_multiplier.max()
                ),
                "up_multiplier_minimum": float(fitted.up_multiplier.min()),
                "up_multiplier_maximum": float(fitted.up_multiplier.max()),
            }
            del (
                fit_inputs,
                validation_inputs,
                teacher_fit_gate,
                teacher_fit_up,
                candidate_fit_gate,
                candidate_fit_up,
                teacher_validation_gate,
                teacher_validation_up,
                candidate_validation_gate,
                candidate_validation_up,
            )
            torch.cuda.empty_cache()
        result[f"{arm}_operator_refit"] = _replace_weights(
            reconstruction_sets[arm],
            replacements,
        )
        metrics[arm] = arm_metrics
    return result, metrics


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


def _paired_payload(
    before: Any,
    after: Any,
    *,
    seed: int,
) -> dict[str, float | int | bool]:
    interval = paired_bootstrap_kl_delta(before, after, seed=seed)
    return {
        "point_delta": interval.point_delta,
        "relative_delta": (
            interval.point_delta / before.kl_nats_per_token
        ),
        "lower_delta": interval.lower_delta,
        "upper_delta": interval.upper_delta,
        "confidence": interval.confidence,
        "resamples": interval.resamples,
        "improved_with_confidence": (
            interval.point_delta < 0 and interval.upper_delta < 0
        ),
    }


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
    parser.add_argument("--operator-scale-refit", action="store_true")
    parser.add_argument("--operator-refit-offset", type=int, default=48)
    parser.add_argument("--operator-refit-samples", type=int, default=4)
    parser.add_argument("--operator-validation-offset", type=int, default=52)
    parser.add_argument("--operator-validation-samples", type=int, default=4)
    parser.add_argument("--operator-gate-grid-points", type=int, default=41)
    parser.add_argument("--operator-minimum-gate-multiplier", type=float, default=0.5)
    parser.add_argument("--operator-maximum-gate-multiplier", type=float, default=1.5)
    parser.add_argument("--operator-minimum-up-multiplier", type=float, default=0.25)
    parser.add_argument("--operator-maximum-up-multiplier", type=float, default=4.0)
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
    if args.operator_scale_refit and set(projections) != {"gate", "up"}:
        raise ValueError("operator scale refit requires gate and up projections")
    if (
        args.operator_refit_offset < 0
        or args.operator_refit_samples <= 0
        or args.operator_validation_offset < 0
        or args.operator_validation_samples <= 0
        or args.operator_gate_grid_points < 2
        or not 0
        < args.operator_minimum_gate_multiplier
        <= 1
        <= args.operator_maximum_gate_multiplier
        or not 0
        < args.operator_minimum_up_multiplier
        <= 1
        <= args.operator_maximum_up_multiplier
    ):
        raise ValueError("operator scale-refit dataset settings are invalid")
    fit_inventory = set(
        range(
            args.operator_refit_offset,
            args.operator_refit_offset + args.operator_refit_samples,
        )
    )
    validation_inventory = set(
        range(
            args.operator_validation_offset,
            args.operator_validation_offset
            + args.operator_validation_samples,
        )
    )
    evaluation_inventory = set(
        range(
            args.wikitext_offset,
            args.wikitext_offset + args.wikitext_samples,
        )
    )
    if args.operator_scale_refit and (
        fit_inventory & validation_inventory
        or fit_inventory & evaluation_inventory
        or validation_inventory & evaluation_inventory
    ):
        raise ValueError(
            "operator fit, validation, and KL evaluation windows must be disjoint"
        )
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

        required_samples = args.wikitext_offset + args.wikitext_samples
        if args.operator_scale_refit:
            required_samples = max(
                required_samples,
                args.operator_refit_offset + args.operator_refit_samples,
                args.operator_validation_offset
                + args.operator_validation_samples,
            )
        all_tokens, dataset_fingerprint, _bos = _wikitext_tokens(
            args.snapshot,
            samples=required_samples,
            sequence_length=args.sequence_length,
            local_files_only=args.local_files_only,
        )
        tokens = _select_token_window(
            all_tokens,
            offset=args.wikitext_offset,
            samples=args.wikitext_samples,
        )
        operator_fit_tokens = (
            _select_token_window(
                all_tokens,
                offset=args.operator_refit_offset,
                samples=args.operator_refit_samples,
            )
            if args.operator_scale_refit
            else None
        )
        operator_validation_tokens = (
            _select_token_window(
                all_tokens,
                offset=args.operator_validation_offset,
                samples=args.operator_validation_samples,
            )
            if args.operator_scale_refit
            else None
        )
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        operator_refit_metrics: dict[str, Any] = {}
        if args.operator_scale_refit:
            assert operator_fit_tokens is not None
            assert operator_validation_tokens is not None
            all_reconstruction_sets, operator_refit_metrics = (
                _operator_refit_sets(
                    teacher,
                    blocks,
                    operator_fit_tokens,
                    operator_validation_tokens,
                    all_reconstruction_sets,
                    device=args.device,
                    gate_grid_points=args.operator_gate_grid_points,
                    minimum_gate_multiplier=(
                        args.operator_minimum_gate_multiplier
                    ),
                    maximum_gate_multiplier=(
                        args.operator_maximum_gate_multiplier
                    ),
                    minimum_up_multiplier=(
                        args.operator_minimum_up_multiplier
                    ),
                    maximum_up_multiplier=(
                        args.operator_maximum_up_multiplier
                    ),
                )
            )
        teacher_nll = 0.0
        teacher_cache: tuple[torch.Tensor, ...] = ()
        selection_results: dict[str, dict[str, Any]] = {}
        for name, threshold, selected_blocks in selection_specs:
            reconstruction_sets = {
                arm: _select_blocks(reconstructions, selected_blocks)
                for arm, reconstructions in all_reconstruction_sets.items()
            }
            kl_results = {}
            for arm, reconstructions in reconstruction_sets.items():
                evaluator = DenseKlSpliceEvaluator(
                    teacher,
                    reconstructions,
                    tokens,
                    device=args.device,
                    batch_size=1,
                    token_chunk_size=128,
                    teacher_cache_mode="cpu",
                )
                if not teacher_cache:
                    teacher_nll, teacher_cache = evaluator.teacher_cache_state()
                else:
                    evaluator.install_teacher_cache(
                        teacher_nll,
                        teacher_cache,
                    )
                kl_results[arm] = evaluator("full")
                del evaluator
            baseline_kl = kl_results["free_words"]
            candidate_kl = kl_results["corrected_codebook"]
            selection_result: dict[str, Any] = {
                "minimum_weighted_rmse_gain_fraction": threshold,
                "blocks": list(selected_blocks),
                "kl": {
                    arm: to_dict(value) for arm, value in kl_results.items()
                },
                "paired_candidate_minus_free_words": _paired_payload(
                    baseline_kl,
                    candidate_kl,
                    seed=args.seed,
                ),
            }
            if args.operator_scale_refit:
                baseline_refit = kl_results["free_words_operator_refit"]
                candidate_refit = kl_results[
                    "corrected_codebook_operator_refit"
                ]
                selection_result[
                    "paired_operator_candidate_minus_operator_free_words"
                ] = _paired_payload(
                    baseline_refit,
                    candidate_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_free_words_refit_minus_free_words"
                ] = _paired_payload(
                    baseline_kl,
                    baseline_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_candidate_refit_minus_candidate"
                ] = _paired_payload(
                    candidate_kl,
                    candidate_refit,
                    seed=args.seed,
                )
            selection_results[name] = selection_result
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
        "schema_version": 6,
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
        "operator_scale_refit": {
            "enabled": args.operator_scale_refit,
            "fit_offset": args.operator_refit_offset,
            "fit_samples": args.operator_refit_samples,
            "validation_offset": args.operator_validation_offset,
            "validation_samples": args.operator_validation_samples,
            "gate_grid_points": args.operator_gate_grid_points,
            "minimum_gate_multiplier": (
                args.operator_minimum_gate_multiplier
            ),
            "maximum_gate_multiplier": (
                args.operator_maximum_gate_multiplier
            ),
            "minimum_up_multiplier": args.operator_minimum_up_multiplier,
            "maximum_up_multiplier": args.operator_maximum_up_multiplier,
            "metrics": operator_refit_metrics,
        },
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
