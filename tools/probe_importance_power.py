"""Select a Fisher-importance power exponent by held-out outputs and KL.

This analysis-only probe tempers each raw diagonal Fisher vector with
``importance**alpha`` while preserving that vector's arithmetic mean. It
reconstructs identical fused-QKV complete blocks at fixed bit budgets and
selects against the raw-Fisher ``alpha=1`` endpoint using disjoint WikiText
functional metrics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_factor_grouping import (
    PINNED_MODEL_REVISION,
    PROJECTION_PATHS,
    GroupSpec,
    MemberSpec,
    ProbeProtocol,
    TopologySpec,
    execute_group_with_reconstruction,
    load_calibration_profiles,
    summarize_topology,
)
from probe_importance_shrinkage import (
    _aggregate_summaries,
    _capture_outputs,
    _dtype,
    _isolated_block_outputs,
    _paired_summary,
    _parse_ints,
)
from safetensors import safe_open

from nanoquant.config.codec import to_dict
from nanoquant.domain.models import BlockId, LayerId
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
DEFAULT_EXPONENTS = (0.5, 0.75, 1.0)
BASELINE_EXPONENT = 1.0


def _parse_exponents(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if (
        not result
        or any(not math.isfinite(item) or not 0 <= item <= 1 for item in result)
        or len(set(result)) != len(result)
    ):
        raise argparse.ArgumentTypeError("exponents must be unique finite values in [0, 1]")
    return result


def _exponent_key(value: float) -> str:
    return format(value, ".12g")


def temper_importance(value: torch.Tensor, exponent: float) -> torch.Tensor:
    """Apply a power exponent without changing the vector's arithmetic mean."""

    if not math.isfinite(exponent) or not 0 <= exponent <= 1:
        raise ValueError("importance exponent must be finite and in [0, 1]")
    result = value.detach().float().clone()
    if not torch.isfinite(result).all() or (result < 0).any():
        raise ValueError("importance values must be finite and non-negative")
    if not result.numel() or exponent == 1:
        return result
    original_mean = result.mean()
    if original_mean <= 0:
        return result
    if exponent == 0:
        return torch.full_like(result, float(original_mean.item()))
    result.pow_(exponent)
    return result.mul_(original_mean / result.mean().clamp_min(1e-30))


def load_power_profiles(
    state_directory: Path,
    exponent: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    raw = load_calibration_profiles(state_directory, 0.0)
    return {
        path: (
            temper_importance(input_importance, exponent),
            temper_importance(output_importance, exponent),
        )
        for path, (input_importance, output_importance) in raw.items()
    }


def block_topology(block: int, exponent: float) -> TopologySpec:
    members = {name: MemberSpec(block, name) for name in PROJECTION_PATHS}
    return TopologySpec(
        "importance-power",
        f"power-{_exponent_key(exponent)}",
        str(block),
        (
            GroupSpec("qkv", tuple(members[name] for name in ("q", "k", "v"))),
            *(GroupSpec(name, (members[name],)) for name in ("o", "gate", "up", "down")),
        ),
    )


def _materialize_power(
    handle: Any,
    blocks: tuple[int, ...],
    exponent: float,
    protocol: ProbeProtocol,
    calibration_state: Path,
) -> tuple[SpliceReconstructionSet, dict[str, Any]]:
    profiles = load_power_profiles(calibration_state, exponent)
    reconstructions: list[SpliceReconstruction] = []
    unit_members: list[tuple[str, tuple[LayerId, ...]]] = []
    unit_errors: list[tuple[str, float]] = []
    summaries: dict[str, Any] = {}
    for block in blocks:
        topology = block_topology(block, exponent)
        group_results = []
        for group in topology.groups:
            print(
                f"factorizing exponent={_exponent_key(exponent)} "
                f"block={block} group={group.label}",
                flush=True,
            )
            result, members = execute_group_with_reconstruction(
                handle,
                topology,
                group,
                protocol,
                profiles,
            )
            group_results.append(result)
            layers = []
            member_metrics = cast(list[dict[str, float]], result["scale_fitted"]["members"])
            for item, metrics in zip(members, member_metrics, strict=True):
                layer = LayerId(BlockId(item.member.block), PROJECTION_PATHS[item.member.projection])
                layers.append(layer)
                reconstructions.append(
                    SpliceReconstruction(
                        layer,
                        item.weight,
                        None,
                        float(metrics["normalized_rmse"]) ** 2,
                    )
                )
            unit_id = f"{block}:{group.label}"
            unit_members.append((unit_id, tuple(layers)))
            unit_errors.append((unit_id, float(result["scale_fitted"]["normalized_rmse"]) ** 2))
        summaries[str(block)] = summarize_topology(topology, tuple(group_results))
    if len(reconstructions) != len(blocks) * 7 or len({item.layer for item in reconstructions}) != len(
        reconstructions
    ):
        raise ValueError("importance-power reconstruction inventory is incomplete")
    values = tuple(summaries[str(block)] for block in blocks)
    return (
        SpliceReconstructionSet(
            tuple(reconstructions),
            tuple(unit_members),
            tuple(unit_errors),
        ),
        {
            "aggregate": _aggregate_summaries(values),
            "blocks": summaries,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 24))
    parser.add_argument("--functional-blocks", type=_parse_ints)
    parser.add_argument("--exponents", type=_parse_exponents, default=DEFAULT_EXPONENTS)
    parser.add_argument("--baseline-exponent", type=float, default=BASELINE_EXPONENT)
    parser.add_argument("--block-output-samples", type=int, default=4)
    parser.add_argument("--wikitext-samples", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--target-bpw", type=float, default=1.0)
    parser.add_argument("--rank-alignment", type=int, default=1)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=400)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.baseline_exponent not in args.exponents:
        raise ValueError("baseline exponent must be one of the requested arms")
    if args.block_output_samples <= 0 or args.wikitext_samples <= 0 or args.sequence_length < 2:
        raise ValueError("importance-power probe dataset dimensions must be positive")
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    expected_blocks = adapter_for_config(config).decoder_block_count_from_config(config)
    if any(block >= expected_blocks for block in args.blocks):
        raise ValueError("requested block is outside the model")
    functional_blocks = args.blocks if args.functional_blocks is None else args.functional_blocks
    if not set(functional_blocks).issubset(args.blocks):
        raise ValueError("functional blocks must be included in the reconstruction inventory")
    protocol = ProbeProtocol(
        1,
        args.model_revision,
        args.target_bpw,
        args.rank_alignment,
        args.scale_bits,
        args.outer_iterations,
        args.inner_iterations,
        args.regularization,
        args.penalty_schedule,
        args.convergence_check_interval,
        True,
        args.scale_fit_passes,
        args.seed,
        args.device,
        str(args.calibration_state.resolve()),
        0.0,
    )
    tokens, dataset_fingerprint, _bos = _wikitext_tokens(
        args.snapshot,
        samples=args.wikitext_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    token_hash = _token_hash(tokens)
    with acquire_device_lease(args.device), safe_open(str(args.model), framework="pt", device="cpu") as handle:
        reconstruction_sets = {}
        reconstruction_metrics = {}
        for exponent in args.exponents:
            key = _exponent_key(exponent)
            reconstruction_sets[key], reconstruction_metrics[key] = _materialize_power(
                handle,
                args.blocks,
                exponent,
                protocol,
                args.calibration_state,
            )
        adapter = adapter_for_config(config)
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        decoder_blocks = tuple(adapter.get_decoder_layers(teacher))
        output_tokens = tokens[: args.block_output_samples]
        output_reference = _capture_outputs(
            teacher,
            {block: decoder_blocks[block] for block in functional_blocks},
            output_tokens,
            device=args.device,
        )
        baseline_key = _exponent_key(args.baseline_exponent)
        ordered_keys = (baseline_key, *(key for key in reconstruction_sets if key != baseline_key))
        arms = ("full", *(f"block:{block}" for block in functional_blocks))
        kl_results = {}
        block_outputs = {}
        teacher_batches: tuple[torch.Tensor, ...] | None = None
        baseline_nll = math.nan
        for key in ordered_keys:
            evaluator = DenseKlSpliceEvaluator(
                teacher,
                reconstruction_sets[key],
                tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            if teacher_batches is None:
                baseline_nll, teacher_batches = evaluator.teacher_cache_state()
            else:
                evaluator.install_teacher_cache(baseline_nll, teacher_batches)
            kl_results[key] = {arm: evaluator(arm) for arm in arms}
            block_outputs[key] = _isolated_block_outputs(
                evaluator,
                teacher,
                decoder_blocks,
                output_reference,
                output_tokens,
                device=args.device,
            )
            del evaluator
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del teacher
    comparisons = {
        key: {
            arm: _paired_summary(kl_results[baseline_key][arm], results[arm])
            for arm in arms
        }
        for key, results in kl_results.items()
        if key != baseline_key
    }
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only importance-power selection; not a compression artifact",
            "model_source": MODEL_SOURCE,
            "model_revision": args.model_revision,
            "blocks": list(args.blocks),
            "functional_blocks": list(functional_blocks),
            "exponents": list(args.exponents),
            "baseline_exponent": args.baseline_exponent,
            "mean_preserving": True,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_slice_hash": token_hash,
            "wikitext_samples": args.wikitext_samples,
            "sequence_length": args.sequence_length,
            "block_output_samples": args.block_output_samples,
            "teacher_baseline_nll": baseline_nll,
            "protocol": to_dict(protocol),
            "reconstruction": reconstruction_metrics,
            "kl": {
                key: {arm: to_dict(result) for arm, result in results.items()}
                for key, results in kl_results.items()
            },
            "isolated_block_output_normalized_rmse": block_outputs,
            "paired_comparisons_vs_baseline": comparisons,
        },
    )
    print(
        json.dumps(
            {
                "baseline": baseline_key,
                "paired_comparisons": comparisons,
                "isolated_block_output_normalized_rmse": block_outputs,
            },
            indent=2,
        )
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
