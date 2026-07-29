"""Select Fisher-importance shrinkage by held-out block outputs and teacher KL.

The production recipe uses 0.6 linear shrinkage toward each importance
vector's mean. This analysis-only probe reconstructs representative complete
blocks at several shrinkages under identical bit budgets and ADMM seeds, then
selects using disjoint WikiText functional metrics rather than in-sample
weighted matrix error.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from contextlib import ExitStack
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
from safetensors import safe_open

from nanoquant.application.kl_budget import KlBudgetArmResult, paired_bootstrap_kl_delta
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
DEFAULT_SHRINKAGES = (0.0, 0.3, 0.6, 0.8, 0.9, 1.0)
BASELINE_SHRINKAGE = 0.6


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("blocks must be unique non-negative integers")
    return result


def _parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if (
        not result
        or any(not math.isfinite(item) or not 0 <= item <= 1 for item in result)
        or len(set(result)) != len(result)
    ):
        raise argparse.ArgumentTypeError("shrinkages must be unique finite values in [0, 1]")
    return result


def _shrinkage_key(value: float) -> str:
    return format(value, ".12g")


def _dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(cast(str, value), torch.float32)


def block_topology(block: int, shrinkage: float) -> TopologySpec:
    members = {name: MemberSpec(block, name) for name in PROJECTION_PATHS}
    return TopologySpec(
        "importance-shrinkage",
        f"shrink-{_shrinkage_key(shrinkage)}",
        str(block),
        (
            GroupSpec("qkv", tuple(members[name] for name in ("q", "k", "v"))),
            *(GroupSpec(name, (members[name],)) for name in ("o", "gate", "up", "down")),
        ),
    )


def _aggregate_summaries(values: tuple[dict[str, Any], ...]) -> dict[str, float | int]:
    error = math.fsum(float(value["error_energy"]) for value in values)
    energy = math.fsum(float(value["target_energy"]) for value in values)
    original_error = math.fsum(float(value["original_error_energy"]) for value in values)
    original_energy = math.fsum(float(value["original_target_energy"]) for value in values)
    source_elements = sum(int(value["source_elements"]) for value in values)
    actual_bits = sum(int(value["actual_bits"]) for value in values)
    return {
        "objective_normalized_rmse": math.sqrt(error / max(energy, 1e-30)),
        "original_normalized_rmse": math.sqrt(original_error / max(original_energy, 1e-30)),
        "source_elements": source_elements,
        "actual_bits": actual_bits,
        "actual_bpw": actual_bits / source_elements,
    }


def _materialize_shrinkage(
    handle: Any,
    blocks: tuple[int, ...],
    shrinkage: float,
    base_protocol: ProbeProtocol,
    calibration_state: Path,
) -> tuple[SpliceReconstructionSet, dict[str, Any]]:
    protocol = ProbeProtocol(
        base_protocol.schema_version,
        base_protocol.model_revision,
        base_protocol.target_bpw,
        base_protocol.rank_alignment,
        base_protocol.scale_bits,
        base_protocol.outer_iterations,
        base_protocol.inner_iterations,
        base_protocol.regularization,
        base_protocol.penalty_schedule,
        base_protocol.convergence_check_interval,
        base_protocol.transpose_wide,
        base_protocol.scale_fit_passes,
        base_protocol.seed,
        base_protocol.device,
        base_protocol.calibration_state,
        shrinkage,
    )
    profiles = load_calibration_profiles(calibration_state, shrinkage)
    reconstructions: list[SpliceReconstruction] = []
    unit_members: list[tuple[str, tuple[LayerId, ...]]] = []
    unit_errors: list[tuple[str, float]] = []
    summaries: dict[str, Any] = {}
    for block in blocks:
        topology = block_topology(block, shrinkage)
        group_results = []
        for group in topology.groups:
            print(
                f"factorizing shrinkage={_shrinkage_key(shrinkage)} "
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
        raise ValueError("shrinkage reconstruction inventory is incomplete")
    block_values = tuple(summaries[str(block)] for block in blocks)
    return (
        SpliceReconstructionSet(
            tuple(reconstructions),
            tuple(unit_members),
            tuple(unit_errors),
        ),
        {
            "protocol": to_dict(protocol),
            "aggregate": _aggregate_summaries(block_values),
            "blocks": summaries,
        },
    )


def _output_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError("decoder block did not return a tensor or tensor-first sequence")


@torch.inference_mode()
def _capture_outputs(
    model: torch.nn.Module,
    modules: dict[int, torch.nn.Module],
    tokens: torch.Tensor,
    *,
    device: str,
) -> dict[int, tuple[torch.Tensor, ...]]:
    captured: dict[int, list[torch.Tensor]] = {block: [] for block in modules}
    with ExitStack() as stack:
        for block, module in modules.items():
            handle = module.register_forward_hook(
                lambda _module, _inputs, output, block=block: captured[block].append(
                    _output_tensor(output).detach().to(device="cpu", dtype=torch.bfloat16)
                )
            )
            stack.callback(handle.remove)
        for index in range(tokens.shape[0]):
            cast(Any, model)(input_ids=tokens[index : index + 1].to(device), use_cache=False)
    if any(len(values) != tokens.shape[0] for values in captured.values()):
        raise ValueError("block-output capture did not cover every requested sequence")
    return {block: tuple(values) for block, values in captured.items()}


def _relative_rmse(reference: tuple[torch.Tensor, ...], candidate: tuple[torch.Tensor, ...]) -> float:
    if len(reference) != len(candidate):
        raise ValueError("block-output sequence inventories differ")
    error = 0.0
    energy = 0.0
    for expected, observed in zip(reference, candidate, strict=True):
        if expected.shape != observed.shape:
            raise ValueError("block-output tensor shapes differ")
        error += float((observed.float() - expected.float()).square().sum())
        energy += float(expected.float().square().sum())
    return math.sqrt(error / max(energy, 1e-30))


def _isolated_block_outputs(
    evaluator: DenseKlSpliceEvaluator,
    model: torch.nn.Module,
    decoder_blocks: tuple[torch.nn.Module, ...],
    reference: dict[int, tuple[torch.Tensor, ...]],
    tokens: torch.Tensor,
    *,
    device: str,
) -> dict[str, float]:
    result = {}
    for block in reference:
        layers = tuple(
            item.layer
            for item in evaluator.reconstructions.layers
            if item.layer.block.index == block
        )
        evaluator._install(layers)
        try:
            observed = _capture_outputs(
                model,
                {block: decoder_blocks[block]},
                tokens,
                device=device,
            )[block]
        finally:
            evaluator._restore(layers)
        result[str(block)] = _relative_rmse(reference[block], observed)
    return result


def _paired_summary(
    baseline: KlBudgetArmResult,
    candidate: KlBudgetArmResult,
) -> dict[str, float | bool | int]:
    interval = paired_bootstrap_kl_delta(baseline, candidate)
    return {
        "candidate_minus_baseline_kl": interval.point_delta,
        "relative_kl_delta": interval.point_delta / baseline.kl_nats_per_token,
        "lower_delta": interval.lower_delta,
        "upper_delta": interval.upper_delta,
        "confidence": interval.confidence,
        "resamples": interval.resamples,
        "improved_with_confidence": interval.point_delta < 0 and interval.upper_delta < 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 24))
    parser.add_argument("--functional-blocks", type=_parse_ints)
    parser.add_argument("--shrinkages", type=_parse_floats, default=DEFAULT_SHRINKAGES)
    parser.add_argument("--baseline-shrinkage", type=float, default=BASELINE_SHRINKAGE)
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
    if args.baseline_shrinkage not in args.shrinkages:
        raise ValueError("baseline shrinkage must be one of the requested arms")
    if args.block_output_samples <= 0 or args.wikitext_samples <= 0 or args.sequence_length < 2:
        raise ValueError("shrinkage probe dataset dimensions must be positive")
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
    base_protocol = ProbeProtocol(
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
        args.baseline_shrinkage,
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
        for shrinkage in args.shrinkages:
            key = _shrinkage_key(shrinkage)
            reconstruction_sets[key], reconstruction_metrics[key] = _materialize_shrinkage(
                handle,
                args.blocks,
                shrinkage,
                base_protocol,
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
        baseline_key = _shrinkage_key(args.baseline_shrinkage)
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
            "role": "analysis-only shrinkage selection; not a compression artifact",
            "model_source": MODEL_SOURCE,
            "model_revision": args.model_revision,
            "blocks": list(args.blocks),
            "functional_blocks": list(functional_blocks),
            "shrinkages": list(args.shrinkages),
            "baseline_shrinkage": args.baseline_shrinkage,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_slice_hash": token_hash,
            "wikitext_samples": args.wikitext_samples,
            "sequence_length": args.sequence_length,
            "block_output_samples": args.block_output_samples,
            "teacher_baseline_nll": baseline_nll,
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
