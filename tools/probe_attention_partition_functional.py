"""Measure attention factor partitions on held-out outputs and teacher KL.

This analysis-only probe materializes the fitted dense weights for two attention
partition topologies, splices them into the pinned teacher, and reports paired
held-out functional metrics. It does not create compression/runtime artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_factor_grouping import (
    PINNED_MODEL_REVISION,
    PROJECTION_PATHS,
    ProbeProtocol,
    TopologySpec,
    attention_partition_topologies,
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
from nanoquant.infrastructure.kl_teacher_cache import load_active_kl_teacher_cache
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _model_config_hash, _teacher_cache_key, _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

MODEL_SOURCE = "google/gemma-3-1b-it"
TOPOLOGY_VARIANTS = ("partition-qkv-o", "partition-qv-ko")


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("block list must contain unique non-negative integers")
    return result


def _dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(cast(str, value), torch.float32)


def _topology(block: int, variant: str) -> TopologySpec:
    try:
        return next(item for item in attention_partition_topologies(block) if item.variant == variant)
    except StopIteration as exc:
        raise ValueError(f"unsupported attention partition: {variant}") from exc


def _aggregate_topologies(blocks: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    values = tuple(blocks)
    error = math.fsum(float(item["error_energy"]) for item in values)
    energy = math.fsum(float(item["target_energy"]) for item in values)
    original_error = math.fsum(float(item["original_error_energy"]) for item in values)
    original_energy = math.fsum(float(item["original_target_energy"]) for item in values)
    source_elements = sum(int(item["source_elements"]) for item in values)
    actual_bits = sum(int(item["actual_bits"]) for item in values)
    return {
        "weighted_normalized_rmse": math.sqrt(error / max(energy, 1e-30)),
        "original_normalized_rmse": math.sqrt(original_error / max(original_energy, 1e-30)),
        "source_elements": source_elements,
        "actual_bits": actual_bits,
        "actual_bpw": actual_bits / source_elements,
    }


def _materialize_topology(
    handle: Any,
    blocks: tuple[int, ...],
    variant: str,
    protocol: ProbeProtocol,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[SpliceReconstructionSet, dict[str, Any]]:
    reconstructions: list[SpliceReconstruction] = []
    unit_members: list[tuple[str, tuple[LayerId, ...]]] = []
    unit_errors: list[tuple[str, float]] = []
    block_summaries: dict[str, Any] = {}
    for block in blocks:
        topology = _topology(block, variant)
        group_results = []
        for group in topology.groups:
            print(f"factorizing {variant} block={block} group={group.label}", flush=True)
            result, members = execute_group_with_reconstruction(
                handle,
                topology,
                group,
                protocol,
                profiles,
            )
            group_results.append(result)
            layers = []
            for item in members:
                layer = LayerId(BlockId(item.member.block), PROJECTION_PATHS[item.member.projection])
                layers.append(layer)
                reconstructions.append(
                    SpliceReconstruction(
                        layer,
                        item.weight,
                        None,
                        float(result["scale_fitted"]["normalized_rmse"]) ** 2,
                    )
                )
            unit_id = f"{block}:{group.label}"
            unit_members.append((unit_id, tuple(layers)))
            unit_errors.append((unit_id, float(result["scale_fitted"]["normalized_rmse"]) ** 2))
        block_summaries[str(block)] = summarize_topology(topology, tuple(group_results))
    if len(reconstructions) != len(blocks) * 4 or len({item.layer for item in reconstructions}) != len(
        reconstructions
    ):
        raise ValueError("functional partition reconstruction inventory is incomplete")
    return (
        SpliceReconstructionSet(
            tuple(reconstructions),
            tuple(unit_members),
            tuple(unit_errors),
        ),
        {
            "aggregate": _aggregate_topologies(block_summaries.values()),
            "blocks": block_summaries,
        },
    )


def _output_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError("attention module did not return a tensor or tensor-first sequence")


@torch.inference_mode()
def _capture_attention_outputs(
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
        raise ValueError("attention-output capture did not cover every requested sequence")
    return {block: tuple(values) for block, values in captured.items()}


def _relative_output_rmse(
    reference: tuple[torch.Tensor, ...],
    candidate: tuple[torch.Tensor, ...],
) -> float:
    if len(reference) != len(candidate):
        raise ValueError("attention-output sequence inventories differ")
    error = 0.0
    energy = 0.0
    for expected, observed in zip(reference, candidate, strict=True):
        if expected.shape != observed.shape:
            raise ValueError("attention-output tensor shapes differ")
        error += float((observed.float() - expected.float()).square().sum())
        energy += float(expected.float().square().sum())
    return math.sqrt(error / max(energy, 1e-30))


def _measure_isolated_attention_outputs(
    evaluator: DenseKlSpliceEvaluator,
    model: torch.nn.Module,
    blocks: tuple[torch.nn.Module, ...],
    reference: dict[int, tuple[torch.Tensor, ...]],
    tokens: torch.Tensor,
    *,
    device: str,
) -> dict[str, float]:
    result = {}
    for block in reference:
        selected = tuple(layer for layer in evaluator.reconstructions.layers if layer.layer.block.index == block)
        layers = tuple(item.layer for item in selected)
        module = cast(torch.nn.Module, blocks[block].self_attn)
        evaluator._install(layers)
        try:
            observed = _capture_attention_outputs(model, {block: module}, tokens, device=device)[block]
        finally:
            evaluator._restore(layers)
        result[str(block)] = _relative_output_rmse(reference[block], observed)
    return result


def _paired_summary(before: KlBudgetArmResult, after: KlBudgetArmResult) -> dict[str, float | bool]:
    interval = paired_bootstrap_kl_delta(before, after)
    return {
        "qv_ko_minus_qkv_o_kl": interval.point_delta,
        "relative_kl_delta": interval.point_delta / before.kl_nats_per_token,
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
    parser.add_argument("--teacher-cache-root", type=Path)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--blocks", type=_parse_ints, default=tuple(range(26)))
    parser.add_argument("--functional-blocks", type=_parse_ints, default=(7, 17))
    parser.add_argument("--attention-output-samples", type=int, default=4)
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
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.attention_output_samples <= 0 or args.wikitext_samples <= 0 or args.sequence_length < 2:
        raise ValueError("functional probe dataset dimensions must be positive")
    if not set(args.functional_blocks).issubset(args.blocks):
        raise ValueError("functional blocks must be included in the reconstructed block inventory")
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    expected_blocks = adapter_for_config(config).decoder_block_count_from_config(config)
    if args.blocks != tuple(range(expected_blocks)):
        raise ValueError("whole-attention KL comparison requires the complete contiguous block inventory")
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
        args.calibration_shrinkage,
    )
    profiles = load_calibration_profiles(args.calibration_state, args.calibration_shrinkage)
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
        for variant in TOPOLOGY_VARIANTS:
            reconstruction_sets[variant], reconstruction_metrics[variant] = _materialize_topology(
                handle,
                args.blocks,
                variant,
                protocol,
                profiles,
            )
        adapter = adapter_for_config(config)
        model_dtype = _dtype(config)
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=model_dtype,
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        decoder_blocks = tuple(adapter.get_decoder_layers(teacher))
        attention_modules = {
            block: cast(torch.nn.Module, decoder_blocks[block].self_attn)
            for block in args.functional_blocks
        }
        output_tokens = tokens[: args.attention_output_samples]
        output_reference = _capture_attention_outputs(
            teacher,
            attention_modules,
            output_tokens,
            device=args.device,
        )
        evaluators = {
            variant: DenseKlSpliceEvaluator(
                teacher,
                reconstruction_sets[variant],
                tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            for variant in TOPOLOGY_VARIANTS
        }
        cache = None
        cache_reused = False
        cache_key = _teacher_cache_key(
            source=MODEL_SOURCE,
            revision=args.model_revision,
            model_hash=_model_config_hash(config),
            token_hash=token_hash,
            model_dtype=model_dtype,
            attention_implementation=adapter.attention_implementation,
            device=args.device,
            batch_size=1,
        )
        if args.teacher_cache_root is not None:
            cache = load_active_kl_teacher_cache(args.teacher_cache_root, cache_key)
            cache_reused = cache is not None
        if cache is not None:
            for evaluator in evaluators.values():
                evaluator.install_teacher_cache(cache.baseline_negative_log_likelihood, cache.batches)
        else:
            baseline_nll, batches = evaluators[TOPOLOGY_VARIANTS[0]].teacher_cache_state()
            for evaluator in evaluators.values():
                evaluator.install_teacher_cache(baseline_nll, batches)
        arms = ("full", *(f"block:{block}" for block in args.functional_blocks))
        kl_results = {
            variant: {arm: evaluators[variant](arm) for arm in arms}
            for variant in TOPOLOGY_VARIANTS
        }
        attention_output = {
            variant: _measure_isolated_attention_outputs(
                evaluators[variant],
                teacher,
                decoder_blocks,
                output_reference,
                output_tokens,
                device=args.device,
            )
            for variant in TOPOLOGY_VARIANTS
        }
        baseline_nll = evaluators[TOPOLOGY_VARIANTS[0]].baseline_negative_log_likelihood
        del evaluators, teacher
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    comparisons = {
        arm: _paired_summary(
            kl_results["partition-qkv-o"][arm],
            kl_results["partition-qv-ko"][arm],
        )
        for arm in arms
    }
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only dense splice; not a compression or runtime artifact",
            "model_source": MODEL_SOURCE,
            "model_revision": args.model_revision,
            "protocol": to_dict(protocol),
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_slice_hash": token_hash,
            "wikitext_samples": args.wikitext_samples,
            "sequence_length": args.sequence_length,
            "attention_output_samples": args.attention_output_samples,
            "teacher_baseline_nll": baseline_nll,
            "teacher_cache_key": cache_key,
            "teacher_cache_reused": cache_reused,
            "reconstruction": reconstruction_metrics,
            "kl": {
                variant: {arm: to_dict(result) for arm, result in results.items()}
                for variant, results in kl_results.items()
            },
            "paired_comparisons": comparisons,
            "isolated_attention_output_normalized_rmse": attention_output,
        },
    )
    print(json.dumps({"paired_comparisons": comparisons, "attention_output": attention_output}, indent=2))
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
