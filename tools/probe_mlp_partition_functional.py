"""Gate reciprocal MLP factor sharing on held-out MLP outputs and teacher KL.

This analysis-only probe compares separate gate/up/down factors with a
gate/down-transpose shared factor on selected blocks. Dense reconstructions are
held in memory and spliced into the pinned teacher; no compression or runtime
artifact is produced.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_attention_partition_functional import (
    MODEL_SOURCE,
    _aggregate_topologies,
    _capture_attention_outputs,
    _dtype,
    _model_config_hash,
    _relative_output_rmse,
    _teacher_cache_key,
    _token_hash,
)
from probe_factor_grouping import (
    PINNED_MODEL_REVISION,
    PROJECTION_PATHS,
    MaterializedMemberReconstruction,
    MemberSpec,
    ProbeProtocol,
    TopologySpec,
    execute_group_with_reconstruction,
    load_objective_profiles,
    mlp_partition_topologies,
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
from nanoquant.quality_evaluation import _wikitext_tokens

BASELINE_VARIANT = "partition-g-u-d"
CANDIDATE_VARIANTS = ("partition-gd-u", "partition-ud-g", "partition-gud")


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("block list must contain unique non-negative integers")
    return result


def _parse_candidates(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(result) - set(CANDIDATE_VARIANTS))
    if not result or unknown or len(set(result)) != len(result):
        detail = "" if not unknown else f": {', '.join(unknown)}"
        raise argparse.ArgumentTypeError(f"candidate list is empty, duplicated, or unsupported{detail}")
    return result


def _paired_mlp_summary(
    separate: KlBudgetArmResult,
    gate_down: KlBudgetArmResult,
) -> dict[str, float | bool]:
    interval = paired_bootstrap_kl_delta(separate, gate_down)
    return {
        "candidate_minus_separate_kl": interval.point_delta,
        "relative_kl_delta": interval.point_delta / separate.kl_nats_per_token,
        "lower_delta": interval.lower_delta,
        "upper_delta": interval.upper_delta,
        "confidence": interval.confidence,
        "resamples": interval.resamples,
        "improved_with_confidence": interval.point_delta < 0 and interval.upper_delta < 0,
    }


def _topology(block: int, variant: str) -> TopologySpec:
    try:
        return next(item for item in mlp_partition_topologies(block) if item.variant == variant)
    except StopIteration as exc:
        raise ValueError(f"unsupported MLP partition: {variant}") from exc


def _materialize_topology(
    handle: Any,
    blocks: tuple[int, ...],
    variant: str,
    protocol: ProbeProtocol,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cache: dict[
        tuple[MemberSpec, ...],
        tuple[dict[str, Any], tuple[MaterializedMemberReconstruction, ...]],
    ],
) -> tuple[SpliceReconstructionSet, dict[str, Any]]:
    reconstructions: list[SpliceReconstruction] = []
    unit_members: list[tuple[str, tuple[LayerId, ...]]] = []
    unit_errors: list[tuple[str, float]] = []
    block_summaries: dict[str, Any] = {}
    for block in blocks:
        topology = _topology(block, variant)
        group_results = []
        for group in topology.groups:
            cached = cache.get(group.members)
            if cached is None:
                print(f"factorizing {variant} block={block} group={group.label}", flush=True)
                cached = execute_group_with_reconstruction(
                    handle,
                    topology,
                    group,
                    protocol,
                    profiles,
                )
                cache[group.members] = cached
            result, members = cached
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
    if len(reconstructions) != len(blocks) * 3 or len({item.layer for item in reconstructions}) != len(
        reconstructions
    ):
        raise ValueError("functional MLP reconstruction inventory is incomplete")
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


def _measure_isolated_mlp_outputs(
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
        layers = tuple(
            item.layer
            for item in evaluator.reconstructions.layers
            if item.layer.block.index == block
        )
        module = cast(torch.nn.Module, blocks[block].mlp)
        evaluator._install(layers)
        try:
            observed = _capture_attention_outputs(model, {block: module}, tokens, device=device)[block]
        finally:
            evaluator._restore(layers)
        result[str(block)] = _relative_output_rmse(reference[block], observed)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--objective-specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=_parse_ints, default=(10, 15, 20, 22))
    parser.add_argument("--candidates", type=_parse_candidates, default=("partition-gd-u",))
    parser.add_argument("--output-samples", type=int, default=4)
    parser.add_argument("--wikitext-samples", type=int, default=8)
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
    if args.output_samples <= 0 or args.wikitext_samples <= 0 or args.sequence_length < 2:
        raise ValueError("functional probe dataset dimensions must be positive")
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    decoder_count = adapter_for_config(config).decoder_block_count_from_config(config)
    if any(block >= decoder_count for block in args.blocks):
        raise ValueError("functional block lies outside the model")
    protocol = ProbeProtocol(
        2,
        PINNED_MODEL_REVISION,
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
        None,
        0.0,
        str(args.objective_specs.resolve()),
    )
    profiles = load_objective_profiles(args.objective_specs)
    tokens, dataset_fingerprint, _bos = _wikitext_tokens(
        args.snapshot,
        samples=args.wikitext_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    token_hash = _token_hash(tokens)
    with acquire_device_lease(args.device), safe_open(str(args.model), framework="pt", device="cpu") as handle:
        fit_cache: dict[
            tuple[MemberSpec, ...],
            tuple[dict[str, Any], tuple[MaterializedMemberReconstruction, ...]],
        ] = {}
        reconstruction_sets = {}
        reconstruction_metrics = {}
        variants = (BASELINE_VARIANT, *args.candidates)
        for variant in variants:
            reconstruction_sets[variant], reconstruction_metrics[variant] = _materialize_topology(
                handle,
                args.blocks,
                variant,
                protocol,
                profiles,
                fit_cache,
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
        modules = {block: cast(torch.nn.Module, decoder_blocks[block].mlp) for block in args.blocks}
        output_tokens = tokens[: args.output_samples]
        output_reference = _capture_attention_outputs(
            teacher,
            modules,
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
            for variant in variants
        }
        baseline_nll, batches = evaluators[BASELINE_VARIANT].teacher_cache_state()
        for evaluator in evaluators.values():
            evaluator.install_teacher_cache(baseline_nll, batches)
        arms = tuple(f"block:{block}" for block in args.blocks)
        kl_results = {
            variant: {arm: evaluators[variant](arm) for arm in arms}
            for variant in variants
        }
        output_results = {
            variant: _measure_isolated_mlp_outputs(
                evaluators[variant],
                teacher,
                decoder_blocks,
                output_reference,
                output_tokens,
                device=args.device,
            )
            for variant in variants
        }
        cache_key = _teacher_cache_key(
            source=MODEL_SOURCE,
            revision=PINNED_MODEL_REVISION,
            model_hash=_model_config_hash(config),
            token_hash=token_hash,
            model_dtype=model_dtype,
            attention_implementation=adapter.attention_implementation,
            device=args.device,
            batch_size=1,
        )
        del evaluators, teacher
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    comparisons = {
        candidate: {
            arm: _paired_mlp_summary(
                kl_results[BASELINE_VARIANT][arm],
                kl_results[candidate][arm],
            )
            for arm in arms
        }
        for candidate in args.candidates
    }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "role": "analysis-only dense MLP splice; not a compression or runtime artifact",
        "model_source": MODEL_SOURCE,
        "model_revision": PINNED_MODEL_REVISION,
        "protocol": to_dict(protocol),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_slice_hash": token_hash,
        "wikitext_samples": args.wikitext_samples,
        "sequence_length": args.sequence_length,
        "output_samples": args.output_samples,
        "candidate_variants": list(args.candidates),
        "teacher_baseline_nll": baseline_nll,
        "teacher_cache_key": cache_key,
        "reconstruction": reconstruction_metrics,
        "kl": {
            variant: {arm: to_dict(result) for arm, result in results.items()}
            for variant, results in kl_results.items()
        },
        "paired_comparisons": comparisons,
        "isolated_mlp_output_normalized_rmse": output_results,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"paired_comparisons": comparisons, "mlp_output": output_results}, indent=2))
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
