"""Functionally compare compute-matched control and tabu real-block weights.

This analysis-only probe consumes the immutable dense weights emitted by
``probe_real_block_tabu.py``.  It evaluates identical control/tabu splice
inventories against a BF16 teacher on the retained WikiText protocol, both one
tested block at a time and with all tested blocks composed.  It also measures
isolated decoder-block output error on a bounded prefix.  No source run or
compression artifact is mutated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from safetensors.torch import load_file

from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    paired_bootstrap_kl_delta,
)
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

MODEL_SOURCE = "google/gemma-3-1b-it"
PINNED_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"


def _dtype(config: dict[str, object]) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(cast(str, config.get("torch_dtype")), torch.float32)


def _parse_member(value: str) -> LayerId:
    block_text, separator, path = value.partition(":")
    if not separator or not path:
        raise ValueError(f"invalid retained-owner member identity: {value!r}")
    return LayerId(BlockId(int(block_text)), path)


def _reconstruction_sets(
    probe: dict[str, Any],
    weights: dict[str, torch.Tensor],
    selected_owners: tuple[str, ...] | None = None,
) -> tuple[dict[str, SpliceReconstructionSet], tuple[int, ...], tuple[str, ...]]:
    results = probe.get("results")
    if probe.get("status") != "completed" or not isinstance(results, list) or not results:
        raise ValueError("real-block tabu probe is not complete")
    by_variant: dict[str, list[SpliceReconstruction]] = {"control": [], "tabu": []}
    units: list[tuple[str, tuple[LayerId, ...]]] = []
    errors: dict[str, list[tuple[str, float]]] = {"control": [], "tabu": []}
    blocks: list[int] = []
    owners: list[str] = []
    seen_layers: set[LayerId] = set()
    for record in results:
        block = int(record["block"])
        owner = str(record["owner"])
        unit_id = f"{block}:{owner}"
        if selected_owners is not None and unit_id not in selected_owners:
            continue
        members = tuple(_parse_member(str(value)) for value in record["members"])
        if not members or any(member.block.index != block for member in members):
            raise ValueError(f"owner {block}:{owner} has an invalid member inventory")
        if seen_layers.intersection(members):
            raise ValueError("real-block tabu probe contains duplicate logical layers")
        seen_layers.update(members)
        units.append((unit_id, members))
        blocks.append(block)
        owners.append(unit_id)
        for variant in by_variant:
            normalized_squared_error = float(record[f"{variant}_nrmse"]) ** 2
            errors[variant].append((unit_id, normalized_squared_error))
            for member in members:
                key = f"{variant}.block_{block}.{member.path}"
                weight = weights.get(key)
                if weight is None:
                    raise ValueError(f"dense tabu weight inventory is missing {key}")
                by_variant[variant].append(
                    SpliceReconstruction(
                        member,
                        weight,
                        None,
                        normalized_squared_error,
                    )
                )
    if selected_owners is not None and set(owners) != set(selected_owners):
        raise ValueError("selected functional owner is absent from the real-block probe")
    expected_keys = {
        f"{variant}.block_{item.layer.block.index}.{item.layer.path}"
        for variant, items in by_variant.items()
        for item in items
    }
    missing_keys = expected_keys - set(weights)
    if missing_keys or (selected_owners is None and set(weights) != expected_keys):
        raise ValueError("dense tabu weight inventory is missing selected tensors")
    return (
        {
            variant: SpliceReconstructionSet(
                tuple(items),
                tuple(units),
                tuple(errors[variant]),
            )
            for variant, items in by_variant.items()
        },
        tuple(dict.fromkeys(blocks)),
        tuple(owners),
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
        raise ValueError("decoder-output capture did not cover every requested sequence")
    return {block: tuple(values) for block, values in captured.items()}


def _relative_rmse(
    reference: tuple[torch.Tensor, ...],
    candidate: tuple[torch.Tensor, ...],
) -> float:
    if len(reference) != len(candidate):
        raise ValueError("decoder-output sequence inventories differ")
    error = 0.0
    energy = 0.0
    for expected, observed in zip(reference, candidate, strict=True):
        if expected.shape != observed.shape:
            raise ValueError("decoder-output tensor shapes differ")
        error += float((observed.float() - expected.float()).square().sum())
        energy += float(expected.float().square().sum())
    return math.sqrt(error / max(energy, 1e-30))


def _block_output_errors(
    evaluator: DenseKlSpliceEvaluator,
    model: torch.nn.Module,
    decoder_blocks: tuple[torch.nn.Module, ...],
    reference: dict[int, tuple[torch.Tensor, ...]],
    tokens: torch.Tensor,
    *,
    device: str,
) -> dict[str, float]:
    errors = {}
    for block, expected in reference.items():
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
        errors[str(block)] = _relative_rmse(expected, observed)
    return errors


def _owner_output_errors(
    evaluator: DenseKlSpliceEvaluator,
    model: torch.nn.Module,
    decoder_blocks: tuple[torch.nn.Module, ...],
    reference: dict[int, tuple[torch.Tensor, ...]],
    tokens: torch.Tensor,
    owners: tuple[str, ...],
    *,
    device: str,
) -> dict[str, float]:
    errors = {}
    for owner in owners:
        block = int(owner.split(":", 1)[0])
        layers = evaluator._selected_layers(f"unit:{owner}")
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
        errors[owner] = _relative_rmse(reference[block], observed)
    return errors


def _paired_summary(
    baseline: KlBudgetArmResult,
    candidate: KlBudgetArmResult,
) -> dict[str, float | bool | int]:
    interval = paired_bootstrap_kl_delta(baseline, candidate)
    return {
        "candidate_minus_control_kl": interval.point_delta,
        "relative_kl_delta": interval.point_delta
        / max(baseline.kl_nats_per_token, 1e-30),
        "lower_delta": interval.lower_delta,
        "upper_delta": interval.upper_delta,
        "confidence": interval.confidence,
        "resamples": interval.resamples,
        "improved_with_confidence": interval.point_delta < 0 and interval.upper_delta < 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wikitext-samples", type=int, default=8)
    parser.add_argument("--wikitext-offset", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--output-samples", type=int, default=4)
    parser.add_argument("--include-owner-arms", action="store_true")
    parser.add_argument(
        "--select-owners",
        help="comma-separated block:owner identities to evaluate",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if (
        args.wikitext_samples <= 0
        or args.wikitext_offset < 0
        or args.output_samples <= 0
        or args.sequence_length < 2
    ):
        raise ValueError("functional tabu probe dimensions must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    weight_tensors = load_file(args.weights, device="cpu")
    selected_owners = (
        tuple(item.strip() for item in args.select_owners.split(",") if item.strip())
        if args.select_owners
        else None
    )
    if selected_owners is not None and (
        not selected_owners or len(set(selected_owners)) != len(selected_owners)
    ):
        raise ValueError("selected functional owners must be unique non-empty identities")
    reconstructions, blocks, owners = _reconstruction_sets(
        probe,
        weight_tensors,
        selected_owners,
    )
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    adapter = adapter_for_config(config)
    expected_blocks = adapter.decoder_block_count_from_config(config)
    if any(block >= expected_blocks for block in blocks):
        raise ValueError("functional tabu block lies outside the model")
    tokens, dataset_fingerprint, bos_token_id = _wikitext_tokens(
        args.snapshot,
        samples=args.wikitext_offset + args.wikitext_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    tokens = tokens[args.wikitext_offset :]
    token_hash = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "role": "analysis-only real-block tabu functional validation",
        "source_probe": str(args.probe.resolve()),
        "source_weights": str(args.weights.resolve()),
        "model_source": MODEL_SOURCE,
        "model_revision": PINNED_MODEL_REVISION,
        "blocks": list(blocks),
        "owners": list(owners),
        "protocol": {
            "wikitext_samples": args.wikitext_samples,
            "wikitext_offset": args.wikitext_offset,
            "sequence_length": args.sequence_length,
            "output_samples": args.output_samples,
            "include_owner_arms": args.include_owner_arms,
            "device": args.device,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_slice_hash": token_hash,
        "bos_token_id": bos_token_id,
    }
    atomic_write_json(args.output, output)
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        decoder_blocks = tuple(adapter.get_decoder_layers(teacher))
        output_tokens = tokens[: args.output_samples]
        reference = _capture_outputs(
            teacher,
            {block: decoder_blocks[block] for block in blocks},
            output_tokens,
            device=args.device,
        )
        evaluators = {
            variant: DenseKlSpliceEvaluator(
                teacher,
                reconstruction,
                tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            for variant, reconstruction in reconstructions.items()
        }
        baseline_nll, teacher_batches = evaluators["control"].teacher_cache_state()
        evaluators["tabu"].install_teacher_cache(baseline_nll, teacher_batches)
        arms = ["full", *(f"block:{block}" for block in blocks)]
        if args.include_owner_arms:
            arms.extend(f"unit:{owner}" for owner in owners)
        kl_results = {
            variant: {arm: evaluator(arm) for arm in arms}
            for variant, evaluator in evaluators.items()
        }
        block_outputs = {
            variant: _block_output_errors(
                evaluator,
                teacher,
                decoder_blocks,
                reference,
                output_tokens,
                device=args.device,
            )
            for variant, evaluator in evaluators.items()
        }
        owner_outputs = (
            {
                variant: _owner_output_errors(
                    evaluator,
                    teacher,
                    decoder_blocks,
                    reference,
                    output_tokens,
                    owners,
                    device=args.device,
                )
                for variant, evaluator in evaluators.items()
            }
            if args.include_owner_arms
            else {}
        )
        del evaluators, teacher
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    comparisons = {
        arm: _paired_summary(kl_results["control"][arm], kl_results["tabu"][arm])
        for arm in arms
    }
    output.update(
        {
            "status": "completed",
            "teacher_baseline_nll": baseline_nll,
            "kl": {
                variant: {arm: to_dict(result) for arm, result in values.items()}
                for variant, values in kl_results.items()
            },
            "paired_comparisons": comparisons,
            "isolated_block_output_normalized_rmse": block_outputs,
            "isolated_block_output_comparisons": {
                str(block): {
                    "tabu_minus_control": block_outputs["tabu"][str(block)]
                    - block_outputs["control"][str(block)],
                    "relative_delta": block_outputs["tabu"][str(block)]
                    / max(block_outputs["control"][str(block)], 1e-30)
                    - 1.0,
                }
                for block in blocks
            },
            "isolated_owner_block_output_normalized_rmse": owner_outputs,
            "isolated_owner_block_output_comparisons": {
                owner: {
                    "tabu_minus_control": owner_outputs["tabu"][owner]
                    - owner_outputs["control"][owner],
                    "relative_delta": owner_outputs["tabu"][owner]
                    / max(owner_outputs["control"][owner], 1e-30)
                    - 1.0,
                }
                for owner in owners
            }
            if owner_outputs
            else {},
        }
    )
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
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
