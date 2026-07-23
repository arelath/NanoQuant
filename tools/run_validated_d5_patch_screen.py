"""Screen D5 o_proj patch ranks from one validated frozen operating point."""

from __future__ import annotations

import argparse
import gc
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from safetensors.torch import save_file

from nanoquant.application.covariance import SplitDenseCovarianceAccumulator
from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    load_kl_budget_profile,
    paired_bootstrap_kl_delta,
)
from nanoquant.application.low_rank_patch import FittedLowRankPatch, fit_low_rank_patch_family
from nanoquant.config.codec import to_dict
from nanoquant.domain.models import ArtifactRef, LayerId
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
    collect_splice_reconstructions,
)
from nanoquant.infrastructure.kl_teacher_cache import (
    commit_active_kl_teacher_cache,
    load_active_kl_teacher_cache,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.kl_budget_workflow import _teacher_cache_key, _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

SOURCE = "unsloth/gemma-3-270m-it"
REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
O_ARM = "type:self_attn.o_proj"
RANKS = (4, 8, 16)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-tokens", type=int, default=4096)
    parser.add_argument("--held-out-tokens", type=int, default=4096)
    parser.add_argument("--wikitext-samples", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=512)
    return parser


def _dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    if not isinstance(value, str):
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(value, torch.float32)


def _calibration(base_run: Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    receipt = json.loads((base_run / "calibration-input.json").read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("validated base run calibration receipt must be an object")
    reference = ArtifactRef(
        "calibration-dataset-manifest",
        str(receipt["artifact_id"]),
        1,
    )
    dataset = load_pinned_calibration(base_run, reference)
    return dataset.input_ids, dataset.attention_mask, dataset.fingerprint


def _require_fresh_validation(base_run: Path) -> None:
    path = base_run / "resident-validation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("complete") is not True
        or int(payload.get("block_records", 0)) != 18
        or int(payload.get("committed_layer_count", 0)) != 90
    ):
        raise ValueError("D5 base run has no complete strict resident validation receipt")


def _o_proj_layers(reconstructions: SpliceReconstructionSet) -> tuple[LayerId, ...]:
    layers = tuple(item.layer for item in reconstructions.layers if item.layer.path == "self_attn.o_proj")
    if not layers:
        raise ValueError("validated base run contains no o_proj reconstruction")
    return layers


@torch.inference_mode()
def _capture_moments(
    model: torch.nn.Module,
    layers: tuple[LayerId, ...],
    adapter: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    fit_tokens: int,
    held_out_tokens: int,
    device: str,
) -> dict[LayerId, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    blocks = adapter.get_decoder_layers(model)
    accumulators: dict[LayerId, SplitDenseCovarianceAccumulator] = {}
    with ExitStack() as stack:
        for layer in layers:
            module = blocks[layer.block.index].get_submodule(layer.path)
            right_factor = getattr(module, "right_binary", None)
            if not isinstance(right_factor, torch.Tensor) or right_factor.ndim != 2:
                raise TypeError("D5 o_proj frozen module has no matrix right factor")
            width = int(right_factor.shape[1])
            accumulator = SplitDenseCovarianceAccumulator(
                width,
                fit_tokens,
                held_out_tokens,
                device=device,
            )
            accumulators[layer] = accumulator

            def capture(
                _module: torch.nn.Module,
                positional: tuple[object, ...],
                *,
                target: SplitDenseCovarianceAccumulator = accumulator,
            ) -> None:
                if not positional or not isinstance(positional[0], torch.Tensor):
                    raise TypeError("D5 o_proj input is not a tensor")
                target.update(positional[0])

            handle = module.register_forward_pre_hook(capture)
            stack.callback(handle.remove)

        rows_per_sample = input_ids.shape[1]
        required_samples = (fit_tokens + held_out_tokens + rows_per_sample - 1) // rows_per_sample
        if required_samples > input_ids.shape[0]:
            raise ValueError("D5 calibration dataset has too few activation rows")
        base = getattr(model, "model", None)
        if not isinstance(base, torch.nn.Module):
            raise TypeError("D5 frozen model has no decoder base")
        for index in range(required_samples):
            batch = input_ids[index : index + 1].to(device)
            mask = attention_mask[index : index + 1].to(device)
            cast(Any, base)(input_ids=batch, attention_mask=mask, use_cache=False)
            del batch, mask
        if any(not accumulator.complete for accumulator in accumulators.values()):
            raise ValueError("D5 activation moment capture was incomplete")

    result = {}
    for layer, accumulator in accumulators.items():
        fit_covariance, fit_mean = accumulator.fit.materialize()
        held_covariance, held_mean = accumulator.held_out.materialize()
        result[layer] = (fit_covariance, held_covariance, fit_mean, held_mean)
    return result


def _fit_patches(
    snapshot: Path,
    config: dict[str, object],
    reconstructions: SpliceReconstructionSet,
    moments: dict[LayerId, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: str,
) -> dict[int, dict[LayerId, FittedLowRankPatch]]:
    source = SafetensorsModelSource(
        snapshot,
        source=SOURCE,
        revision=REVISION,
        verify_hashes=True,
    )
    source.inventory()
    adapter = adapter_for_config(config)
    base = {item.layer: item for item in reconstructions.layers}
    fitted: dict[int, dict[LayerId, FittedLowRankPatch]] = {rank: {} for rank in RANKS}
    for layer, moment_values in moments.items():
        with source.read_tensor(adapter.source_key(layer), device=device) as target:
            candidates = fit_low_rank_patch_family(
                target,
                base[layer].weight.to(device),
                *(value.to(device) for value in moment_values),
                ranks=RANKS,
                ridge_fraction=1e-2,
                storage_dtype=torch.float16,
                require_held_out_acceptance=True,
            )
        for candidate in candidates:
            fitted[candidate.rank][layer] = candidate
    return fitted


def _patched_reconstructions(
    base: SpliceReconstructionSet,
    patches: dict[LayerId, FittedLowRankPatch],
) -> SpliceReconstructionSet:
    layers = []
    for item in base.layers:
        patch = patches.get(item.layer)
        weight = item.weight
        if patch is not None and patch.accepted:
            weight = weight.float() + patch.left.float() @ patch.right.float()
        layers.append(
            SpliceReconstruction(
                item.layer,
                weight,
                item.bias,
                item.weighted_normalized_squared_error,
            )
        )
    return SpliceReconstructionSet(
        tuple(layers),
        base.unit_members,
        base.unit_weighted_normalized_squared_errors,
    )


def _interval(before: KlBudgetArmResult, after: KlBudgetArmResult) -> dict[str, float | bool]:
    value = paired_bootstrap_kl_delta(before, after)
    relative = value.point_delta / before.kl_nats_per_token
    return {
        "before": before.kl_nats_per_token,
        "after": after.kl_nats_per_token,
        "delta": value.point_delta,
        "relative_delta": relative,
        "lower_delta": value.lower_delta,
        "upper_delta": value.upper_delta,
        "confidence": value.confidence,
        "resamples": float(value.resamples),
        "improved_with_confidence": value.point_delta < 0 and value.upper_delta < 0,
    }


def _save_patches(
    output: Path,
    fitted: dict[int, dict[LayerId, FittedLowRankPatch]],
) -> dict[str, object]:
    inventory: dict[str, object] = {}
    for rank, patches in fitted.items():
        values: dict[str, torch.Tensor] = {}
        accepted = []
        for layer, patch in sorted(patches.items(), key=lambda item: item[0].block.index):
            if patch.accepted:
                prefix = f"blocks.{layer.block.index}.{layer.path}"
                values[f"{prefix}.patch_left"] = patch.left
                values[f"{prefix}.patch_right"] = patch.right
                accepted.append(f"{layer.block.index}:{layer.path}")
        path = output / f"patches-rank-{rank}.safetensors"
        if values:
            save_file(values, path)
        inventory[str(rank)] = {
            "path": str(path),
            "sha256": None if not values else f"sha256:{hash_file(path)}",
            "accepted_layers": accepted,
            "accepted_owner_count": len(accepted),
            "actual_patch_bits": sum(value.numel() * value.element_size() * 8 for value in values.values()),
            "layer_metrics": {
                f"{layer.block.index}:{layer.path}": {
                    "rank": patch.rank,
                    "fit_error_before": patch.fit_error_before,
                    "fit_error_after": patch.fit_error_after,
                    "held_out_error_before": patch.held_out_error_before,
                    "held_out_error_after": patch.held_out_error_after,
                    "accepted": patch.accepted,
                    "rejection_reason": patch.rejection_reason,
                }
                for layer, patch in sorted(patches.items(), key=lambda item: item[0].block.index)
            },
        }
    return inventory


def _baseline_arm(profile: Path, fingerprint: str, token_hash: str) -> KlBudgetArmResult:
    loaded = load_kl_budget_profile(profile / "kl-budget-profile.json")
    if (
        loaded.provenance.model_source != SOURCE
        or loaded.provenance.model_revision != REVISION
        or loaded.provenance.dataset_fingerprint != fingerprint
        or loaded.provenance.dataset_slice_hash != token_hash
    ):
        raise ValueError("D5 baseline profile does not match the screening model and dataset")
    result = next((arm for arm in loaded.arms if arm.arm == O_ARM), None)
    if result is None:
        raise ValueError("D5 baseline profile contains no o_proj arm")
    return result


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if args.fit_tokens <= 0 or args.held_out_tokens <= 0:
        raise ValueError("D5 fit and held-out token counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    _require_fresh_validation(args.base_run)
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("D5 model config must be an object")
    config = cast(dict[str, object], config_payload)
    calibration_ids, calibration_mask, calibration_fingerprint = _calibration(args.base_run)

    with acquire_device_lease(args.device):
        loaded = load_frozen_run(
            args.base_run,
            args.snapshot,
            source_name=SOURCE,
            revision=REVISION,
            device=args.device,
            verify_hashes=True,
            backend="factorized",
            use_global_tuning=False,
        )
        base_reconstructions = collect_splice_reconstructions(loaded)
        o_layers = _o_proj_layers(base_reconstructions)
        moments = _capture_moments(
            loaded.model,
            o_layers,
            adapter_for_config(config),
            calibration_ids,
            calibration_mask,
            fit_tokens=args.fit_tokens,
            held_out_tokens=args.held_out_tokens,
            device=args.device,
        )
        identity = loaded.identity
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fitted = _fit_patches(
            args.snapshot,
            config,
            base_reconstructions,
            moments,
            device=args.device,
        )
        patch_inventory = _save_patches(args.output, fitted)
        tokens, dataset_fingerprint, _bos = _wikitext_tokens(
            args.snapshot,
            samples=args.wikitext_samples,
            sequence_length=args.sequence_length,
            local_files_only=True,
        )
        token_hash = _token_hash(tokens)
        baseline = _baseline_arm(args.baseline_profile, dataset_fingerprint, token_hash)
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter_for_config(config).attention_implementation,
            local_files_only=True,
        ).to(args.device)
        teacher.eval()
        cache_key = _teacher_cache_key(
            source=SOURCE,
            revision=REVISION,
            model_hash=identity.model_hash,
            token_hash=token_hash,
            model_dtype=_dtype(config),
            attention_implementation=adapter_for_config(config).attention_implementation,
            device=args.device,
            batch_size=1,
        )
        cache = load_active_kl_teacher_cache(args.teacher_cache_root, cache_key)
        results: dict[int, KlBudgetArmResult] = {}
        for rank in RANKS:
            evaluator = DenseKlSpliceEvaluator(
                teacher,
                _patched_reconstructions(base_reconstructions, fitted[rank]),
                tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            if cache is None:
                baseline_nll, batches = evaluator.teacher_cache_state()
                cache = commit_active_kl_teacher_cache(
                    args.teacher_cache_root,
                    cache_key,
                    baseline_nll,
                    batches,
                )
            else:
                evaluator.install_teacher_cache(
                    cache.baseline_negative_log_likelihood,
                    cache.batches,
                )
            results[rank] = evaluator(O_ARM)
            del evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gates = {rank: _interval(baseline, result) for rank, result in results.items()}
    eligible = [
        rank
        for rank in RANKS
        if cast(dict[str, object], patch_inventory[str(rank)])["accepted_owner_count"]
        and bool(gates[rank]["improved_with_confidence"])
    ]
    selected_rank = min(eligible) if eligible else 0
    atomic_write_json(
        args.output / "patch-screen-summary.json",
        {
            "schema_version": 1,
            "status": "completed",
            "screening_role": "rank-selection-only; final equal-BPW adoption requires recompression",
            "base_run": str(args.base_run.resolve()),
            "base_identity": to_dict(identity),
            "calibration_fingerprint": calibration_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_slice_hash": token_hash,
            "fit_tokens": args.fit_tokens,
            "held_out_tokens": args.held_out_tokens,
            "baseline": to_dict(baseline),
            "ranks": {
                str(rank): {
                    "kl": to_dict(results[rank]),
                    "gate": gates[rank],
                    "patch": patch_inventory[str(rank)],
                }
                for rank in RANKS
            },
            "selected_rank": selected_rank,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
