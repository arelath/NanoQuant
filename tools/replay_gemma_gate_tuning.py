"""Replay block-0 gate tuning from retained legacy and rewrite initial states."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.layers import BlockEditor, TrainableFactorizedLinear
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.application.tuning import TuningRequest, tune_factorized
from nanoquant.domain.models import ArtifactRef
from nanoquant.domain.scale_fit import MaterializedScaleFitResult, fit_scales
from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.resident_quantization import (
    _block_loss,
    _checkpoint_dtype,
    _legacy_cuda_numerics,
    _run_block_batched,
    _run_prefix_batched,
)

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
CALIBRATION_ARTIFACT = "sha256-ad1f609729f86db7598eed5c703c55aacbb9cb024cab816ca7b300d574b7a4c8"
GATE_PATH = "mlp.gate_proj"


def _read_tensors(path: Path, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        missing = sorted(set(keys) - set(handle.keys()))
        if missing:
            raise KeyError(f"{path} is missing tensors: {missing}")
        return {key: handle.get_tensor(key) for key in keys}


def _legacy_initial(path: Path) -> dict[str, torch.Tensor]:
    values = _read_tensors(
        path,
        (
            "U_latent",
            "V_latent",
            "scale_pre",
            "scale_mid",
            "scale_post",
            "salient_idx",
            "salient_weight",
        ),
    )
    return {
        "left": values["U_latent"],
        "right": values["V_latent"],
        "scale_pre": values["scale_pre"],
        "scale_mid": values["scale_mid"],
        "scale_post": values["scale_post"],
        "outlier_indices": values["salient_idx"].long(),
        "outlier_values": values["salient_weight"],
    }


def _rewrite_initial(factor_path: Path, scale_path: Path, frozen_path: Path) -> dict[str, torch.Tensor]:
    factors = _read_tensors(factor_path, ("left_latent", "right_latent"))
    scales = _read_tensors(scale_path, ("scale_pre", "scale_mid", "scale_post"))
    outliers = _read_tensors(frozen_path, ("outlier_indices", "outlier_values"))
    return {
        "left": factors["left_latent"],
        "right": factors["right_latent"],
        **scales,
        **outliers,
    }


def _rewrite_pre_scale_fit(factor_path: Path, frozen_path: Path) -> dict[str, torch.Tensor]:
    factors = _read_tensors(
        factor_path,
        ("left_latent", "right_latent", "scale_pre", "scale_mid", "scale_post"),
    )
    outliers = _read_tensors(frozen_path, ("outlier_indices", "outlier_values"))
    return {
        "left": factors["left_latent"],
        "right": factors["right_latent"],
        "scale_pre": factors["scale_pre"],
        "scale_mid": factors["scale_mid"],
        "scale_post": factors["scale_post"],
        **outliers,
    }


def _refit_state(
    state: dict[str, torch.Tensor],
    target_weight: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    alternating_passes: int,
) -> tuple[dict[str, torch.Tensor], MaterializedScaleFitResult]:
    """Refit a retained factorization against its residual-weight objective."""
    residual = target_weight.detach().float().clone()
    device_state = {key: value.to(residual.device) for key, value in state.items()}
    protected = device_state["outlier_indices"].long()
    residual[:, protected] = 0
    fitted = fit_scales(
        residual,
        device_state["left"],
        device_state["right"],
        device_state["scale_pre"],
        device_state["scale_mid"],
        device_state["scale_post"],
        input_importance.to(residual.device),
        output_importance.to(residual.device),
        alternating_passes=alternating_passes,
        protected_columns=protected,
    )
    refitted = dict(device_state)
    refitted.update(
        {
            "scale_pre": fitted.scale_pre,
            "scale_mid": fitted.scale_mid,
            "scale_post": fitted.scale_post,
        }
    )
    return refitted, fitted


def _module(state: dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> TrainableFactorizedLinear:
    return TrainableFactorizedLinear(
        state["left"],
        state["right"],
        state["scale_pre"],
        state["scale_mid"],
        state["scale_post"],
        outlier_indices=state["outlier_indices"],
        outlier_values=state["outlier_values"],
    ).to(device=device, dtype=dtype)


def _weighted_weight_metrics(
    source: torch.Tensor,
    module: TrainableFactorizedLinear,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> dict[str, float]:
    prediction = module.dense_weight().float()
    target = source.float()
    weighted_error = (
        (prediction - target).square()
        * input_importance.float().reshape(1, -1)
        * output_importance.float().reshape(-1, 1)
    ).sum()
    target_norm = (
        target.square() * input_importance.float().reshape(1, -1) * output_importance.float().reshape(-1, 1)
    ).sum()
    return {
        "weighted_error": float(weighted_error.detach()),
        "target_weighted_norm": float(target_norm.detach()),
        "weighted_normalized_error": float(
            (weighted_error / target_norm.clamp_min(1e-12)).detach()
        ),
    }


def _comparison(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in ("left", "right", "scale_pre", "scale_mid", "scale_post", "outlier_values"):
        a = left[key].detach().float().cpu().reshape(-1)
        b = right[key].detach().float().cpu().reshape(-1)
        if key in {"left", "right"}:
            a = torch.where(a >= 0, 1.0, -1.0)
            b = torch.where(b >= 0, 1.0, -1.0)
        result[key] = {
            "shape": list(left[key].shape),
            "exact": bool(torch.equal(a, b)),
            "agreement": float((a == b).float().mean()),
            "maximum_absolute_difference": float((a - b).abs().max()),
            "relative_l2_difference": float((a - b).norm() / b.norm().clamp_min(1e-12)),
        }
    result["outlier_indices_exact"] = bool(
        torch.equal(left["outlier_indices"].long().cpu(), right["outlier_indices"].long().cpu())
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path(".cache/nanoquant/calibration/experiment018"))
    parser.add_argument("--fisher", type=Path, required=True)
    parser.add_argument("--legacy-initial", type=Path, required=True)
    parser.add_argument("--rewrite-factor", type=Path, required=True)
    parser.add_argument("--rewrite-scales", type=Path, required=True)
    parser.add_argument("--rewrite-frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", default="8,32")
    parser.add_argument(
        "--ls-scale-fit-passes",
        default="0,1,2,4,8",
        help="comma-separated alternating-pass counts for the pre-tuning LS sweep",
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--block-forward-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--wait-for-device-seconds",
        type=float,
        default=0.0,
        help="wait this long for the named device lease instead of failing immediately",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    epochs = tuple(int(value.strip()) for value in args.epochs.split(",") if value.strip())
    if not epochs or any(value <= 0 for value in epochs):
        raise ValueError("epochs must contain positive integers")
    ls_scale_fit_passes = tuple(
        int(value.strip()) for value in args.ls_scale_fit_passes.split(",") if value.strip()
    )
    if not ls_scale_fit_passes or any(value < 0 for value in ls_scale_fit_passes):
        raise ValueError("ls-scale-fit-passes must contain non-negative integers")
    legacy = _legacy_initial(args.legacy_initial)
    rewrite = _rewrite_initial(args.rewrite_factor, args.rewrite_scales, args.rewrite_frozen)
    rewrite_pre_scale_fit = _rewrite_pre_scale_fit(args.rewrite_factor, args.rewrite_frozen)
    calibration = load_pinned_calibration(
        args.calibration,
        ArtifactRef("calibration-dataset-manifest", CALIBRATION_ARTIFACT, 1),
    )
    if args.samples <= 0 or args.samples > calibration.input_ids.shape[0]:
        raise ValueError("sample count is outside the pinned calibration tensor")

    source = SafetensorsModelSource(
        args.snapshot,
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        verify_hashes=True,
    )
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    inventory = adapter.model_inventory(source)
    with safe_open(args.fisher, framework="pt", device="cpu") as handle:
        input_importance = handle.get_tensor("i.model.layers.0.mlp.gate_proj")
        gate_output_importance = handle.get_tensor("o.model.layers.0.mlp.gate_proj")
        block_output_importance = handle.get_tensor("o.model.layers.0.mlp.down_proj")

    payload: dict[str, object] = {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "samples": args.samples,
        "epochs": list(epochs),
        "epoch_loss_mode": "legacy_training",
        "ls_scale_fit_passes": list(ls_scale_fit_passes),
        "protocol": {
            "snapshot": str(args.snapshot.resolve()),
            "calibration": str(args.calibration.resolve()),
            "fisher": str(args.fisher.resolve()),
            "legacy_initial": str(args.legacy_initial.resolve()),
            "rewrite_factor": str(args.rewrite_factor.resolve()),
            "rewrite_scales": str(args.rewrite_scales.resolve()),
            "rewrite_frozen": str(args.rewrite_frozen.resolve()),
            "device": args.device,
            "batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size,
            "block_forward_batch_size": args.block_forward_batch_size,
        },
        "environment": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        },
        "initial_state_comparison": _comparison(legacy, rewrite),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wait_for_device_lease(args.device, args.wait_for_device_seconds), _legacy_cuda_numerics():
        tokens = calibration.input_ids[: args.samples].to(args.device)
        model = cast(
            nn.Module,
            AutoModelForCausalLM.from_pretrained(
                args.snapshot,
                local_files_only=True,
                torch_dtype=_checkpoint_dtype(checkpoint.config),
                attn_implementation=adapter.attention_implementation,
            ),
        ).to(args.device)
        model.eval()
        layers = getattr(getattr(model, "model", None), "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("model does not expose decoder layers")
        text_model = getattr(model, "model", model)
        capture = capture_prefix_invocations(
            layers[0],
            (lambda: cast(Any, text_model)(input_ids=tokens[:1], use_cache=False),),
        )[0]
        metadata = capture.keyword
        inputs = _run_prefix_batched(adapter, model, tokens, args.block_forward_batch_size, "cpu").detach()
        source_block = layers[0]
        targets = _run_block_batched(
            adapter,
            source_block,
            inputs,
            metadata,
            args.block_forward_batch_size,
            "cpu",
        ).detach()
        with source.read_tensor("model.layers.0.mlp.gate_proj.weight", args.device) as source_weight:
            pre_fit_module = _module(rewrite_pre_scale_fit, args.device, inputs.dtype)
            post_fit_module = _module(rewrite, args.device, inputs.dtype)
            pre_fit_metrics = _weighted_weight_metrics(
                source_weight,
                pre_fit_module,
                input_importance.to(args.device),
                gate_output_importance.to(args.device),
            )
            post_fit_metrics = _weighted_weight_metrics(
                source_weight,
                post_fit_module,
                input_importance.to(args.device),
                gate_output_importance.to(args.device),
            )
            payload["rewrite_ls_scale_fit"] = {
                "before": pre_fit_metrics,
                "after": post_fit_metrics,
                "weighted_error_improvement_fraction": (
                    pre_fit_metrics["weighted_error"] - post_fit_metrics["weighted_error"]
                )
                / pre_fit_metrics["weighted_error"],
            }
            del pre_fit_module, post_fit_module
            ls_scale_fit_sweep: list[dict[str, object]] = []
            for pass_count in ls_scale_fit_passes:
                refitted_state, fitted = _refit_state(
                    rewrite_pre_scale_fit,
                    source_weight,
                    input_importance.to(args.device),
                    gate_output_importance.to(args.device),
                    pass_count,
                )
                block = adapter.load_block(source, inventory.blocks[0].block, args.device)
                block.eval()
                refitted_module = _module(refitted_state, args.device, inputs.dtype)
                BlockEditor().install_trainable_layer(block, GATE_PATH, refitted_module)
                row: dict[str, object] = {
                    "alternating_passes": pass_count,
                    "accepted": fitted.accepted,
                    "rollback_reason": fitted.rollback_reason,
                    "residual_weighted_error_before": fitted.before_error,
                    "residual_weighted_error_after": fitted.after_error,
                    "full_weight": _weighted_weight_metrics(
                        source_weight,
                        refitted_module,
                        input_importance.to(args.device),
                        gate_output_importance.to(args.device),
                    ),
                    "block_loss": _block_loss(
                        adapter,
                        block,
                        inputs,
                        targets,
                        block_output_importance,
                        metadata,
                        args.block_forward_batch_size,
                    ),
                    "persisted_rewrite_scale_comparison": _comparison(rewrite, refitted_state),
                }
                ls_scale_fit_sweep.append(row)
                print(json.dumps({"ls_scale_fit": row}, indent=2, sort_keys=True))
                del block, refitted_module, refitted_state
            payload["ls_scale_fit_sweep"] = ls_scale_fit_sweep
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            for name, state in (("legacy", legacy), ("rewrite", rewrite)):
                for epoch_count in epochs:
                    if args.device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(args.device)
                    block = adapter.load_block(source, inventory.blocks[0].block, args.device)
                    block.eval()
                    trainable = _module(state, args.device, inputs.dtype)
                    BlockEditor().install_trainable_layer(block, GATE_PATH, trainable)
                    before_weight = _weighted_weight_metrics(
                        source_weight,
                        trainable,
                        input_importance.to(args.device),
                        gate_output_importance.to(args.device),
                    )
                    before_block = _block_loss(
                        adapter,
                        block,
                        inputs,
                        targets,
                        block_output_importance,
                        metadata,
                        args.block_forward_batch_size,
                    )
                    started = time.perf_counter()
                    trajectory: list[dict[str, float | int]] = []
                    metrics = tune_factorized(
                        block,
                        GATE_PATH,
                        TuningRequest(
                            inputs,
                            targets,
                            epoch_count,
                            args.batch_size,
                            1e-5,
                            output_importance=block_output_importance,
                            seed=0,
                            microbatch_size=args.microbatch_size,
                            restore_best_state=False,
                            epoch_loss_mode="legacy_training",
                            epoch_observer=lambda epoch, loss, trajectory=trajectory: trajectory.append(
                                {"epoch": epoch, "loss": loss}
                            ),
                        ),
                        lambda module, value: adapter.run_block(module, value, **metadata),
                    )
                    after_weight = _weighted_weight_metrics(
                        source_weight,
                        trainable,
                        input_importance.to(args.device),
                        gate_output_importance.to(args.device),
                    )
                    row = {
                        "initialization": name,
                        "epochs": epoch_count,
                        "before_block_loss": before_block,
                        "tuning_before_loss": (
                            None if metrics.before is None else metrics.before.loss
                        ),
                        "best_loss": metrics.best.loss,
                        "final_loss": metrics.final.loss,
                        "best_epoch": metrics.best_epoch,
                        "trajectory": trajectory,
                        "initial_weight": before_weight,
                        "final_weight": after_weight,
                        "wall_seconds": time.perf_counter() - started,
                        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
                        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)),
                        "final_cuda_allocated_bytes": int(torch.cuda.memory_allocated(args.device)),
                        "final_cuda_reserved_bytes": int(torch.cuda.memory_reserved(args.device)),
                    }
                    cast(list[object], payload["runs"]).append(row)
                    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                    print(json.dumps(row, indent=2, sort_keys=True))
                    del block, trainable
                    torch.cuda.empty_cache()
        model.to("cpu")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
