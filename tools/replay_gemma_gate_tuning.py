"""Replay one Gemma projection's tuning from retained legacy and rewrite states."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.kl_budget import KlBudgetArmResult, paired_bootstrap_kl_delta
from nanoquant.application.layers import BlockEditor, TrainableFactorizedLinear
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.application.tuning import (
    FactorizedTuningLearningRates,
    TuningRequest,
    tune_factorized,
)
from nanoquant.config.codec import to_dict
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.domain.scale_fit import MaterializedScaleFitResult, fit_scales
from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.hf_calibration_dataset import load_or_prepare_calibration
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
)
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
DEFAULT_LAYER_PATH = "mlp.gate_proj"


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
        "weighted_normalized_error": float((weighted_error / target_norm.clamp_min(1e-12)).detach()),
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
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--layer-path", default=DEFAULT_LAYER_PATH)
    parser.add_argument("--fisher", type=Path)
    parser.add_argument(
        "--resident-calibration",
        type=Path,
        help="resident calibration tensor artifact containing the selected projection importance",
    )
    parser.add_argument(
        "--calibration-directory",
        type=Path,
        help="reuse calibration-input.json and its artifact store from this run directory",
    )
    parser.add_argument("--legacy-initial", type=Path)
    parser.add_argument("--rewrite-factor", type=Path, required=True)
    parser.add_argument("--rewrite-scales", type=Path, required=True)
    parser.add_argument("--rewrite-frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", default="8,32")
    parser.add_argument(
        "--binary-learning-rates",
        default="1e-5",
        help="comma-separated binary rates; scale, outlier, and bias remain at 1e-5",
    )
    parser.add_argument(
        "--initializations",
        default="legacy,rewrite",
        help="comma-separated subset of legacy,rewrite",
    )
    parser.add_argument(
        "--checkpoint-policies",
        default="legacy_final",
        help=(
            "comma-separated subset of legacy_final,fit_best,heldout_best; "
            "heldout_best selects and restores on the reserved sample suffix"
        ),
    )
    parser.add_argument(
        "--ls-scale-fit-passes",
        default="0,1,2,4,8",
        help="comma-separated alternating-pass counts for the pre-tuning LS sweep",
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="start at this calibration row so confirmation runs can use a disjoint sample window",
    )
    parser.add_argument(
        "--fit-samples",
        type=int,
        help="fit on this prefix and reserve the remaining requested samples for held-out block loss",
    )
    parser.add_argument(
        "--kl-samples",
        type=int,
        default=0,
        help="evaluate each tuned gate as a dense full-model splice on this many calibration rows",
    )
    parser.add_argument(
        "--kl-offset",
        type=int,
        default=0,
        help="absolute calibration-row offset for the optional language-model splice gate",
    )
    parser.add_argument("--kl-sequence-length", type=int, default=512)
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
    if (args.fisher is None) == (args.resident_calibration is None):
        raise ValueError("provide exactly one of --fisher or --resident-calibration")
    if args.block_index < 0:
        raise ValueError("block index must be non-negative")
    layer_path = str(args.layer_path).strip()
    if not layer_path or layer_path == "self_attn.attn_qkv":
        raise ValueError("layer path must name one independent projection")
    epochs = tuple(int(value.strip()) for value in args.epochs.split(",") if value.strip())
    if not epochs or any(value <= 0 for value in epochs):
        raise ValueError("epochs must contain positive integers")
    binary_learning_rates = tuple(
        float(value.strip()) for value in args.binary_learning_rates.split(",") if value.strip()
    )
    if not binary_learning_rates or any(not math.isfinite(value) or value <= 0 for value in binary_learning_rates):
        raise ValueError("binary-learning-rates must contain positive finite values")
    initializations = tuple(value.strip() for value in args.initializations.split(",") if value.strip())
    if (
        not initializations
        or len(initializations) != len(set(initializations))
        or any(value not in {"legacy", "rewrite"} for value in initializations)
    ):
        raise ValueError("initializations must be a unique subset of legacy,rewrite")
    checkpoint_policies = tuple(value.strip() for value in args.checkpoint_policies.split(",") if value.strip())
    valid_checkpoint_policies = {"legacy_final", "fit_best", "heldout_best"}
    if (
        not checkpoint_policies
        or len(checkpoint_policies) != len(set(checkpoint_policies))
        or any(value not in valid_checkpoint_policies for value in checkpoint_policies)
    ):
        raise ValueError("checkpoint-policies must be a unique subset of legacy_final,fit_best,heldout_best")
    ls_scale_fit_passes = tuple(int(value.strip()) for value in args.ls_scale_fit_passes.split(",") if value.strip())
    if not ls_scale_fit_passes or any(value < 0 for value in ls_scale_fit_passes):
        raise ValueError("ls-scale-fit-passes must contain non-negative integers")
    if "legacy" in initializations and args.legacy_initial is None:
        raise ValueError("legacy initialization requires --legacy-initial")
    legacy = None if args.legacy_initial is None else _legacy_initial(args.legacy_initial)
    rewrite = _rewrite_initial(args.rewrite_factor, args.rewrite_scales, args.rewrite_frozen)
    rewrite_pre_scale_fit = _rewrite_pre_scale_fit(args.rewrite_factor, args.rewrite_frozen)
    calibration = load_or_prepare_calibration(
        args.snapshot,
        args.output.parent if args.calibration_directory is None else args.calibration_directory,
    )
    if args.sample_offset < 0:
        raise ValueError("sample offset must be non-negative")
    sample_stop = args.sample_offset + args.samples
    if args.samples <= 0 or sample_stop > calibration.input_ids.shape[0]:
        raise ValueError("sample count is outside the pinned calibration tensor")
    fit_samples = args.samples if args.fit_samples is None else args.fit_samples
    if fit_samples <= 0 or fit_samples > args.samples:
        raise ValueError("fit sample count is outside the requested sample range")
    if "heldout_best" in checkpoint_policies and fit_samples == args.samples:
        raise ValueError("heldout_best requires a non-empty reserved sample suffix")
    if args.kl_samples < 0 or args.kl_offset < 0 or args.kl_sequence_length <= 1:
        raise ValueError("KL splice protocol values are invalid")
    kl_stop = args.kl_offset + args.kl_samples
    if kl_stop > calibration.input_ids.shape[0]:
        raise ValueError("KL sample count is outside the pinned calibration tensor")
    fit_start = args.sample_offset
    fit_stop = fit_start + fit_samples
    if args.kl_samples and max(fit_start, args.kl_offset) < min(fit_stop, kl_stop):
        raise ValueError("KL splice rows must not overlap the tuning-fit rows")
    if args.kl_samples and args.kl_sequence_length > calibration.input_ids.shape[1]:
        raise ValueError("KL sequence length exceeds the pinned calibration tensor")

    source = SafetensorsModelSource(
        args.snapshot,
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        verify_hashes=True,
    )
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    inventory = adapter.model_inventory(source)
    if args.block_index >= len(inventory.blocks):
        raise ValueError("block index is outside the model inventory")
    importance_path = args.fisher if args.resident_calibration is None else args.resident_calibration
    if importance_path is None:
        raise AssertionError("importance source validation did not resolve a path")
    with safe_open(importance_path, framework="pt", device="cpu") as handle:
        if args.resident_calibration is None:
            input_importance = handle.get_tensor(f"i.model.layers.{args.block_index}.{layer_path}")
            layer_output_importance = handle.get_tensor(f"o.model.layers.{args.block_index}.{layer_path}")
            block_output_importance = handle.get_tensor(f"o.model.layers.{args.block_index}.mlp.down_proj")
        else:
            input_importance = handle.get_tensor(f"block_{args.block_index}.{layer_path}.input_importance")
            layer_output_importance = handle.get_tensor(f"block_{args.block_index}.{layer_path}.output_importance")
            block_output_importance = handle.get_tensor(f"block_{args.block_index}.mlp.down_proj.output_importance")

    payload: dict[str, object] = {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "block_index": args.block_index,
        "layer_path": layer_path,
        "sample_offset": args.sample_offset,
        "samples": args.samples,
        "fit_samples": fit_samples,
        "held_out_samples": args.samples - fit_samples,
        "kl_samples": args.kl_samples,
        "epochs": list(epochs),
        "binary_learning_rates": list(binary_learning_rates),
        "initializations": list(initializations),
        "checkpoint_policies": list(checkpoint_policies),
        "epoch_loss_mode": (
            "legacy_training" if checkpoint_policies == ("legacy_final",) else "varies_by_checkpoint_policy"
        ),
        "ls_scale_fit_passes": list(ls_scale_fit_passes),
        "protocol": {
            "snapshot": str(args.snapshot.resolve()),
            "calibration_artifact": calibration.reference.artifact_id,
            "importance": str(importance_path.resolve()),
            "importance_kind": "legacy_fisher" if args.resident_calibration is None else "resident_calibration",
            "calibration_directory": (
                None if args.calibration_directory is None else str(args.calibration_directory.resolve())
            ),
            "legacy_initial": None if args.legacy_initial is None else str(args.legacy_initial.resolve()),
            "rewrite_factor": str(args.rewrite_factor.resolve()),
            "rewrite_scales": str(args.rewrite_scales.resolve()),
            "rewrite_frozen": str(args.rewrite_frozen.resolve()),
            "device": args.device,
            "batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size,
            "block_forward_batch_size": args.block_forward_batch_size,
            "sample_offset": args.sample_offset,
            "fit_samples": fit_samples,
            "kl_offset": args.kl_offset,
            "kl_samples": args.kl_samples,
            "kl_sequence_length": args.kl_sequence_length,
        },
        "environment": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        },
        "initial_state_comparison": (
            None if legacy is None or set(initializations) != {"legacy", "rewrite"} else _comparison(legacy, rewrite)
        ),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wait_for_device_lease(args.device, args.wait_for_device_seconds), _legacy_cuda_numerics():
        kl_reconstructions: list[torch.Tensor] = []
        tokens = calibration.input_ids[args.sample_offset : sample_stop].to(args.device)
        model = cast(
            nn.Module,
            AutoModelForCausalLM.from_pretrained(
                args.snapshot,
                local_files_only=False,
                torch_dtype=_checkpoint_dtype(checkpoint.config),
                attn_implementation=adapter.attention_implementation,
            ),
        ).to(args.device)
        model.eval()
        layers = getattr(getattr(model, "model", None), "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("model does not expose decoder layers")
        text_model = getattr(model, "model", model)
        source_block = layers[args.block_index]
        capture = capture_prefix_invocations(
            source_block,
            (lambda: cast(Any, text_model)(input_ids=tokens[:1], use_cache=False),),
        )[0]
        metadata = capture.keyword
        if args.block_index == 0:
            inputs = _run_prefix_batched(adapter, model, tokens, args.block_forward_batch_size, "cpu").detach()
        else:
            captured_inputs: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, tokens.shape[0], args.block_forward_batch_size):
                    end = min(start + args.block_forward_batch_size, tokens.shape[0])
                    invocation = capture_prefix_invocations(
                        source_block,
                        (
                            lambda start=start, end=end: cast(Any, text_model)(
                                input_ids=tokens[start:end], use_cache=False
                            ),
                        ),
                    )[0]
                    if not invocation.positional or not isinstance(invocation.positional[0], torch.Tensor):
                        raise TypeError("captured block invocation has no tensor input")
                    captured_inputs.append(invocation.positional[0].to("cpu"))
            inputs = torch.cat(captured_inputs, dim=0).detach()
        targets = _run_block_batched(
            adapter,
            source_block,
            inputs,
            metadata,
            args.block_forward_batch_size,
            "cpu",
        ).detach()
        source_weight_key = f"model.layers.{args.block_index}.{layer_path}.weight"
        with source.read_tensor(source_weight_key, args.device) as source_weight:
            pre_fit_module = _module(rewrite_pre_scale_fit, args.device, inputs.dtype)
            post_fit_module = _module(rewrite, args.device, inputs.dtype)
            pre_fit_metrics = _weighted_weight_metrics(
                source_weight,
                pre_fit_module,
                input_importance.to(args.device),
                layer_output_importance.to(args.device),
            )
            post_fit_metrics = _weighted_weight_metrics(
                source_weight,
                post_fit_module,
                input_importance.to(args.device),
                layer_output_importance.to(args.device),
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
                    layer_output_importance.to(args.device),
                    pass_count,
                )
                block = adapter.load_block(source, inventory.blocks[args.block_index].block, args.device)
                block.eval()
                refitted_module = _module(refitted_state, args.device, inputs.dtype)
                BlockEditor().install_trainable_layer(block, layer_path, refitted_module)
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
                        layer_output_importance.to(args.device),
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
            states = {"rewrite": rewrite}
            if legacy is not None:
                states["legacy"] = legacy
            for name in initializations:
                state = states[name]
                for epoch_count in epochs:
                    for binary_learning_rate in binary_learning_rates:
                        for checkpoint_policy in checkpoint_policies:
                            epoch_loss_mode = (
                                "legacy_training" if checkpoint_policy == "legacy_final" else "full_evaluation"
                            )
                            restore_best_state = checkpoint_policy != "legacy_final"
                            selection_inputs = inputs[fit_samples:] if checkpoint_policy == "heldout_best" else None
                            selection_targets = targets[fit_samples:] if checkpoint_policy == "heldout_best" else None
                            if args.device.startswith("cuda"):
                                torch.cuda.reset_peak_memory_stats(args.device)
                            block = adapter.load_block(source, inventory.blocks[args.block_index].block, args.device)
                            block.eval()
                            trainable = _module(state, args.device, inputs.dtype)
                            initial_left = torch.where(trainable.left_latent.detach() >= 0, 1, -1)
                            initial_right = torch.where(trainable.right_latent.detach() >= 0, 1, -1)
                            BlockEditor().install_trainable_layer(block, layer_path, trainable)
                            before_weight = _weighted_weight_metrics(
                                source_weight,
                                trainable,
                                input_importance.to(args.device),
                                layer_output_importance.to(args.device),
                            )
                            before_block = _block_loss(
                                adapter,
                                block,
                                inputs[:fit_samples],
                                targets[:fit_samples],
                                block_output_importance,
                                metadata,
                                args.block_forward_batch_size,
                            )
                            before_held_out = (
                                None
                                if fit_samples == args.samples
                                else _block_loss(
                                    adapter,
                                    block,
                                    inputs[fit_samples:],
                                    targets[fit_samples:],
                                    block_output_importance,
                                    metadata,
                                    args.block_forward_batch_size,
                                )
                            )
                            started = time.perf_counter()
                            trajectory: list[dict[str, float | int]] = []

                            def observe_epoch(
                                epoch: int,
                                loss: float,
                                observed: list[dict[str, float | int]] = trajectory,
                            ) -> None:
                                observed.append({"epoch": epoch, "loss": loss})

                            metrics = tune_factorized(
                                block,
                                layer_path,
                                TuningRequest(
                                    inputs[:fit_samples],
                                    targets[:fit_samples],
                                    epoch_count,
                                    args.batch_size,
                                    1e-5,
                                    output_importance=block_output_importance,
                                    seed=0,
                                    microbatch_size=args.microbatch_size,
                                    restore_best_state=restore_best_state,
                                    epoch_loss_mode=epoch_loss_mode,
                                    epoch_observer=observe_epoch,
                                    selection_inputs=selection_inputs,
                                    selection_targets=selection_targets,
                                ),
                                lambda module, value: adapter.run_block(module, value, **metadata),
                                learning_rates=FactorizedTuningLearningRates(
                                    binary_learning_rate,
                                    1e-5,
                                    1e-5,
                                    1e-5,
                                ),
                            )
                            after_weight = _weighted_weight_metrics(
                                source_weight,
                                trainable,
                                input_importance.to(args.device),
                                layer_output_importance.to(args.device),
                            )
                            after_held_out = (
                                None
                                if fit_samples == args.samples
                                else _block_loss(
                                    adapter,
                                    block,
                                    inputs[fit_samples:],
                                    targets[fit_samples:],
                                    block_output_importance,
                                    metadata,
                                    args.block_forward_batch_size,
                                )
                            )
                            final_left = torch.where(trainable.left_latent.detach() >= 0, 1, -1)
                            final_right = torch.where(trainable.right_latent.detach() >= 0, 1, -1)
                            row = {
                                "checkpoint_policy": checkpoint_policy,
                                "persisted_dtype": str(trainable.scale_pre.dtype).removeprefix("torch."),
                                "initialization": name,
                                "epochs": epoch_count,
                                "binary_learning_rate": binary_learning_rate,
                                "before_block_loss": before_block,
                                "before_held_out_block_loss": before_held_out,
                                "tuning_before_loss": (None if metrics.before is None else metrics.before.loss),
                                "best_loss": metrics.best.loss,
                                "final_loss": metrics.final.loss,
                                "after_held_out_block_loss": after_held_out,
                                "best_epoch": metrics.best_epoch,
                                "trajectory": trajectory,
                                "left_sign_changes": int((final_left != initial_left).sum()),
                                "right_sign_changes": int((final_right != initial_right).sum()),
                                "initial_weight": before_weight,
                                "final_weight": after_weight,
                                "wall_seconds": time.perf_counter() - started,
                                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
                                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)),
                                "final_cuda_allocated_bytes": int(torch.cuda.memory_allocated(args.device)),
                                "final_cuda_reserved_bytes": int(torch.cuda.memory_reserved(args.device)),
                            }
                            cast(list[object], payload["runs"]).append(row)
                            if args.kl_samples:
                                kl_reconstructions.append(
                                    trainable.dense_weight().detach().to(device="cpu").contiguous()
                                )
                            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                            print(json.dumps(row, indent=2, sort_keys=True))
                            del block, trainable
                            torch.cuda.empty_cache()
        if args.kl_samples:
            kl_tokens = calibration.input_ids[args.kl_offset : kl_stop, : args.kl_sequence_length].contiguous()
            layer = LayerId(BlockId(args.block_index), layer_path)
            splice_key = f"{args.block_index}:{layer_path}"
            teacher_state: tuple[float, tuple[torch.Tensor, ...]] | None = None
            kl_results: list[KlBudgetArmResult] = []
            runs = cast(list[dict[str, object]], payload["runs"])
            if len(kl_reconstructions) != len(runs):
                raise AssertionError("KL reconstruction inventory differs from tuning runs")
            for row, reconstruction in zip(runs, kl_reconstructions, strict=True):
                normalized_error = float(
                    cast(Any, cast(dict[str, object], row["final_weight"])["weighted_normalized_error"])
                )
                reconstruction_set = SpliceReconstructionSet(
                    (SpliceReconstruction(layer, reconstruction, None, normalized_error**2),),
                    ((splice_key, (layer,)),),
                    ((splice_key, normalized_error**2),),
                )
                evaluator = DenseKlSpliceEvaluator(
                    model,
                    reconstruction_set,
                    kl_tokens,
                    device=args.device,
                    batch_size=1,
                    token_chunk_size=128,
                    teacher_cache_mode="cpu",
                )
                if teacher_state is None:
                    teacher_state = evaluator.teacher_cache_state()
                else:
                    evaluator.install_teacher_cache(*teacher_state)
                kl_result = evaluator("full")
                kl_results.append(kl_result)
                row["language_model_splice"] = to_dict(kl_result)
                args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                del evaluator, reconstruction_set
                gc.collect()
                torch.cuda.empty_cache()
            baseline = kl_results[0]
            payload["language_model_splice_comparisons_to_first"] = [
                {
                    "run_index": index,
                    "negative_log_likelihood_delta": (
                        result.negative_log_likelihood - baseline.negative_log_likelihood
                    ),
                    "kl": to_dict(paired_bootstrap_kl_delta(baseline, result, seed=0)),
                }
                for index, result in enumerate(kl_results[1:], start=1)
            ]
        model.to("cpu")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
