"""Compare dense MLP scale refits with an equal-size factorized encoding."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import _paths  # noqa: F401
import torch
from evaluate_wikitext import _protocol_tokens
from probe_corrected_codebook_splice import _parse_block_policy
from probe_mlp_policy_frozen_transfer import (
    MODEL_SOURCE,
    PINNED_MODEL_REVISION,
    _evaluate_per_sequence,
    _install_dense_linear,
    _load_overlay,
    _paired_nll_payload,
)
from torch import nn

from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.config.codec import to_dict
from nanoquant.domain.linear_math import rescale_factorized_terms
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json, hash_file
from nanoquant.infrastructure.safetensors_io import SAFETENSORS
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--policy", type=_parse_block_policy, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-component-overlay", type=Path)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--wikitext-offset", type=int, default=0)
    parser.add_argument("--fit-iterations", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-global-tuning", action="store_true")
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Fit and export components without running redundant dense/factor screens.",
    )
    return parser


def _module_at_path(block: nn.Module, path: str) -> nn.Module:
    current = block
    for part in path.split("."):
        child = current[part] if isinstance(current, nn.ModuleDict) else getattr(current, part, None)
        if not isinstance(child, nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current


def _axis_scales(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    fit_rows: bool,
    fit_columns: bool,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit positive separable row/column scales by alternating least squares."""

    if source.ndim != 2 or target.shape != source.shape or iterations <= 0:
        raise ValueError("separable scale fit requires aligned matrices and positive iterations")
    source_float = source.float()
    target_float = target.to(device=source.device, dtype=torch.float32)
    rows = torch.ones(source.shape[0], device=source.device, dtype=torch.float32)
    columns = torch.ones(source.shape[1], device=source.device, dtype=torch.float32)

    def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        fitted = torch.where(denominator > 0, numerator / denominator, torch.ones_like(denominator))
        return fitted.clamp_(min=0.01, max=100.0)

    for _ in range(iterations):
        if fit_rows:
            column_scaled = source_float * columns.reshape(1, -1)
            rows = _ratio(
                (column_scaled * target_float).sum(dim=1),
                column_scaled.square().sum(dim=1),
            )
        if fit_columns:
            row_scaled = source_float * rows.reshape(-1, 1)
            columns = _ratio(
                (row_scaled * target_float).sum(dim=0),
                row_scaled.square().sum(dim=0),
            )
    return rows, columns


def _payload_bytes(module: FactorizedReferenceLinear) -> int:
    names = (
        "left_binary",
        "right_binary",
        "scale_pre",
        "scale_mid",
        "scale_post",
        "outlier_indices",
        "outlier_values",
        "outlier_scales",
        "patch_left",
        "patch_right",
        "bias",
    )
    return sum(
        value.numel() * value.element_size()
        for name in names
        if isinstance((value := getattr(module, name)), torch.Tensor)
    )


def _rescaled_payload_bytes(module: FactorizedReferenceLinear) -> int:
    return sum(
        value.numel() * value.element_size()
        for name in ("scale_pre", "scale_post", "outlier_values", "patch_left", "patch_right")
        if isinstance((value := getattr(module, name)), torch.Tensor)
    )


def _apply_factorized_rescale(
    module: FactorizedReferenceLinear,
    target: torch.Tensor,
    *,
    fit_rows: bool,
    fit_columns: bool,
    iterations: int,
) -> dict[str, object]:
    source = module.dense_weight().detach()
    target_device = target.to(device=source.device, dtype=source.dtype)
    baseline_error = (source.float() - target_device.float()).square().mean().sqrt()
    rows, columns = _axis_scales(
        source,
        target_device,
        fit_rows=fit_rows,
        fit_columns=fit_columns,
        iterations=iterations,
    )
    before_bytes = _payload_bytes(module)
    rescaled_bytes_before = _rescaled_payload_bytes(module)
    scaled = rescale_factorized_terms(
        module.scale_pre,
        module.scale_post,
        input_multiplier=columns,
        output_multiplier=rows,
        outlier_indices=module.outlier_indices,
        outlier_values=module.outlier_values,
        patch_left=module.patch_left,
        patch_right=module.patch_right,
    )
    with torch.no_grad():
        module.scale_pre.copy_(scaled.scale_pre)
        module.scale_post.copy_(scaled.scale_post)
        if module.outlier_values is not None and scaled.outlier_values is not None:
            module.outlier_values.copy_(scaled.outlier_values)
        if module.patch_left is not None and scaled.patch_left is not None:
            module.patch_left.copy_(scaled.patch_left)
        if module.patch_right is not None and scaled.patch_right is not None:
            module.patch_right.copy_(scaled.patch_right)
    reconstructed = module.dense_weight().detach()
    difference = reconstructed.float() - target_device.float()
    target_rms = target_device.float().square().mean().sqrt()
    after_bytes = _payload_bytes(module)
    rescaled_bytes_after = _rescaled_payload_bytes(module)
    result = {
        "fit_rows": fit_rows,
        "fit_columns": fit_columns,
        "input_multiplier_minimum": float(columns.min()),
        "input_multiplier_maximum": float(columns.max()),
        "output_multiplier_minimum": float(rows.min()),
        "output_multiplier_maximum": float(rows.max()),
        "baseline_target_rmse": float(baseline_error),
        "factor_target_rmse": float(difference.square().mean().sqrt()),
        "factor_target_normalized_rmse": float(difference.square().mean().sqrt() / target_rms),
        "factor_target_maximum_absolute_error": float(difference.abs().max()),
        "payload_bytes_before": before_bytes,
        "payload_bytes_after": after_bytes,
        "payload_byte_delta": after_bytes - before_bytes,
        "rescaled_payload_bytes_before": rescaled_bytes_before,
        "rescaled_payload_bytes_after": rescaled_bytes_after,
    }
    del source, target_device, reconstructed, difference
    return result


def _fit_axes(path: str, choice: str) -> tuple[bool, bool]:
    if path in {"mlp.gate_proj", "mlp.up_proj"}:
        return choice != "base", False
    if path != "mlp.down_proj":
        raise ValueError(f"unsupported MLP projection: {path}")
    return choice in {"output", "joint"}, choice in {"input", "joint"}


def _component_tensors(
    tensor_name: str,
    module: FactorizedReferenceLinear,
) -> dict[str, torch.Tensor]:
    prefix = tensor_name.removesuffix(".weight")
    values = {
        f"{prefix}.scale_pre": module.scale_pre,
        f"{prefix}.scale_post": module.scale_post,
    }
    for name in ("outlier_values", "patch_left", "patch_right"):
        value = getattr(module, name)
        if isinstance(value, torch.Tensor):
            values[f"{prefix}.{name}"] = value
    return {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in values.items()
    }


def _export_component_overlay(
    destination: Path,
    tensors: dict[str, torch.Tensor],
    *,
    dense_manifest: dict[str, object],
    identity: dict[str, str],
    global_tuning: dict[str, object] | None,
    policy: tuple[tuple[int, str], ...],
    replaced_payload_bytes: int,
) -> dict[str, object]:
    if not tensors or any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("factor-compatible component overlay tensors are invalid")
    replacement_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    with atomic_workspace(destination) as temporary:
        tensor_path = temporary / "components.safetensors"
        SAFETENSORS.save(tensors, tensor_path)
        manifest = {
            "schema_version": 2,
            "semantics": "replace-existing-factorized-components",
            "source_dense_tensor_sha256": dense_manifest["tensor_sha256"],
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "policy": {str(block): choice for block, choice in policy},
            "tensor_sha256": hash_file(tensor_path),
            "tensor_count": len(tensors),
            "replaced_payload_bytes": replaced_payload_bytes,
            "replacement_payload_bytes": replacement_bytes,
            "payload_byte_delta": replacement_bytes - replaced_payload_bytes,
            "tensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
                for name, value in sorted(tensors.items())
            },
        }
        atomic_write_json(temporary / "manifest.json", manifest)
    return {"directory": str(destination), **manifest}


def run(args: argparse.Namespace) -> int:
    if (
        args.samples <= 0
        or args.sequence_length <= 1
        or args.wikitext_offset < 0
        or args.fit_iterations <= 0
    ):
        raise ValueError("factor-compatible MLP refit protocol is invalid")
    tensors, manifest = _load_overlay(args.overlay)
    policy = dict(args.policy)
    if set(policy) != set(manifest["blocks"]):
        raise ValueError("factor-compatible policy must choose every overlay block")
    if args.skip_evaluation:
        tokens = None
        dataset_fingerprint = None
        bos_token_id = None
    elif args.wikitext_offset == 0:
        tokens, dataset_fingerprint, bos_token_id = _protocol_tokens(
            args.snapshot,
            args.samples,
            args.sequence_length,
        )
    else:
        all_tokens, dataset_fingerprint, bos_token_id = _wikitext_tokens(
            args.snapshot,
            samples=args.wikitext_offset + args.samples,
            sequence_length=args.sequence_length,
            local_files_only=False,
        )
        tokens = all_tokens[args.wikitext_offset : args.wikitext_offset + args.samples]

    with acquire_device_lease(args.device):
        loaded = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name=MODEL_SOURCE,
            revision=args.model_revision,
            device=args.device,
            verify_hashes=False,
            backend="factorized",
            use_global_tuning=args.use_global_tuning,
        )
        loaded.model.eval()
        baseline = (
            None
            if tokens is None
            else _evaluate_per_sequence(loaded.model, tokens, args.device)
        )
        base = getattr(loaded.model, "model", None)
        decoder = getattr(base, "layers", None)
        if not isinstance(decoder, nn.ModuleList):
            raise TypeError("frozen model does not expose decoder blocks")

        layers = {}
        component_tensors = {}
        for tensor_name, target in tensors.items():
            logical = tensor_name.removeprefix("model.layers.").removesuffix(".weight")
            block_text, path = logical.split(".", maxsplit=1)
            block_index = int(block_text)
            module = _module_at_path(decoder[block_index], path)
            if not isinstance(module, FactorizedReferenceLinear):
                raise TypeError(f"MLP projection is not factorized: {tensor_name}")
            fit_rows, fit_columns = _fit_axes(path, policy[block_index])
            layers[tensor_name] = _apply_factorized_rescale(
                module,
                target,
                fit_rows=fit_rows,
                fit_columns=fit_columns,
                iterations=args.fit_iterations,
            )
            component_tensors.update(_component_tensors(tensor_name, module))
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        factor_candidate = (
            None
            if tokens is None
            else _evaluate_per_sequence(loaded.model, tokens, args.device)
        )
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        component_overlay = (
            None
            if args.export_component_overlay is None
            else _export_component_overlay(
                args.export_component_overlay,
                component_tensors,
                dense_manifest=manifest,
                identity=identity,
                global_tuning=(
                    None
                    if loaded.global_tuning is None
                    else to_dict(loaded.global_tuning)
                ),
                policy=args.policy,
                replaced_payload_bytes=sum(
                    int(item["rescaled_payload_bytes_before"])
                    for item in layers.values()
                ),
            )
        )
        dense_candidate = None
        if tokens is not None:
            for tensor_name, target in tensors.items():
                logical = tensor_name.removeprefix("model.layers.").removesuffix(".weight")
                block_text, path = logical.split(".", maxsplit=1)
                _install_dense_linear(decoder[int(block_text)], path, target, device=args.device)
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            dense_candidate = _evaluate_per_sequence(loaded.model, tokens, args.device)

    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only equal-size factor-compatible MLP scale refit",
            "model_revision": args.model_revision,
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "use_global_tuning": args.use_global_tuning,
            "overlay": {"directory": str(args.overlay), **manifest},
            "component_overlay": component_overlay,
            "policy": {str(block): choice for block, choice in args.policy},
            "protocol": {
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "wikitext_offset": args.wikitext_offset,
                "fit_iterations": args.fit_iterations,
                "dataset_fingerprint": dataset_fingerprint,
                "bos_token_id": bos_token_id,
                "token_hash": None if tokens is None else _token_hash(tokens),
                "evaluation_skipped": args.skip_evaluation,
            },
            "layers": layers,
            "payload_byte_delta": sum(int(item["payload_byte_delta"]) for item in layers.values()),
            "baseline": baseline,
            "factor_candidate": factor_candidate,
            "dense_candidate": dense_candidate,
            "paired_factor_minus_baseline_nll": (
                None
                if baseline is None or factor_candidate is None
                else _paired_nll_payload(baseline, factor_candidate)
            ),
            "paired_dense_minus_baseline_nll": (
                None
                if baseline is None or dense_candidate is None
                else _paired_nll_payload(baseline, dense_candidate)
            ),
            "paired_factor_minus_dense_nll": (
                None
                if dense_candidate is None or factor_candidate is None
                else _paired_nll_payload(dense_candidate, factor_candidate)
            ),
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
