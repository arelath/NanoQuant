"""Regenerate exact v3 product components and fold accepted scale corrections."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from materialize_product_codebook_mixed_policy import (
    MLP_PATHS,
    MaterializationJob,
    build_probe_command,
    load_policy_jobs,
)
from safetensors import safe_open

from nanoquant.domain.scale_fit import reconstruct
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.runtime import (
    PRODUCT_CODEBOOK_FORMAT_VERSION,
    ProductCodebookLayerState,
    open_packed_artifact,
    open_product_codebook_artifact,
    write_product_codebook_artifact,
)

COMPONENT_FORMAT = "product-codebook-factor-components"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON receipt must be an object: {path}")
    return cast(dict[str, Any], payload)


def fit_row_multiplier(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if base.shape != target.shape or base.ndim != 2:
        raise ValueError("row-multiplier tensors must be same-shaped matrices")
    base32 = base.float()
    target32 = target.float()
    denominator = base32.square().sum(dim=1)
    numerator = (base32 * target32).sum(dim=1)
    return torch.where(denominator > 0, numerator / denominator, torch.ones_like(numerator))


def fit_separable_multipliers(
    base: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if base.shape != target.shape or base.ndim != 2 or iterations <= 0:
        raise ValueError("separable-fit inputs are invalid")
    base32 = base.float()
    target32 = target.float()
    columns = torch.ones(base.shape[1], dtype=torch.float32)
    rows = torch.ones(base.shape[0], dtype=torch.float32)
    for _ in range(iterations):
        scaled = base32 * columns.reshape(1, -1)
        denominator = scaled.square().sum(dim=1)
        rows = torch.where(
            denominator > 0,
            (scaled * target32).sum(dim=1) / denominator,
            torch.ones_like(denominator),
        )
        scaled = base32 * rows.reshape(-1, 1)
        denominator = scaled.square().sum(dim=0)
        columns = torch.where(
            denominator > 0,
            (scaled * target32).sum(dim=0) / denominator,
            torch.ones_like(denominator),
        )
    return rows.contiguous(), columns.contiguous()


def _component_manifest(directory: Path, job: MaterializationJob) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    tensor_path = directory / "components.safetensors"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"component bundle is incomplete: {directory}")
    manifest = _read_json(manifest_path)
    layers = manifest.get("layers")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != COMPONENT_FORMAT
        or manifest.get("layer_count") != len(job.projections)
        or manifest.get("tensor_sha256") != hash_file(tensor_path)
        or not isinstance(layers, list)
    ):
        raise ValueError(f"component bundle identity differs: {directory}")
    observed = {
        (layer.get("block"), layer.get("projection"))
        for layer in layers
        if isinstance(layer, dict)
    }
    expected = {(job.block, projection) for projection in job.projections}
    if observed != expected:
        raise ValueError(f"component layer inventory differs: {directory}")
    return manifest


def _validate_job(
    receipt_path: Path,
    component_dir: Path,
    job: MaterializationJob,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    exported = receipt.get("product_component_export")
    if (
        receipt.get("status") != "completed"
        or receipt.get("blocks") != [job.block]
        or not isinstance(exported, dict)
        or Path(str(exported.get("directory"))).resolve() != component_dir.resolve()
    ):
        raise ValueError(f"component replay job receipt differs: {receipt_path}")
    return _component_manifest(component_dir, job)


def _load_component_layer(
    component_dir: Path,
    layer: dict[str, Any],
) -> dict[str, torch.Tensor]:
    inventory = layer.get("tensors")
    if not isinstance(inventory, dict):
        raise ValueError("component tensor inventory is absent")
    tensor_path = component_dir / "components.safetensors"
    values = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        for role, metadata in inventory.items():
            if not isinstance(role, str) or not isinstance(metadata, dict):
                raise ValueError("component tensor metadata is invalid")
            key = metadata.get("key")
            if not isinstance(key, str) or key not in handle.keys():
                raise ValueError("component tensor key is invalid")
            value = handle.get_tensor(key).contiguous()
            if list(value.shape) != metadata.get("shape") or str(value.dtype).removeprefix(
                "torch."
            ) != metadata.get("dtype"):
                raise ValueError("component tensor shape or dtype differs")
            values[role] = value
    return values


def _correction_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = predicted.float() - target.float()
    return {
        "rmse": float(difference.square().mean().sqrt()),
        "maximum_absolute_error": float(difference.abs().max()),
        "exact_element_fraction": float((predicted == target).float().mean()),
    }


def _physical_reconstruction(state: ProductCodebookLayerState) -> torch.Tensor:
    logical = state.to_packed().to_logical()
    result = reconstruct(
        logical.left_binary,
        logical.right_binary,
        logical.scale_pre,
        logical.scale_mid,
        logical.scale_post,
    )
    if logical.outlier_indices is not None:
        assert logical.outlier_values is not None
        result[:, logical.outlier_indices.to(torch.int64)] = logical.outlier_values.to(result)
    return result.to(torch.bfloat16).contiguous()


def fold_component_correction(
    *,
    spec: Any,
    layer: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    row_multiplier: torch.Tensor,
    column_multiplier: torch.Tensor | None,
) -> ProductCodebookLayerState:
    transposed = bool(layer["factorization_transposed"])
    pre = tensors["factor_scale_pre"].float().clone()
    middle = tensors["factor_scale_mid"].float().clone()
    post = tensors["factor_scale_post"].float().clone()
    indices = tensors["outlier_indices"].to(torch.int32).contiguous()
    outliers = tensors["outlier_values"].float().clone()
    if transposed:
        if column_multiplier is not None or pre.shape != row_multiplier.shape:
            raise ValueError("transposed product correction axes differ")
        pre *= row_multiplier
        post[indices.to(torch.int64)] = 0
        outliers *= row_multiplier.reshape(-1, 1)
    else:
        if column_multiplier is None or post.shape != row_multiplier.shape or pre.shape != column_multiplier.shape:
            raise ValueError("source-orientation product correction axes differ")
        pre *= column_multiplier
        post *= row_multiplier
        pre[indices.to(torch.int64)] = 0
        outliers *= row_multiplier.reshape(-1, 1)
        outliers *= column_multiplier.index_select(0, indices.to(torch.int64)).reshape(1, -1)
    return ProductCodebookLayerState(
        spec,
        PRODUCT_CODEBOOK_FORMAT_VERSION,
        transposed,
        int(layer["right_free_rows"]),
        tensors["factor_left_words"].to(torch.int32).contiguous(),
        tensors["factor_right_free_words"].to(torch.int32).contiguous(),
        tensors["factor_right_coded_payload"].to(torch.int32).contiguous(),
        tensors["factor_right_first_half_words"].to(torch.int16).contiguous(),
        tensors["factor_right_second_half_words"].to(torch.int16).contiguous(),
        pre.to(torch.bfloat16).contiguous(),
        middle.to(torch.bfloat16).contiguous(),
        post.to(torch.bfloat16).contiguous(),
        indices,
        outliers.to(torch.bfloat16).contiguous(),
    )


def _assemble(
    args: argparse.Namespace,
    allocation: dict[str, Any],
    jobs: tuple[MaterializationJob, ...],
) -> dict[str, Any]:
    base = open_packed_artifact(args.base_packed, verify_hashes=True)
    specs = {
        layer.spec.name: layer.spec
        for block in base.manifest.blocks
        for layer in block.layers
    }
    replacements: dict[int, list[ProductCodebookLayerState]] = {}
    replay_metrics = {}
    maximum_rmse = 0.0
    maximum_absolute_error = 0.0
    with safe_open(args.base_overlay / "weights.safetensors", framework="pt", device="cpu") as base_handle, safe_open(
        args.target_overlay / "weights.safetensors", framework="pt", device="cpu"
    ) as target_handle:
        for job in jobs:
            component_dir = args.output_dir / "components" / job.job_id
            manifest = _component_manifest(component_dir, job)
            for layer_value in cast(list[dict[str, Any]], manifest["layers"]):
                projection = str(layer_value["projection"])
                block = int(layer_value["block"])
                tensor_name = str(layer_value["tensor_name"])
                packed_name = f"blocks.{block}.{MLP_PATHS[projection]}"
                source = base_handle.get_tensor(tensor_name)
                target = target_handle.get_tensor(tensor_name)
                if projection in {"gate", "up"}:
                    rows = fit_row_multiplier(source, target)
                    columns = None
                    separable = source.float() * rows.reshape(-1, 1)
                else:
                    rows, columns = fit_separable_multipliers(source, target)
                    separable = source.float() * rows.reshape(-1, 1) * columns.reshape(1, -1)
                fit_metrics = _correction_metrics(separable.to(torch.bfloat16), target)
                state = fold_component_correction(
                    spec=specs[packed_name],
                    layer=layer_value,
                    tensors=_load_component_layer(component_dir, layer_value),
                    row_multiplier=rows,
                    column_multiplier=columns,
                )
                packed_metrics = _correction_metrics(_physical_reconstruction(state), target)
                replay_metrics[packed_name] = {
                    "separable_dense_fit": fit_metrics,
                    "packed_component_replay": packed_metrics,
                }
                maximum_rmse = max(maximum_rmse, packed_metrics["rmse"])
                maximum_absolute_error = max(
                    maximum_absolute_error,
                    packed_metrics["maximum_absolute_error"],
                )
                replacements.setdefault(block, []).append(state)
                del source, target, separable
    expected_bits = sum(int(value["bits"]) for value in allocation["selections"])
    observed_bits = sum(
        state.compact_logical_bits() for states in replacements.values() for state in states
    )
    if sum(len(states) for states in replacements.values()) != 78 or observed_bits != expected_bits:
        raise ValueError(
            f"component replay inventory/bits differ: layers={sum(len(v) for v in replacements.values())} "
            f"bits={observed_bits} expected={expected_bits}"
        )
    overlay = write_product_codebook_artifact(
        args.output_dir / "packed-overlay",
        base,
        replacements,
        allocation_sha256=hash_file(args.allocation),
        allocation_total_bits=int(allocation["total_bits"]),
        effective_bpw=float(allocation["effective_bpw"]),
        correction_source_sha256=hash_file(args.target_overlay / "weights.safetensors"),
        replay={
            "maximum_rmse": maximum_rmse,
            "maximum_absolute_error": maximum_absolute_error,
            "layer_metrics": replay_metrics,
        },
    )
    return {
        "overlay": str(overlay.root),
        "layer_count": overlay.manifest.layer_count,
        "compact_mlp_bits": observed_bits,
        "allocation_total_bits": int(allocation["total_bits"]),
        "effective_bpw": float(allocation["effective_bpw"]),
        "maximum_rmse": maximum_rmse,
        "maximum_absolute_error": maximum_absolute_error,
        "replay_metrics": replay_metrics,
    }


def _identity(args: argparse.Namespace, allocation: dict[str, Any]) -> dict[str, Any]:
    return {
        "allocation": str(args.allocation.resolve()),
        "allocation_sha256": hash_file(args.allocation),
        "allocation_total_bits": int(allocation["total_bits"]),
        "base_packed": str(args.base_packed.resolve()),
        "base_packed_descriptor_sha256": hash_file(
            args.base_packed / "nanoquant-packed-model.json"
        ),
        "base_overlay": str(args.base_overlay.resolve()),
        "base_overlay_sha256": hash_file(args.base_overlay / "weights.safetensors"),
        "target_overlay": str(args.target_overlay.resolve()),
        "target_overlay_sha256": hash_file(args.target_overlay / "weights.safetensors"),
        "model": str(args.model.resolve()),
        "snapshot": str(args.snapshot.resolve()),
        "calibration_state": str(args.calibration_state.resolve()),
        "reconstruction_cache": str(args.reconstruction_cache.resolve()),
        "model_revision": args.model_revision,
        "device": args.device,
    }


def run(args: argparse.Namespace) -> int:
    allocation, jobs = load_policy_jobs(args.allocation)
    identity = _identity(args, allocation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts = args.output_dir / "jobs"
    components = args.output_dir / "components"
    receipts.mkdir(exist_ok=True)
    components.mkdir(exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    previous = _read_json(manifest_path) if manifest_path.is_file() else None
    if previous is not None and previous.get("identity") != identity:
        raise ValueError("existing component replay has a different identity")
    if previous is not None and previous.get("status") == "completed":
        open_product_codebook_artifact(
            args.output_dir / "packed-overlay", args.base_packed, verify_hashes=True
        )
        return 0
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": previous.get("started_at", _utc_now()) if previous else _utc_now(),
        "updated_at": _utc_now(),
        "identity": identity,
        "job_count": len(jobs),
        "completed_jobs": previous.get("completed_jobs", []) if previous else [],
    }
    atomic_write_json(manifest_path, summary)
    probe = Path(__file__).resolve().with_name("probe_corrected_codebook_splice.py")
    try:
        for job in jobs:
            receipt = receipts / f"{job.job_id}.json"
            component_dir = components / job.job_id
            if receipt.is_file() and component_dir.is_dir():
                _validate_job(receipt, component_dir, job)
            else:
                summary["current_job"] = job.job_id
                summary["updated_at"] = _utc_now()
                atomic_write_json(manifest_path, summary)
                print(f"starting exact components {job.job_id}", flush=True)
                command = build_probe_command(
                    job,
                    python=Path(sys.executable).resolve(),
                    probe=probe,
                    model=args.model.resolve(),
                    snapshot=args.snapshot.resolve(),
                    calibration_state=args.calibration_state.resolve(),
                    output=receipt.resolve(),
                    reconstruction_cache=args.reconstruction_cache.resolve(),
                    model_revision=args.model_revision,
                    device=args.device,
                    product_component_output=component_dir.resolve(),
                )
                subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
                _validate_job(receipt, component_dir, job)
            completed = cast(list[str], summary["completed_jobs"])
            if job.job_id not in completed:
                completed.append(job.job_id)
            summary["updated_at"] = _utc_now()
            atomic_write_json(manifest_path, summary)
            print(f"completed exact components {job.job_id}", flush=True)
        packed = args.output_dir / "packed-overlay"
        result = (
            {
                "overlay": str(packed.resolve()),
                "layer_count": open_product_codebook_artifact(
                    packed, args.base_packed, verify_hashes=True
                ).manifest.layer_count,
            }
            if packed.exists()
            else _assemble(args, allocation, jobs)
        )
        summary["component_replay"] = result
        summary["status"] = "completed"
        summary["completed_at"] = _utc_now()
        summary["updated_at"] = summary["completed_at"]
        summary.pop("current_job", None)
        atomic_write_json(manifest_path, summary)
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["updated_at"] = _utc_now()
        atomic_write_json(manifest_path, summary)
        raise
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--reconstruction-cache", type=Path, required=True)
    parser.add_argument("--base-packed", type=Path, required=True)
    parser.add_argument("--base-overlay", type=Path, required=True)
    parser.add_argument("--target-overlay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
