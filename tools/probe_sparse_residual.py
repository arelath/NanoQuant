"""Compare equal-bit sparse-entry and column patches on factorization residuals.

This analysis-only probe uses production ADMM and scale fitting. Each patch arm
starts from the same rank-adjusted reconstruction, so the comparison isolates
the residual representation rather than allowing factorization to adapt to one
patch shape. Results are checkpointed after every layer and patch budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from nanoquant.domain.calibration_math import shrink_importance
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.models import BitCost
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease

PINNED_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
PROJECTION_PATHS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
SUPPORTED_PROJECTIONS = tuple(PROJECTION_PATHS)


@dataclass(frozen=True, slots=True)
class SparseResidualProtocol:
    schema_version: int
    model_revision: str
    target_bpw: float
    rank_alignment: int
    scale_bits: int
    value_bits: int
    index_bits: int
    outer_iterations: int
    inner_iterations: int
    regularization: float
    penalty_schedule: str
    convergence_check_interval: int
    transpose_wide: bool
    scale_fit_passes: int
    seed: int
    device: str
    calibration_state: str
    calibration_shrinkage: float


def maximum_rank_for_budget(
    out_features: int,
    in_features: int,
    target_bits: int,
    *,
    scale_bits: int,
    rank_alignment: int,
    fixed_bits: int = 0,
) -> int:
    if min(out_features, in_features, target_bits, scale_bits, fixed_bits) < 0 or rank_alignment <= 0:
        raise ValueError("rank budget inputs are invalid")
    accepted = 0
    for rank in range(1, min(out_features, in_features) + 1):
        cost = factor_bit_cost(
            out_features,
            in_features,
            rank,
            scale_bits=scale_bits,
            rank_alignment=rank_alignment,
        ).total
        if cost + fixed_bits > target_bits:
            if rank_alignment == 1:
                break
            continue
        accepted = rank
    if accepted == 0:
        raise ValueError("target bit budget cannot fund rank one")
    return accepted


def load_calibration_profiles(
    state_directory: Path,
    shrinkage: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    manifest = json.loads((state_directory / "manifest.json").read_text(encoding="utf-8"))
    sample_count = int(manifest["sample_count"])
    layers = manifest.get("layers")
    if sample_count <= 0 or not isinstance(layers, list) or not layers:
        raise ValueError("calibration state manifest is invalid")
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with safe_open(str(state_directory / "state.safetensors"), framework="pt", device="cpu") as handle:
        for index, layer in enumerate(layers):
            path = str(layer["path"])
            result[path] = (
                shrink_importance(
                    handle.get_tensor(f"layer_{index}.inputs.total").float() / sample_count,
                    shrinkage,
                ),
                shrink_importance(
                    handle.get_tensor(f"layer_{index}.outputs.total").float() / sample_count,
                    shrinkage,
                ),
            )
    return result


def column_patch_bit_cost(
    out_features: int,
    count: int,
    *,
    value_bits: int = 16,
    index_bits: int = 32,
) -> BitCost:
    if min(out_features, count, value_bits, index_bits) < 0:
        raise ValueError("column patch cost inputs must not be negative")
    return BitCost(
        outlier_value_bits=out_features * count * value_bits,
        outlier_index_bits=count * index_bits,
    )


def sparse_entry_bit_cost(
    count: int,
    *,
    value_bits: int = 16,
    index_bits: int = 32,
) -> BitCost:
    if min(count, value_bits, index_bits) < 0:
        raise ValueError("sparse patch cost inputs must not be negative")
    return BitCost(
        outlier_value_bits=count * value_bits,
        outlier_index_bits=count * index_bits,
    )


def matched_sparse_entry_count(
    column_bits: int,
    *,
    value_bits: int = 16,
    index_bits: int = 32,
) -> int:
    per_entry = value_bits + index_bits
    if column_bits < 0 or per_entry <= 0:
        raise ValueError("matched sparse-entry budget is invalid")
    return column_bits // per_entry


def rank_for_total_budget(
    out_features: int,
    in_features: int,
    target_bits: int,
    patch_bits: int,
    *,
    scale_bits: int,
    rank_alignment: int,
) -> int:
    return maximum_rank_for_budget(
        out_features,
        in_features,
        target_bits,
        scale_bits=scale_bits,
        rank_alignment=rank_alignment,
        fixed_bits=patch_bits,
    )


def _parse_ints(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    result = tuple(int(item.strip()) for item in value.split(","))
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("integer lists must not contain negative values")
    return result


def _parse_projections(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(result) - set(SUPPORTED_PROJECTIONS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported projections: {', '.join(unknown)}")
    return result


def _protocol_hash(protocol: SparseResidualProtocol) -> str:
    encoded = json.dumps(asdict(protocol), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_output(path: Path, protocol: SparseResidualProtocol) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "protocol_hash": _protocol_hash(protocol),
            "protocol": asdict(protocol),
            "results": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_hash") != _protocol_hash(protocol):
        raise ValueError("existing output uses a different sparse-residual protocol")
    if not isinstance(payload.get("results"), dict):
        raise ValueError("existing output is missing its result map")
    return payload


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _logical_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _tensor_name(block: int, projection: str) -> str:
    return f"model.layers.{block}.{PROJECTION_PATHS[projection]}.weight"


def _calibration_path(block: int, projection: str) -> str:
    return f"block.{block}.{PROJECTION_PATHS[projection]}"


def _metrics(
    weight: torch.Tensor,
    difference: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> dict[str, float]:
    difference32 = difference.float()
    weight32 = weight.float()
    element_importance = output_importance.float()[:, None] * input_importance.float()[None, :]
    raw_error = float(difference32.square().sum())
    raw_target = float(weight32.square().sum())
    weighted_error = float((difference32.square() * element_importance).sum())
    weighted_target = float((weight32.square() * element_importance).sum())
    return {
        "raw_error_energy": raw_error,
        "raw_target_energy": raw_target,
        "raw_normalized_rmse": math.sqrt(raw_error / max(raw_target, 1e-30)),
        "weighted_error_energy": weighted_error,
        "weighted_target_energy": weighted_target,
        "weighted_normalized_rmse": math.sqrt(weighted_error / max(weighted_target, 1e-30)),
    }


def _indices_hash(indices: torch.Tensor) -> str:
    ordered = indices.detach().to(device="cpu", dtype=torch.int64).sort().values
    return "sha256:" + hashlib.sha256(ordered.numpy().tobytes()).hexdigest()


def evaluate_residual_patches(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    column_count: int,
    sparse_count: int,
    *,
    value_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Select oracle diagonal-Fisher patches and return scalar evidence."""

    if column_count < 0 or column_count > weight.shape[1]:
        raise ValueError("column patch count is outside the input dimension")
    if sparse_count < 0 or sparse_count > weight.numel():
        raise ValueError("sparse patch count is outside the matrix")
    residual = weight.float() - reconstruction.float()
    scores = (
        residual.square()
        * output_importance.float()[:, None]
        * input_importance.float()[None, :]
    )
    baseline = _metrics(weight, residual, input_importance, output_importance)

    if column_count:
        column_scores = scores.sum(dim=0)
        column_indices = torch.topk(column_scores, column_count, sorted=False).indices
        column_difference = residual.clone()
        stored_columns = residual[:, column_indices].to(value_dtype).float()
        column_difference[:, column_indices] -= stored_columns
    else:
        column_indices = torch.empty(0, device=weight.device, dtype=torch.int64)
        column_difference = residual
    column_metrics = _metrics(weight, column_difference, input_importance, output_importance)

    if sparse_count:
        flat_indices = torch.topk(scores.reshape(-1), sparse_count, sorted=False).indices
        sparse_difference = residual.clone().reshape(-1)
        stored_entries = residual.reshape(-1)[flat_indices].to(value_dtype).float()
        sparse_difference[flat_indices] -= stored_entries
        sparse_difference = sparse_difference.reshape_as(residual)
        rows = torch.div(flat_indices, weight.shape[1], rounding_mode="floor")
        columns = flat_indices.remainder(weight.shape[1])
        unique_rows = int(rows.unique().numel())
        unique_columns = int(columns.unique().numel())
    else:
        flat_indices = torch.empty(0, device=weight.device, dtype=torch.int64)
        sparse_difference = residual
        unique_rows = 0
        unique_columns = 0
    sparse_metrics = _metrics(weight, sparse_difference, input_importance, output_importance)

    return {
        "unpatched": baseline,
        "columns": {
            "count": column_count,
            "indices_hash": _indices_hash(column_indices),
            "metrics": column_metrics,
        },
        "sparse_entries": {
            "count": sparse_count,
            "indices_hash": _indices_hash(flat_indices),
            "unique_rows": unique_rows,
            "unique_columns": unique_columns,
            "metrics": sparse_metrics,
        },
    }


def _factorize(
    weight: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    rank: int,
    protocol: SparseResidualProtocol,
    key: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device=protocol.device).manual_seed(_logical_seed(protocol.seed, key))
    parameters = AdmmParameters(
        outer_iterations=protocol.outer_iterations,
        inner_iterations=protocol.inner_iterations,
        regularization=protocol.regularization,
        penalty_schedule=protocol.penalty_schedule,
        convergence_check_interval=protocol.convergence_check_interval,
        transpose_wide=protocol.transpose_wide,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(protocol.device)
        torch.cuda.synchronize(protocol.device)
    started = time.perf_counter()
    factorized = factorize_admm_with_parameters(
        weight,
        input_importance,
        output_importance,
        rank,
        generator,
        parameters,
    )
    fitted = fit_scales(
        weight,
        factorized.left_binary,
        factorized.right_binary,
        factorized.scale_pre,
        factorized.scale_mid,
        factorized.scale_post,
        input_importance,
        output_importance,
        alternating_passes=protocol.scale_fit_passes,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.synchronize(protocol.device)
    details = {
        "iterations_completed": factorized.iterations_completed,
        "scale_fit_accepted": fitted.accepted,
        "scale_fit_rollback_reason": fitted.rollback_reason,
        "wall_seconds": time.perf_counter() - started,
        "peak_device_bytes": (
            int(torch.cuda.max_memory_allocated(protocol.device))
            if protocol.device.startswith("cuda")
            else 0
        ),
    }
    reconstruction = fitted.reconstruction.detach().clone()
    del factorized, fitted
    return reconstruction, details


def _execute_layer_budget(
    handle: Any,
    block: int,
    projection: str,
    column_count: int,
    protocol: SparseResidualProtocol,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    tensor_name = _tensor_name(block, projection)
    weight = handle.get_tensor(tensor_name).to(protocol.device)
    out_features, in_features = weight.shape
    source_elements = weight.numel()
    target_bits = math.floor(source_elements * protocol.target_bpw)
    column_cost = column_patch_bit_cost(
        out_features,
        column_count,
        value_bits=protocol.value_bits,
        index_bits=protocol.index_bits,
    )
    sparse_count = matched_sparse_entry_count(
        column_cost.total,
        value_bits=protocol.value_bits,
        index_bits=protocol.index_bits,
    )
    sparse_cost = sparse_entry_bit_cost(
        sparse_count,
        value_bits=protocol.value_bits,
        index_bits=protocol.index_bits,
    )
    rank = rank_for_total_budget(
        out_features,
        in_features,
        target_bits,
        column_cost.total,
        scale_bits=protocol.scale_bits,
        rank_alignment=protocol.rank_alignment,
    )
    factor_cost = factor_bit_cost(
        out_features,
        in_features,
        rank,
        scale_bits=protocol.scale_bits,
        rank_alignment=protocol.rank_alignment,
    )
    try:
        input_importance_cpu, output_importance_cpu = profiles[_calibration_path(block, projection)]
    except KeyError as exc:
        raise ValueError(f"calibration state is missing block {block} {projection}") from exc
    input_importance = input_importance_cpu.to(protocol.device).float()
    output_importance = output_importance_cpu.to(protocol.device).float()
    key = f"{block}|{projection}|columns={column_count}|rank={rank}"
    reconstruction, factorization = _factorize(
        weight,
        input_importance,
        output_importance,
        rank,
        protocol,
        key,
    )
    patches = evaluate_residual_patches(
        weight,
        reconstruction,
        input_importance,
        output_importance,
        column_count,
        sparse_count,
    )
    result = {
        "block": block,
        "projection": projection,
        "tensor_name": tensor_name,
        "shape": [out_features, in_features],
        "source_elements": source_elements,
        "target_bits": target_bits,
        "rank": rank,
        "column_equivalent_count": column_count,
        "factor_bit_cost": asdict(factor_cost),
        "column_patch_bit_cost": asdict(column_cost),
        "sparse_patch_bit_cost": asdict(sparse_cost),
        "column_total_bits": factor_cost.total + column_cost.total,
        "sparse_total_bits": factor_cost.total + sparse_cost.total,
        "column_actual_bpw": (factor_cost.total + column_cost.total) / source_elements,
        "sparse_actual_bpw": (factor_cost.total + sparse_cost.total) / source_elements,
        "factorization": factorization,
        "patches": patches,
    }
    del weight, reconstruction, input_importance, output_importance
    if protocol.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _comparison_line(result: dict[str, Any]) -> str:
    patches = result["patches"]
    baseline = float(patches["unpatched"]["weighted_normalized_rmse"])
    column = float(patches["columns"]["metrics"]["weighted_normalized_rmse"])
    sparse = float(patches["sparse_entries"]["metrics"]["weighted_normalized_rmse"])
    return (
        f"{result['block']}:{result['projection']} columns={result['column_equivalent_count']} "
        f"rank={result['rank']} unpatched={baseline:.6f} column={column:.6f} "
        f"sparse={sparse:.6f} sparse-vs-column={(sparse / column - 1) * 100:+.2f}%"
    )


def run(args: argparse.Namespace) -> int:
    protocol = SparseResidualProtocol(
        1,
        args.model_revision,
        args.target_bpw,
        args.rank_alignment,
        args.scale_bits,
        16,
        args.index_bits,
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
    output = _load_output(args.output, protocol)
    profiles = load_calibration_profiles(args.calibration_state, args.calibration_shrinkage)
    requested = tuple(
        (block, projection, count)
        for block in args.blocks
        for projection in args.projections
        for count in args.column_counts
    )
    if not requested:
        raise ValueError("blocks, projections, and column counts must all be non-empty")
    lease_context = acquire_device_lease(args.device) if args.device.startswith("cuda") else nullcontext()
    with lease_context, safe_open(str(args.model), framework="pt", device="cpu") as handle:
        for block, projection, count in requested:
            key = f"{block}|{projection}|columns={count}"
            result = output["results"].get(key)
            if result is None:
                print(f"running {key}", flush=True)
                result = _execute_layer_budget(
                    handle,
                    block,
                    projection,
                    count,
                    protocol,
                    profiles,
                )
                output["results"][key] = result
                _write_output(args.output, output)
                print("completed " + _comparison_line(result), flush=True)
            else:
                print("reusing " + _comparison_line(result), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 24))
    parser.add_argument("--projections", type=_parse_projections, default=SUPPORTED_PROJECTIONS)
    parser.add_argument("--column-counts", type=_parse_ints, default=(0, 1, 2, 4, 8))
    parser.add_argument("--target-bpw", type=float, default=1.0)
    parser.add_argument("--rank-alignment", type=int, default=1)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--index-bits", type=int, default=32)
    parser.add_argument("--outer-iterations", type=int, default=400)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
