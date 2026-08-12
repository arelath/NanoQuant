"""Screen late-selected structured INT8 sidecars with fixed-codebook scale refits.

Each arm starts from the same retained factor signs and exact bit budget.  A
round selects a patch from the current diagonal-Fisher-weighted residual and
then refits only the separable factor scales against the patch-subtracted
target.  Product-codebook selectors and binary factors remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import OwnerCase, _load_cases
from safetensors.torch import save_file

from nanoquant.config.codec import to_dict
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.domain.structured_sidecar import (
    StructuredSidecarCost,
    aligned_tile_int8_cost,
    select_int8_aligned_tile_patch,
    select_int8_column_patch,
    weighted_error,
    whole_column_int8_cost,
)
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json, hash_file


@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    rows: int | None = None
    columns: int | None = None

    @property
    def is_column(self) -> bool:
        return self.rows is None


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("blocks must be non-negative")
    return result


def _parse_shapes(value: str) -> tuple[Shape, ...]:
    result = []
    for raw in (item.strip().lower() for item in value.split(",")):
        if not raw:
            continue
        if raw == "column":
            result.append(Shape("column"))
            continue
        rows, separator, columns = raw.partition("x")
        try:
            shape = Shape(raw, int(rows), int(columns))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("shapes must be column or ROWSxCOLUMNS") from exc
        if not separator or min(shape.rows or 0, shape.columns or 0) <= 0:
            raise argparse.ArgumentTypeError("shapes must be column or ROWSxCOLUMNS")
        result.append(shape)
    if not result or len({shape.name for shape in result}) != len(result):
        raise argparse.ArgumentTypeError("shapes must be unique and non-empty")
    return tuple(result)


def _parse_overlay_shapes(value: str) -> tuple[Shape, ...]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    controls = [Shape("control")] if "control" in names else []
    shapes = [name for name in names if name != "control"]
    return tuple(controls) + (() if not shapes else _parse_shapes(",".join(shapes)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 25))
    parser.add_argument("--owner", default="mlp.down_proj")
    parser.add_argument("--expected-blocks", type=int, default=26)
    parser.add_argument(
        "--shapes",
        type=_parse_shapes,
        default=_parse_shapes("column,16x1,32x1,64x1,8x2,4x4,1x32"),
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--scale-passes", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--overlay-shapes", type=_parse_overlay_shapes)
    return parser


def _count_for_budget(
    shape: Shape,
    budget: int,
    out_features: int,
    in_features: int,
) -> tuple[int, StructuredSidecarCost]:
    if shape.is_column:
        cost = whole_column_int8_cost(out_features, 1, in_features)
        if cost.total != budget:
            raise AssertionError("column control does not define the requested budget")
        return 1, cost
    assert shape.rows is not None and shape.columns is not None
    count = 0
    while True:
        candidate = aligned_tile_int8_cost(
            shape.rows,
            shape.columns,
            count + 1,
            out_features,
            in_features,
        )
        if candidate.total > budget:
            break
        count += 1
    if count == 0:
        raise ValueError(f"one {shape.name} tile exceeds the column budget")
    return count, aligned_tile_int8_cost(
        shape.rows,
        shape.columns,
        count,
        out_features,
        in_features,
    )


def _select_patch(
    shape: Shape,
    residual: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shape.is_column:
        return select_int8_column_patch(residual, input_importance, output_importance, count)
    assert shape.rows is not None and shape.columns is not None
    return select_int8_aligned_tile_patch(
        residual,
        input_importance,
        output_importance,
        count,
        tile_rows=shape.rows,
        tile_columns=shape.columns,
    )


def _indices_hash(indices: torch.Tensor) -> str:
    return "sha256:" + hashlib.sha256(
        indices.detach().to(device="cpu", dtype=torch.int64).sort().values.numpy().tobytes()
    ).hexdigest()


def evaluate_case(
    case: OwnerCase,
    shapes: tuple[Shape, ...],
    *,
    rounds: int,
    scale_passes: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Return equal-bit shape results and final dense tensors for one owner."""

    if rounds <= 0 or scale_passes <= 0:
        raise ValueError("late-refit rounds and scale passes must be positive")
    target = case.target.to(device).float()
    input_importance = case.input_importance.to(device).float()
    output_importance = case.output_importance.to(device).float()
    left = case.left_binary.to(device)
    right = case.right_binary.to(device)
    initial_pre = case.scale_pre.to(device)
    initial_mid = case.scale_mid.to(device)
    initial_post = case.scale_post.to(device)
    protected = None if case.outlier_indices is None else case.outlier_indices.to(device)
    fixed_patch = (
        torch.zeros_like(target)
        if case.patch is None
        else case.patch.to(device).float()
    )
    factor_target = target - fixed_patch
    control = fit_scales(
        factor_target,
        left,
        right,
        initial_pre,
        initial_mid,
        initial_post,
        input_importance,
        output_importance,
        alternating_passes=scale_passes,
        protected_columns=protected,
    )
    control_factor = control.reconstruction.float()
    control_error = weighted_error(
        target - (control_factor + fixed_patch),
        input_importance,
        output_importance,
    )
    out_features, in_features = target.shape
    budget = whole_column_int8_cost(out_features, 1, in_features).total
    arms: dict[str, Any] = {}
    control_dense = control_factor + fixed_patch
    if case.outlier_indices is not None and case.outlier_values is not None:
        control_dense = control_dense.clone()
        control_dense[:, case.outlier_indices.to(device)] += case.outlier_values.to(device)
    dense: dict[str, torch.Tensor] = {
        "control": control_dense.to(torch.bfloat16).cpu().contiguous()
    }
    for shape in shapes:
        count, cost = _count_for_budget(shape, budget, out_features, in_features)
        scale_pre = control.scale_pre
        scale_mid = control.scale_mid
        scale_post = control.scale_post
        factor = control_factor.clone()
        patch = torch.zeros_like(target)
        round_records = []
        for round_index in range(rounds):
            residual = target - (factor + fixed_patch)
            patch, indices = _select_patch(
                shape,
                residual,
                input_importance,
                output_importance,
                count,
            )
            selected_error = weighted_error(
                target - (factor + fixed_patch + patch),
                input_importance,
                output_importance,
            )
            fitted = fit_scales(
                factor_target - patch,
                left,
                right,
                scale_pre,
                scale_mid,
                scale_post,
                input_importance,
                output_importance,
                alternating_passes=scale_passes,
                protected_columns=protected,
            )
            factor = fitted.reconstruction.float()
            scale_pre = fitted.scale_pre
            scale_mid = fitted.scale_mid
            scale_post = fitted.scale_post
            refit_error = weighted_error(
                target - (factor + fixed_patch + patch),
                input_importance,
                output_importance,
            )
            round_records.append(
                {
                    "round": round_index + 1,
                    "indices_sha256": _indices_hash(indices),
                    "selected_weighted_error": selected_error,
                    "refit_weighted_error": refit_error,
                    "gain_fraction_vs_control": 1.0 - refit_error / control_error,
                    "scale_refit_accepted": fitted.accepted,
                    "scale_refit_rollback_reason": fitted.rollback_reason,
                }
            )
        final = factor + fixed_patch + patch
        if case.outlier_indices is not None and case.outlier_values is not None:
            final = final.clone()
            final[:, case.outlier_indices.to(device)] += case.outlier_values.to(device)
        arms[shape.name] = {
            "count": count,
            "cost": {
                "value_bits": cost.value_bits,
                "scale_bits": cost.scale_bits,
                "index_bits": cost.index_bits,
                "total": cost.total,
            },
            "rounds": round_records,
            "final_weighted_error": round_records[-1]["refit_weighted_error"],
            "final_gain_fraction_vs_control": round_records[-1]["gain_fraction_vs_control"],
        }
        dense[shape.name] = final.to(torch.bfloat16).cpu().contiguous()
    return (
        {
            "block": case.block,
            "owner": case.name,
            "shape": list(target.shape),
            "column_budget_bits": budget,
            "control_scale_refit_accepted": control.accepted,
            "control_weighted_error": control_error,
            "arms": arms,
        },
        dense,
    )


def _write_overlay(path: Path, arm: str, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    with atomic_workspace(path) as temporary:
        tensor_path = temporary / "weights.safetensors"
        save_file(tensors, tensor_path)
        manifest = {
            "schema_version": 1,
            "arm": arm,
            "layer_count": len(tensors),
            "tensor_sha256": hash_file(tensor_path),
            "tensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
                for name, value in tensors.items()
            },
        }
        atomic_write_json(temporary / "manifest.json", manifest)
    return {"directory": str(path.resolve()), **manifest}


def _case_overlay_tensors(case: OwnerCase, value: torch.Tensor) -> dict[str, torch.Tensor]:
    if value.shape[0] != sum(case.member_rows):
        raise ValueError("late-refit overlay rows differ from the owner members")
    result = {}
    offset = 0
    for member, rows in zip(case.members, case.member_rows, strict=True):
        result[f"model.layers.{case.block}.{member.path}.weight"] = value[
            offset : offset + rows
        ].contiguous()
        offset += rows
    return result


def run(args: argparse.Namespace) -> int:
    identity, cases = _load_cases(
        args.run_output,
        args.model,
        args.blocks,
        (args.owner,),
        args.expected_blocks,
    )
    overlay_names = {
        shape.name for shape in (() if args.overlay_shapes is None else args.overlay_shapes)
    }
    unknown = overlay_names - ({shape.name for shape in args.shapes} | {"control"})
    if unknown:
        raise ValueError(f"overlay shapes were not screened: {sorted(unknown)}")
    overlays: dict[str, dict[str, torch.Tensor]] = {name: {} for name in overlay_names}
    results = []
    lease = (
        acquire_device_lease(args.device)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    with lease:
        for case in cases:
            record, tensors = evaluate_case(
                case,
                args.shapes,
                rounds=args.rounds,
                scale_passes=args.scale_passes,
                device=args.device,
            )
            results.append(record)
            for name in overlay_names:
                overlays[name].update(_case_overlay_tensors(case, tensors[name]))
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    overlay_manifests = {}
    if overlay_names:
        if args.overlay_root is None:
            raise ValueError("overlay-root is required when overlay-shapes are requested")
        for name, tensors in overlays.items():
            overlay_manifests[name] = _write_overlay(
                args.overlay_root / name.replace("x", "-by-"),
                f"late-selected-int8-{name}-fixed-codebook-scale-refit",
                tensors,
            )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "role": "analysis-only late structured sidecar fixed-codebook scale-refit screen",
        "source_run": str(args.run_output.resolve()),
        "identity": to_dict(identity),
        "protocol": {
            "blocks": list(args.blocks),
            "owner": args.owner,
            "shapes": [shape.name for shape in args.shapes],
            "rounds": args.rounds,
            "scale_passes_per_round": args.scale_passes,
            "selection_objective": "retained-diagonal-Fisher-weighted-post-fit-residual",
            "factor_policy": "fixed persisted signs and product-codebook selectors; refit scales only",
            "budget": "no more than one whole-column INT8 sidecar including BF16 scale and index",
        },
        "results": results,
        "overlays": overlay_manifests,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
