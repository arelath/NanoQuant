"""Compare whole-column and row-segment INT8 sidecars at equal exact bits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import _load_cases
from safetensors.torch import save_file

from nanoquant.config.codec import to_dict
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.domain.structured_sidecar import (
    aligned_tile_int8_cost,
    row_segment_int8_cost,
    select_int8_aligned_tile_patch,
    select_int8_column_patch,
    select_int8_row_segment_patch,
    weighted_error,
    whole_column_int8_cost,
)
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json, hash_file


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("blocks must be non-negative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--column-overlay", type=Path, required=True)
    parser.add_argument("--segment-overlay", type=Path, required=True)
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 25))
    parser.add_argument("--owner", default="mlp.down_proj")
    parser.add_argument("--expected-blocks", type=int, default=26)
    parser.add_argument("--column-count", type=int, default=1)
    parser.add_argument("--segment-rows", type=int, default=32)
    parser.add_argument("--tile-shape", type=_tile_shape, action="append", default=[])
    return parser


def _tile_shape(value: str) -> tuple[int, int]:
    rows, separator, columns = value.partition("x")
    try:
        result = int(rows), int(columns)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tile shape must use ROWSxCOLUMNS") from exc
    if not separator or min(result) <= 0:
        raise argparse.ArgumentTypeError("tile shape must use ROWSxCOLUMNS")
    return result


def _write_overlay(path: Path, arm: str, tensors: dict[str, torch.Tensor]) -> dict[str, object]:
    with atomic_workspace(path) as temporary:
        tensor_path = temporary / "weights.safetensors"
        save_file(tensors, tensor_path)
        manifest = {
            "schema_version": 1,
            "arm": arm,
            "layer_count": len(tensors),
            "blocks": sorted(int(name.split(".")[2]) for name in tensors),
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


def run(args: argparse.Namespace) -> int:
    identity, cases = _load_cases(
        args.run_output, args.model, args.blocks, (args.owner,), args.expected_blocks
    )
    column_tensors = {}
    segment_tensors = {}
    tile_tensors: dict[str, dict[str, torch.Tensor]] = {
        f"{rows}x{columns}": {} for rows, columns in args.tile_shape
    }
    results = []
    for case in cases:
        baseline_factor = reconstruct(
            case.left_binary,
            case.right_binary,
            case.scale_pre,
            case.scale_mid,
            case.scale_post,
        ).float()
        residual = case.target.float() - baseline_factor
        column_cost = whole_column_int8_cost(
            residual.shape[0], args.column_count, residual.shape[1]
        )
        segment_count = 0
        while row_segment_int8_cost(
            args.segment_rows,
            segment_count + 1,
            residual.shape[0],
            residual.shape[1],
        ).total <= column_cost.total:
            segment_count += 1
        column_patch, column_indices = select_int8_column_patch(
            residual,
            case.input_importance,
            case.output_importance,
            args.column_count,
        )
        segment_patch, segment_indices = select_int8_row_segment_patch(
            residual,
            case.input_importance,
            case.output_importance,
            segment_count,
            rows=args.segment_rows,
        )
        segment_cost = row_segment_int8_cost(
            args.segment_rows,
            segment_count,
            residual.shape[0],
            residual.shape[1],
        )
        base_error = weighted_error(residual, case.input_importance, case.output_importance)
        column_error = weighted_error(
            residual - column_patch, case.input_importance, case.output_importance
        )
        tile_records = {}
        tile_patches = {}
        for tile_rows, tile_columns in args.tile_shape:
            tile_count = 0
            while aligned_tile_int8_cost(
                tile_rows,
                tile_columns,
                tile_count + 1,
                residual.shape[0],
                residual.shape[1],
            ).total <= column_cost.total:
                tile_count += 1
            tile_patch, tile_indices = select_int8_aligned_tile_patch(
                residual,
                case.input_importance,
                case.output_importance,
                tile_count,
                tile_rows=tile_rows,
                tile_columns=tile_columns,
            )
            tile_cost = aligned_tile_int8_cost(
                tile_rows,
                tile_columns,
                tile_count,
                residual.shape[0],
                residual.shape[1],
            )
            tile_error = weighted_error(
                residual - tile_patch,
                case.input_importance,
                case.output_importance,
            )
            key = f"{tile_rows}x{tile_columns}"
            tile_patches[key] = tile_patch
            tile_records[key] = {
                "count": tile_count,
                "weighted_error": tile_error,
                "gain_fraction": 1 - tile_error / base_error,
                "vs_column_error_fraction": tile_error / column_error - 1,
                "cost": {
                    "value_bits": tile_cost.value_bits,
                    "scale_bits": tile_cost.scale_bits,
                    "index_bits": tile_cost.index_bits,
                    "total": tile_cost.total,
                },
                "indices_sha256": "sha256:"
                + hashlib.sha256(tile_indices.cpu().numpy().tobytes()).hexdigest(),
            }
        segment_error = weighted_error(
            residual - segment_patch, case.input_importance, case.output_importance
        )
        base_dense = baseline_factor
        if case.outlier_indices is not None and case.outlier_values is not None:
            base_dense = base_dense.clone()
            base_dense[:, case.outlier_indices] += case.outlier_values
        if case.patch is not None:
            base_dense += case.patch
        name = f"model.layers.{case.block}.{args.owner}.weight"
        column_tensors[name] = (base_dense + column_patch).to(torch.bfloat16).contiguous()
        segment_tensors[name] = (base_dense + segment_patch).to(torch.bfloat16).contiguous()
        for key, patch in tile_patches.items():
            tile_tensors[key][name] = (base_dense + patch).to(torch.bfloat16).contiguous()
        results.append(
            {
                "block": case.block,
                "shape": list(residual.shape),
                "baseline_weighted_error": base_error,
                "column_weighted_error": column_error,
                "segment_weighted_error": segment_error,
                "column_gain_fraction": 1 - column_error / base_error,
                "segment_gain_fraction": 1 - segment_error / base_error,
                "segment_vs_column_error_fraction": segment_error / column_error - 1,
                "column_count": args.column_count,
                "segment_count": segment_count,
                "segment_rows": args.segment_rows,
                "column_cost": {
                    "value_bits": column_cost.value_bits,
                    "scale_bits": column_cost.scale_bits,
                    "index_bits": column_cost.index_bits,
                    "total": column_cost.total,
                },
                "segment_cost": {
                    "value_bits": segment_cost.value_bits,
                    "scale_bits": segment_cost.scale_bits,
                    "index_bits": segment_cost.index_bits,
                    "total": segment_cost.total,
                },
                "column_indices": column_indices.tolist(),
                "segment_indices_sha256": "sha256:"
                + hashlib.sha256(segment_indices.cpu().numpy().tobytes()).hexdigest(),
                "tiles": tile_records,
            }
        )
    column_manifest = _write_overlay(args.column_overlay, "whole-column-int8", column_tensors)
    segment_manifest = _write_overlay(args.segment_overlay, "row-segment-int8", segment_tensors)
    tile_manifests = {
        key: _write_overlay(
            args.segment_overlay.with_name(args.segment_overlay.name + f"-tile-{key}"),
            f"aligned-tile-int8-{key}",
            tensors,
        )
        for key, tensors in tile_tensors.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "role": "analysis-only equal-bit structured sidecar screen",
        "source_run": str(args.run_output.resolve()),
        "identity": to_dict(identity),
        "protocol": {
            "blocks": list(args.blocks),
            "owner": args.owner,
            "column_count": args.column_count,
            "segment_rows": args.segment_rows,
            "tile_shapes": [list(value) for value in args.tile_shape],
            "selection_objective": "retained-diagonal-Fisher-weighted-residual",
            "storage": "symmetric-int8-per-column-or-segment-with-bf16-scale",
        },
        "results": results,
        "column_overlay": column_manifest,
        "segment_overlay": segment_manifest,
        "tile_overlays": tile_manifests,
    }
    payload["all_segments_dominate_columns"] = all(
        float(item["segment_weighted_error"]) < float(item["column_weighted_error"])
        for item in results
    )
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
