"""Exact-cost, analysis-side structured residual sidecars."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class StructuredSidecarCost:
    value_bits: int
    scale_bits: int
    index_bits: int

    @property
    def total(self) -> int:
        return self.value_bits + self.scale_bits + self.index_bits


def whole_column_int8_cost(out_features: int, count: int, in_features: int) -> StructuredSidecarCost:
    if min(out_features, count, in_features) <= 0:
        raise ValueError("whole-column sidecar dimensions must be positive")
    return StructuredSidecarCost(
        out_features * count * 8,
        count * 16,
        count * max(1, math.ceil(math.log2(in_features))),
    )


def row_segment_int8_cost(
    rows: int,
    count: int,
    out_features: int,
    in_features: int,
) -> StructuredSidecarCost:
    if min(rows, count, out_features, in_features) <= 0 or rows > out_features:
        raise ValueError("row-segment sidecar dimensions are invalid")
    segments = math.ceil(out_features / rows)
    return StructuredSidecarCost(
        rows * count * 8,
        count * 16,
        count * max(1, math.ceil(math.log2(in_features * segments))),
    )


def aligned_tile_int8_cost(
    tile_rows: int,
    tile_columns: int,
    count: int,
    out_features: int,
    in_features: int,
) -> StructuredSidecarCost:
    if min(tile_rows, tile_columns, count, out_features, in_features) <= 0:
        raise ValueError("aligned-tile sidecar dimensions must be positive")
    tile_inventory = math.ceil(out_features / tile_rows) * math.ceil(in_features / tile_columns)
    return StructuredSidecarCost(
        tile_rows * tile_columns * count * 8,
        count * 16,
        count * max(1, math.ceil(math.log2(tile_inventory))),
    )


def select_int8_column_patch(
    residual: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an oracle diagonal-Fisher INT8 column patch and flat indices."""

    if residual.ndim != 2 or count <= 0 or count > residual.shape[1]:
        raise ValueError("INT8 column selection inputs are invalid")
    scores = residual.float().square() * output_importance.float()[:, None]
    scores *= input_importance.float()[None, :]
    columns = torch.topk(scores.sum(dim=0), count, sorted=False).indices
    values = residual[:, columns].float()
    scales = values.abs().amax(dim=0).clamp_min(1e-12) / 127
    stored = torch.round(values / scales).clamp(-127, 127) * scales
    patch = torch.zeros_like(residual, dtype=torch.float32)
    patch[:, columns] = stored
    return patch, columns


def select_int8_row_segment_patch(
    residual: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    count: int,
    *,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return oracle non-overlapping column/row-segment INT8 residual tiles."""

    if residual.ndim != 2 or count <= 0 or rows <= 0 or rows > residual.shape[0]:
        raise ValueError("INT8 row-segment selection inputs are invalid")
    out_features, in_features = residual.shape
    segments = math.ceil(out_features / rows)
    weighted = residual.float().square() * output_importance.float()[:, None]
    weighted *= input_importance.float()[None, :]
    padded_rows = segments * rows
    if padded_rows != out_features:
        weighted = torch.nn.functional.pad(weighted, (0, 0, 0, padded_rows - out_features))
    scores = weighted.reshape(segments, rows, in_features).sum(dim=1).reshape(-1)
    if count > scores.numel():
        raise ValueError("row-segment count exceeds available tiles")
    chosen = torch.topk(scores, count, sorted=False).indices
    patch = torch.zeros_like(residual, dtype=torch.float32)
    for flat in chosen.tolist():
        segment, column = divmod(flat, in_features)
        start = segment * rows
        end = min(start + rows, out_features)
        values = residual[start:end, column].float()
        scale = values.abs().amax().clamp_min(1e-12) / 127
        patch[start:end, column] = torch.round(values / scale).clamp(-127, 127) * scale
    return patch, chosen


def select_int8_aligned_tile_patch(
    residual: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    count: int,
    *,
    tile_rows: int,
    tile_columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return oracle non-overlapping aligned INT8 residual tiles."""

    if residual.ndim != 2 or min(count, tile_rows, tile_columns) <= 0:
        raise ValueError("INT8 aligned-tile selection inputs are invalid")
    out_features, in_features = residual.shape
    row_tiles = math.ceil(out_features / tile_rows)
    column_tiles = math.ceil(in_features / tile_columns)
    padded_rows = row_tiles * tile_rows
    padded_columns = column_tiles * tile_columns
    weighted = residual.float().square() * output_importance.float()[:, None]
    weighted *= input_importance.float()[None, :]
    weighted = torch.nn.functional.pad(
        weighted,
        (0, padded_columns - in_features, 0, padded_rows - out_features),
    )
    scores = weighted.reshape(row_tiles, tile_rows, column_tiles, tile_columns)
    scores = scores.sum(dim=(1, 3)).reshape(-1)
    if count > scores.numel():
        raise ValueError("aligned-tile count exceeds available tiles")
    chosen = torch.topk(scores, count, sorted=False).indices
    patch = torch.zeros_like(residual, dtype=torch.float32)
    for flat in chosen.tolist():
        row_tile, column_tile = divmod(flat, column_tiles)
        row_start = row_tile * tile_rows
        row_end = min(row_start + tile_rows, out_features)
        column_start = column_tile * tile_columns
        column_end = min(column_start + tile_columns, in_features)
        values = residual[row_start:row_end, column_start:column_end].float()
        scale = values.abs().amax().clamp_min(1e-12) / 127
        patch[row_start:row_end, column_start:column_end] = (
            torch.round(values / scale).clamp(-127, 127) * scale
        )
    return patch, chosen


def weighted_error(
    difference: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> float:
    return float(
        (
            difference.float().square()
            * output_importance.float()[:, None]
            * input_importance.float()[None, :]
        ).sum()
    )


__all__ = [
    "StructuredSidecarCost",
    "aligned_tile_int8_cost",
    "row_segment_int8_cost",
    "select_int8_column_patch",
    "select_int8_aligned_tile_patch",
    "select_int8_row_segment_patch",
    "weighted_error",
    "whole_column_int8_cost",
]
