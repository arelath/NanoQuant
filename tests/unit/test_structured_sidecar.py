import torch

from nanoquant.domain.structured_sidecar import (
    aligned_tile_int8_cost,
    row_segment_int8_cost,
    select_int8_aligned_tile_patch,
    select_int8_column_patch,
    select_int8_row_segment_patch,
    weighted_error,
    whole_column_int8_cost,
)


def test_equal_budget_segments_recover_localized_residual() -> None:
    residual = torch.zeros(32, 4)
    residual[:4, 0] = 5
    residual[4:8, 1] = 4
    inputs = torch.ones(4)
    outputs = torch.ones(32)
    column_cost = whole_column_int8_cost(32, 1, 4)
    segment_cost = row_segment_int8_cost(4, 2, 8, 4)
    assert segment_cost.total <= column_cost.total

    column, _ = select_int8_column_patch(residual, inputs, outputs, 1)
    segment, _ = select_int8_row_segment_patch(residual, inputs, outputs, 2, rows=4)
    assert weighted_error(residual - segment, inputs, outputs) == 0
    assert weighted_error(residual - segment, inputs, outputs) < weighted_error(
        residual - column, inputs, outputs
    )


def test_segment_quantization_is_finite_and_bounded() -> None:
    residual = torch.randn(7, 5, generator=torch.Generator().manual_seed(5))
    patch, indices = select_int8_row_segment_patch(
        residual, torch.ones(5), torch.ones(7), 3, rows=4
    )
    assert indices.numel() == 3
    assert torch.isfinite(patch).all()
    assert torch.count_nonzero(patch) <= 12


def test_aligned_tile_recovers_local_residual_at_bounded_cost() -> None:
    residual = torch.zeros(16, 16)
    residual[:8, :8] = 3
    patch, indices = select_int8_aligned_tile_patch(
        residual,
        torch.ones(16),
        torch.ones(16),
        1,
        tile_rows=8,
        tile_columns=8,
    )
    assert aligned_tile_int8_cost(8, 8, 1, 16, 16).total == 530
    assert indices.numel() == 1
    assert weighted_error(residual - patch, torch.ones(16), torch.ones(16)) == 0
