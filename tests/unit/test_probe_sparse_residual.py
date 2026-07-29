from __future__ import annotations

import torch

from nanoquant.domain.planning import factor_bit_cost
from tools.probe_sparse_residual import (
    column_patch_bit_cost,
    evaluate_residual_patches,
    matched_sparse_entry_count,
    rank_for_total_budget,
    sparse_entry_bit_cost,
)


def test_sparse_entries_are_matched_to_the_column_patch_budget() -> None:
    column_cost = column_patch_bit_cost(1024, 3)
    count = matched_sparse_entry_count(column_cost.total)
    sparse_cost = sparse_entry_bit_cost(count)

    assert column_cost.total == 3 * (1024 * 16 + 32)
    assert sparse_cost.total <= column_cost.total
    assert column_cost.total - sparse_cost.total < 48


def test_patch_budget_reduces_rank_to_keep_total_bits_bounded() -> None:
    target_bits = 1024 * 1152
    patch_bits = column_patch_bit_cost(1024, 4).total
    rank = rank_for_total_budget(
        1024,
        1152,
        target_bits,
        patch_bits,
        scale_bits=16,
        rank_alignment=1,
    )

    assert factor_bit_cost(1024, 1152, rank).total + patch_bits <= target_bits
    assert factor_bit_cost(1024, 1152, rank + 1).total + patch_bits > target_bits


def test_sparse_patch_wins_when_weighted_error_is_dispersed() -> None:
    weight = torch.tensor(
        [
            [4.0, 0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0, 3.0],
        ]
    )
    result = evaluate_residual_patches(
        weight,
        torch.zeros_like(weight),
        torch.ones(4),
        torch.ones(2),
        column_count=1,
        sparse_count=2,
    )

    column_error = result["columns"]["metrics"]["weighted_error_energy"]
    sparse_error = result["sparse_entries"]["metrics"]["weighted_error_energy"]
    assert sparse_error < column_error
    assert result["sparse_entries"]["unique_columns"] == 2


def test_column_patch_wins_when_weighted_error_is_column_concentrated() -> None:
    weight = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    result = evaluate_residual_patches(
        weight,
        torch.zeros_like(weight),
        torch.ones(3),
        torch.ones(3),
        column_count=1,
        sparse_count=2,
    )

    column_error = result["columns"]["metrics"]["weighted_error_energy"]
    sparse_error = result["sparse_entries"]["metrics"]["weighted_error_energy"]
    assert column_error == 0.0
    assert sparse_error > column_error


def test_patch_selection_respects_diagonal_fisher_importance() -> None:
    weight = torch.tensor([[5.0, 1.0]])
    result = evaluate_residual_patches(
        weight,
        torch.zeros_like(weight),
        torch.tensor([0.01, 100.0]),
        torch.ones(1),
        column_count=1,
        sparse_count=1,
    )

    assert result["columns"]["metrics"]["weighted_error_energy"] == 0.25
    assert result["sparse_entries"]["metrics"]["weighted_error_energy"] == 0.25
