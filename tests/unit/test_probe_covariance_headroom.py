from __future__ import annotations

import pytest
import torch

from tools.probe_covariance_headroom import (
    _module_input_width,
    compare_covariance_floors,
    covariance_groups,
    covariance_rank_floor,
    diagonal_rank_floor,
)


def test_covariance_groups_cover_supported_complete_block_units() -> None:
    groups = covariance_groups(12)

    assert [group.label for group in groups] == ["qkv", "o", "gate", "up"]
    assert [member.projection for member in groups[0].members] == ["q", "k", "v"]
    assert all(member.block == 12 for group in groups for member in group.members)


def test_module_input_width_uses_each_projection_weight_shape() -> None:
    assert _module_input_width(torch.nn.Linear(5, 3, bias=False)) == 5


def test_diagonal_and_covariance_rank_floors_match_for_diagonal_covariance() -> None:
    target = torch.tensor([[2.0, 1.0], [0.0, 1.0]])
    diagonal = torch.tensor([1.0, 3.0])
    covariance = torch.diag(diagonal)
    output = torch.tensor([1.0, 2.0])

    diagonal_fit = diagonal_rank_floor(target, diagonal, output, 1)
    covariance_fit = covariance_rank_floor(target, covariance, output, 1)

    assert torch.allclose(diagonal_fit, covariance_fit, rtol=1e-5, atol=1e-5)


def test_correlated_covariance_exposes_held_out_same_rank_headroom() -> None:
    target = torch.eye(2)
    covariance = torch.tensor([[1.0, 0.9], [0.9, 1.0]])

    result = compare_covariance_floors(
        target,
        covariance,
        covariance,
        torch.ones(2),
        1,
        damp_fraction=0.0,
        promotion_threshold=0.2,
    )

    held_out = result["held_out"]
    assert held_out["covariance_floor_error"] < held_out["diagonal_floor_error"]
    assert held_out["covariance_relative_error_reduction"] == pytest.approx(0.9, rel=1e-4)
    assert result["promotes_covariance"] is True


def test_floor_helpers_reject_dimension_mismatches() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        diagonal_rank_floor(torch.ones((2, 2)), torch.ones(3), torch.ones(2), 1)
    with pytest.raises(ValueError, match="dimensions"):
        covariance_rank_floor(torch.ones((2, 2)), torch.eye(3), torch.ones(2), 1)
