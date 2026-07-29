from __future__ import annotations

import pytest
import torch

from tools.probe_input_hadamard import (
    _parse_seeds,
    block_groups,
    make_structured_hadamard,
    rotated_covariance_diagonal,
)


def test_structured_hadamard_is_orthogonal_and_round_trips() -> None:
    transform = make_structured_hadamard(8, 4, 7)
    value = torch.arange(24, dtype=torch.float32).reshape(3, 8)

    rotated = transform.apply_right(value)
    restored = transform.inverse_right(rotated)
    matrix = transform.matrix()

    assert torch.allclose(restored, value, rtol=1e-6, atol=1e-6)
    assert torch.allclose(matrix.mT @ matrix, torch.eye(8), rtol=1e-6, atol=1e-6)
    assert torch.allclose(rotated.square().sum(), value.square().sum(), rtol=1e-6)


def test_rotated_covariance_diagonal_matches_explicit_transform_and_preserves_trace() -> None:
    transform = make_structured_hadamard(4, 4, 11)
    covariance = torch.tensor(
        [
            [2.0, 0.4, 0.2, 0.0],
            [0.4, 1.0, 0.1, 0.0],
            [0.2, 0.1, 3.0, 0.5],
            [0.0, 0.0, 0.5, 4.0],
        ]
    )
    matrix = transform.matrix()
    expected = torch.diagonal(matrix.mT @ covariance @ matrix)
    actual = rotated_covariance_diagonal(covariance, transform)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.sum() == pytest.approx(float(torch.trace(covariance)), rel=1e-6)


def test_hadamard_validation_and_seed_parser_are_stable() -> None:
    assert _parse_seeds("0, 2,7") == (0, 2, 7)
    with pytest.raises(ValueError, match="power of two"):
        make_structured_hadamard(12, 6, 0)
    with pytest.raises(ValueError, match="dividing"):
        make_structured_hadamard(10, 4, 0)
    with pytest.raises(ValueError, match="non-negative"):
        make_structured_hadamard(8, 4, -1)


def test_hadamard_block_inventory_is_complete_and_shares_mlp_input_role() -> None:
    groups = block_groups(12)

    assert [group.label for group in groups] == ["qkv", "o", "gate", "up", "down"]
    assert sorted(member.projection for group in groups for member in group.members) == [
        "down",
        "gate",
        "k",
        "o",
        "q",
        "up",
        "v",
    ]
    assert all(member.block == 12 for group in groups for member in group.members)
