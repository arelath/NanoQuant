from __future__ import annotations

import pytest
import torch

from tools.probe_importance_power import (
    _exponent_key,
    _parse_exponents,
    block_topology,
    temper_importance,
)


def test_temper_importance_preserves_mean_and_raw_endpoint() -> None:
    value = torch.tensor([0.0, 1.0, 4.0, 9.0])

    tempered = temper_importance(value, 0.5)
    raw = temper_importance(value, 1.0)

    assert tempered.mean() == pytest.approx(value.mean())
    assert tempered.tolist() == pytest.approx([0.0, 7.0 / 3.0, 14.0 / 3.0, 7.0], rel=1e-6)
    assert torch.equal(raw, value)
    assert raw.data_ptr() != value.data_ptr()


def test_temper_importance_zero_endpoint_is_uniform_at_original_mean() -> None:
    value = torch.tensor([1.0, 3.0])

    assert torch.equal(temper_importance(value, 0.0), torch.tensor([2.0, 2.0]))


def test_temper_importance_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="exponent"):
        temper_importance(torch.ones(2), 1.1)
    with pytest.raises(ValueError, match="non-negative"):
        temper_importance(torch.tensor([-1.0, 1.0]), 0.5)
    with pytest.raises(ValueError, match="finite"):
        temper_importance(torch.tensor([1.0, float("nan")]), 0.5)


def test_power_parser_key_and_topology_are_stable() -> None:
    assert _parse_exponents("0.5, 0.75, 1") == (0.5, 0.75, 1.0)
    assert _exponent_key(0.7500000000000001) == "0.75"
    topology = block_topology(12, 0.75)

    assert topology.comparison == "importance-power"
    assert topology.variant == "power-0.75"
    assert topology.location == "12"
    assert [group.label for group in topology.groups] == ["qkv", "o", "gate", "up", "down"]
    assert sorted(member.projection for group in topology.groups for member in group.members) == [
        "down",
        "gate",
        "k",
        "o",
        "q",
        "up",
        "v",
    ]
