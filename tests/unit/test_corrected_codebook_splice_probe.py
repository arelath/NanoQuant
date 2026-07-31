from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nanoquant.domain.models import BlockId, LayerId
from nanoquant.infrastructure.kl_splice import (
    SpliceReconstruction,
    SpliceReconstructionSet,
)


def _probe_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("probe_corrected_codebook_splice")


def test_splice_probe_selects_blocks_without_copying_other_layers() -> None:
    probe = _probe_module()
    first = LayerId(BlockId(0), "mlp.down_proj")
    second = LayerId(BlockId(2), "mlp.down_proj")
    reconstructions = SpliceReconstructionSet(
        (
            SpliceReconstruction(first, torch.ones((1, 1)), None, 1.0),
            SpliceReconstruction(second, torch.ones((1, 1)), None, 2.0),
        ),
        (
            ("0:mlp.down_proj", (first,)),
            ("2:mlp.down_proj", (second,)),
        ),
        (
            ("0:mlp.down_proj", 1.0),
            ("2:mlp.down_proj", 2.0),
        ),
    )

    selected = probe._select_blocks(reconstructions, (2,))

    assert tuple(item.layer for item in selected.layers) == (second,)
    assert selected.unit_members == (("2:mlp.down_proj", (second,)),)


def test_splice_probe_selects_a_disjoint_token_window() -> None:
    probe = _probe_module()
    tokens = torch.arange(30).reshape(10, 3)

    selected = probe._select_token_window(tokens, offset=4, samples=3)

    assert torch.equal(selected, tokens[4:7])
    with pytest.raises(ValueError, match="shorter"):
        probe._select_token_window(tokens, offset=9, samples=2)


def test_splice_probe_accepts_multiple_projection_layers_per_block() -> None:
    probe = _probe_module()
    gate = LayerId(BlockId(2), "mlp.gate_proj")
    up = LayerId(BlockId(2), "mlp.up_proj")
    reconstructions = SpliceReconstructionSet(
        (
            SpliceReconstruction(gate, torch.ones((1, 1)), None, 1.0),
            SpliceReconstruction(up, torch.ones((1, 1)), None, 2.0),
        ),
        (
            ("2:mlp.gate_proj", (gate,)),
            ("2:mlp.up_proj", (up,)),
        ),
        (
            ("2:mlp.gate_proj", 1.0),
            ("2:mlp.up_proj", 2.0),
        ),
    )

    selected = probe._select_blocks(reconstructions, (2,))

    assert tuple(item.layer for item in selected.layers) == (gate, up)
    assert probe._parse_projections("gate,up") == ("gate", "up")


def test_splice_probe_composes_gated_down_outputs() -> None:
    probe = _probe_module()
    inputs = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    gate_weight = torch.tensor([[0.5, 1.0], [-1.0, 0.25]])
    up_weight = torch.tensor([[1.0, -0.25], [0.5, 0.75]])
    down_weight = torch.tensor([[0.75, -0.5], [0.25, 1.0]])

    observed = probe._gated_down_outputs(
        inputs,
        gate_weight,
        up_weight,
        down_weight,
        device="cpu",
    )
    gated = F.silu(F.linear(inputs, gate_weight)) * F.linear(
        inputs,
        up_weight,
    )
    expected = F.linear(gated, down_weight)

    assert torch.allclose(observed.float(), expected, atol=2e-2, rtol=2e-2)


def test_splice_probe_composes_per_block_downstream_policy() -> None:
    probe = _probe_module()
    first = LayerId(BlockId(0), "mlp.down_proj")
    second = LayerId(BlockId(12), "mlp.down_proj")

    def arm(first_value: float, second_value: float) -> SpliceReconstructionSet:
        return SpliceReconstructionSet(
            (
                SpliceReconstruction(
                    first,
                    torch.tensor([[first_value]]),
                    None,
                    1.0,
                ),
                SpliceReconstruction(
                    second,
                    torch.tensor([[second_value]]),
                    None,
                    1.0,
                ),
            ),
            (("0", (first,)), ("12", (second,))),
            (("0", 1.0), ("12", 1.0)),
        )

    sets = {}
    for prefix in ("free_words", "corrected_codebook"):
        sets[f"{prefix}_operator_refit"] = arm(1.0, 1.0)
        sets[f"{prefix}_operator_downstream_input_refit"] = arm(2.0, 2.0)
        sets[f"{prefix}_operator_downstream_joint_refit"] = arm(3.0, 3.0)

    result = probe._downstream_policy_sets(
        sets,
        probe._parse_block_policy("0:joint,12:input"),
    )

    policy = result["corrected_codebook_operator_policy_refit"]
    assert [float(item.weight.item()) for item in policy.layers] == [3.0, 2.0]
