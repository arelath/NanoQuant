from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from nanoquant.application.layers import FrozenReferenceLinear


def _probe_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("probe_tuned_mlp_scale_refit")


def _frozen(rows: int, columns: int) -> FrozenReferenceLinear:
    return FrozenReferenceLinear(
        torch.ones((rows, 1)),
        torch.ones((1, columns)),
        torch.ones(columns),
        torch.ones(1),
        torch.ones(rows),
        torch.zeros(rows),
    )


def test_tuned_refit_collects_only_selected_mlp_layers() -> None:
    probe = _probe_module()

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.gate_proj = _frozen(3, 2)
            self.mlp.up_proj = _frozen(3, 2)
            self.mlp.down_proj = _frozen(2, 3)

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList((Block(), Block()))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Base()

    result = probe._collect_selected_mlp_reconstructions(Model(), (1,))

    assert tuple(item.layer.block.index for item in result.layers) == (1, 1, 1)
    assert tuple(item.layer.path for item in result.layers) == (
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    assert tuple(tuple(item.weight.shape) for item in result.layers) == (
        (3, 2),
        (3, 2),
        (2, 3),
    )


def test_tuned_refit_arm_parser_requires_baseline() -> None:
    probe = _probe_module()

    assert probe._parse_arms("baseline,policy") == ("baseline", "policy")
    with pytest.raises(
        Exception,
        match="including baseline",
    ):
        probe._parse_arms("policy")
