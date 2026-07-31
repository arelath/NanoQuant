from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

from nanoquant.domain.models import BlockId, LayerId


def _probe_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("probe_composed_context_coordinate_sweep")


def test_coordinate_policy_parser_includes_context() -> None:
    probe = _probe_module()

    assert probe._parse_policy(
        "0:output:student_function,24:joint:teacher_function"
    ) == (
        (0, "output", "student_function"),
        (24, "joint", "teacher_function"),
    )
    with pytest.raises(Exception, match="unique"):
        probe._parse_policy(
            "0:output:student_function,0:joint:teacher_function"
        )


def test_coordinate_overlay_names_map_to_layer_ids() -> None:
    probe = _probe_module()
    weight = torch.ones((3, 2))

    result = probe._overlay_replacements(
        {"model.layers.7.mlp.gate_proj.weight": weight}
    )

    assert set(result) == {LayerId(BlockId(7), "mlp.gate_proj")}
    assert result[LayerId(BlockId(7), "mlp.gate_proj")] is weight


def test_coordinate_overlay_rejects_attention_tensor() -> None:
    probe = _probe_module()

    with pytest.raises(ValueError, match="non-MLP"):
        probe._overlay_replacements(
            {"model.layers.7.self_attn.q_proj.weight": torch.ones((3, 2))}
        )

