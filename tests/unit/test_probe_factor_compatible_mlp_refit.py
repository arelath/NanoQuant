from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


def _probe_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("probe_factor_compatible_mlp_refit")


def test_factor_probe_recovers_separable_dense_rescale() -> None:
    probe = _probe_module()
    source = torch.tensor(
        [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
    )
    expected_rows = torch.tensor([0.5, 2.0])
    expected_columns = torch.tensor([1.5, 0.75, 3.0])
    target = expected_rows.reshape(-1, 1) * source * expected_columns.reshape(1, -1)

    rows, columns = probe._axis_scales(
        source,
        target,
        fit_rows=True,
        fit_columns=True,
        iterations=12,
    )

    torch.testing.assert_close(
        rows.reshape(-1, 1) * source * columns.reshape(1, -1),
        target,
    )


def test_factor_probe_maps_policy_to_downstream_axes() -> None:
    probe = _probe_module()

    assert probe._fit_axes("mlp.gate_proj", "operator") == (True, False)
    assert probe._fit_axes("mlp.down_proj", "output") == (True, False)
    assert probe._fit_axes("mlp.down_proj", "input") == (False, True)
    assert probe._fit_axes("mlp.down_proj", "joint") == (True, True)
