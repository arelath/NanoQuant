from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


def _probe_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("probe_foldable_mlp_initializer_transfer")


def test_seed_variants_filter_blocks_and_restore_original_fit_bounds() -> None:
    probe = _probe_module()
    tensors = {
        "model.layers.18.mlp.gate_proj.output_log_multiplier": torch.log(
            torch.tensor([0.01, 1.0, 10.0])
        ),
        "model.layers.18.mlp.up_proj.output_log_multiplier": torch.log(
            torch.tensor([0.05, 1.0, 100.0])
        ),
        "model.layers.23.mlp.down_proj.input_log_multiplier": torch.log(
            torch.tensor([0.1, 2.0, 8.0])
        ),
    }

    grouped, report = probe._variant_logs(
        tensors,
        probe.SeedArm("block18", (18,), True),
    )

    assert set(grouped) == {(18, "mlp.gate_proj"), (18, "mlp.up_proj")}
    torch.testing.assert_close(
        grouped[(18, "mlp.gate_proj")]["output"],
        torch.tensor([0.25, 1.0, 2.0]),
    )
    torch.testing.assert_close(
        grouped[(18, "mlp.up_proj")]["output"],
        torch.tensor([0.1, 1.0, 8.0]),
    )
    assert report["lower_winsorized_count"] == 2
    assert report["upper_winsorized_count"] == 2


def test_prefix_order_parser_requires_unique_non_negative_blocks() -> None:
    probe = _probe_module()

    assert probe._parse_blocks("25,24,23,17") == (25, 24, 23, 17)
