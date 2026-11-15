from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    tools = str(REPOSITORY / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("plot_kl_article_figures")


def test_error_grid_orders_projections_by_row_and_blocks_by_column() -> None:
    tool = _tool()
    grid = tool.error_grid({"1:mlp.down_proj": 0.3, "0:self_attn.q_proj": 0.1, "0:mlp.down_proj": 0.2})
    assert grid.shape == (7, 2)
    assert grid[0, 0] == 0.1 and grid[6, 0] == 0.2 and grid[6, 1] == 0.3
    assert np.isnan(grid[0, 1])


def test_file_names_shorten_to_the_quantization_label() -> None:
    tool = _tool()
    assert tool.short_name("gemma-3-270m-it-Q4_K_M.gguf") == "Q4_K_M"
    assert tool.display_name("gemma-3-270m-it-nanoquant-pre-kd.gguf") == "Binary factors, before distillation"


def test_article_figures_render_from_joint_and_gguf_reports(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    tool = _tool()
    tool.require_plotting()
    errors = {f"{block}:{projection}": 0.01 * (block + 1) for block in range(2) for projection in tool.ROWS}
    variant = {"kl_above_control": 0.2}
    files = [
        {"file": f"m-it-{name}.gguf", "projection_bits_per_weight": bits, "projection_errors": errors,
         "additive_prediction": 0.4, "variants": {"projections": variant}}
        for name, bits in (("Q4_K_M", 5.4), ("nanoquant", 2.8), ("nanoquant-pre-kd", 2.8))
    ]
    joint = {"results": [
        {"kl_target": 0, "runs": []},
        {"kl_target": 0.001, "projection_count": 14, "mean_additive_prediction": 0.014,
         "mean_kl_above_control": 0.0137, "mean_measured_over_additive": 0.98,
         "ratio_confidence_interval_95": [0.96, 1.0],
         "runs": [{"measured_over_additive": 0.97}, {"measured_over_additive": 0.99}]},
    ]}
    paths = [
        tool.plot_additivity(joint, tmp_path), tool.plot_error_maps({"files": files}, tmp_path),
        tool.plot_predicted_vs_measured({"files": files}, joint, tmp_path),
    ]
    assert all(path.stat().st_size > 0 for path in paths)
