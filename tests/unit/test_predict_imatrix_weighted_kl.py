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
    return importlib.import_module("predict_imatrix_weighted_kl")


def test_noise_equivalent_error_weights_each_columns_residual_by_mean_one_importance() -> None:
    tool = _tool()
    original = np.ones((2, 4), dtype=np.float32)
    quantized = original.copy()
    quantized[:, 0] += 0.2
    even = np.full(4, 3.0, dtype=np.float32)
    raw, weighted = tool.noise_equivalent_error(original, quantized, even)
    assert raw == pytest.approx(0.1, rel=1e-5) and weighted == pytest.approx(raw)
    quiet = np.array([0.0, 4.0, 4.0, 4.0], dtype=np.float32)
    assert tool.noise_equivalent_error(original, quantized, quiet)[1] == 0
    loud = np.array([8.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert tool.noise_equivalent_error(original, quantized, loud)[1] == pytest.approx(2 * raw)
    with pytest.raises(ValueError, match="input columns"):
        tool.noise_equivalent_error(original, quantized, np.ones(3, dtype=np.float32))


def test_summary_compares_both_predictions_with_isolated_and_joint_kl() -> None:
    tool = _tool()
    rows = {
        "a": {"unweighted_prediction": 0.4, "weighted_prediction": 0.1, "measured_isolated_kl": 0.1},
        "b": {"unweighted_prediction": 0.2, "weighted_prediction": 0.3, "measured_isolated_kl": 0.3},
    }
    summary = tool.summarize(rows, joint_kl=0.2)
    assert summary["sum_isolated_kl"] == pytest.approx(0.4)
    assert summary["unweighted"]["over_isolated"] == pytest.approx(1.5)
    assert summary["weighted"]["over_isolated"] == pytest.approx(1.0)
    assert summary["weighted"]["over_joint"] == pytest.approx(2.0)
    assert summary["unweighted"]["spearman"] == pytest.approx(-1)
    assert summary["weighted"]["spearman"] == pytest.approx(1)
    assert summary["weighted"]["median_abs_log_ratio"] == pytest.approx(0)
    markdown = tool.render_markdown({
        "protocol": {"teacher": {"source": "test/model"}}, "computed_at": "now", "isolated_report": "iso.json",
        "imatrix": {"path": "im.gguf"}, "files": [{"file": "m.gguf", **summary}], "json": "out.json",
    })
    assert "| m.gguf | 0.6000 | 0.4000 | 0.4000 | 0.2000 | 1.500 | 1.000 | 2.000 | -1.00 | 1.00 |" in markdown
