from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TARGETS = [f"{block}:{name}" for block in range(2) for name in ("self_attn.q_proj", "mlp.down_proj")]


def _tool() -> ModuleType:
    tools = str(REPOSITORY / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("measure_gguf_isolated_kl")


def _isolated(tool: ModuleType) -> dict:
    curves = {name: {"coefficient": 2.0, "exponent": 2.0, "measured_error_range": [0.01, 0.35]} for name in TARGETS}
    errors = dict(zip(TARGETS, (0.1, 0.2, 0.005, 0.3), strict=True))

    def kl_of(names) -> float:
        # Real error costs half the Gaussian prediction alone, and 20% less again when combined.
        alone = sum(0.5 * 2.0 * errors[name] ** 2 for name in names)
        return alone if len(names) == 1 else 0.8 * alone

    return tool.isolate(TARGETS, errors, curves, kl_of, log=lambda _text: None)


def test_groupings_cover_everything_each_block_and_each_type() -> None:
    groups = _tool().groupings(TARGETS)
    assert groups["all"] == {"all": TARGETS}
    assert groups["block"] == {"0": TARGETS[:2], "1": TARGETS[2:]}
    assert groups["type"] == {"self_attn.q_proj": TARGETS[0::2], "mlp.down_proj": TARGETS[1::2]}


def test_isolate_splits_measured_over_predicted_into_shape_and_interaction() -> None:
    result = _isolated(_tool())
    first = result["projections"]["0:self_attn.q_proj"]
    assert first["gaussian_prediction"] == pytest.approx(0.02) and first["kl_above_control"] == pytest.approx(0.01)
    assert first["within_measured_range"] and not result["projections"]["1:self_attn.q_proj"]["within_measured_range"]
    everything = result["groups"]["all"]["all"]
    assert everything["members"] == 4
    assert everything["shape_factor"] == pytest.approx(0.5)
    assert everything["interaction_factor"] == pytest.approx(0.8)
    assert everything["measured_over_predicted"] == pytest.approx(0.4)
    assert everything["joint_kl"] == pytest.approx(0.8 * everything["sum_isolated_kl"])
    assert result["groups"]["block"]["1"]["sum_isolated_kl"] == pytest.approx(0.005**2 + 0.3**2)
    assert set(result["groups"]["type"]) == {"self_attn.q_proj", "mlp.down_proj"}


def test_markdown_reports_the_two_factors_and_group_interaction() -> None:
    tool = _tool()
    report = {
        "protocol": {"teacher": {"source": "test/model"}}, "measured_at": "now", "measurement_sweep": "sweep",
        "clean": {"token_count": 12}, "projection_types": ["self_attn.q_proj", "mlp.down_proj"],
        "files": [{"file": "m-Q4.gguf", "sha256": "abc", **_isolated(tool)}], "json": "gguf-isolated-kl.json",
    }
    markdown = tool.render_markdown(report)
    assert "| m-Q4.gguf | 0.2801 | 0.1400 | 0.1120 | 0.500 | 0.800 | 0.400 |" in markdown
    assert "| m-Q4.gguf | 0.50 (0.50–0.50) | 1 | 0.50 | 0.50 |" in markdown
    assert "| m-Q4.gguf | 0.800 | 0.80 (0.80–0.80) | 0.80 (0.80–0.80) | 0.80 | 0.80 |" in markdown
