from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

REPOSITORY = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    tools = str(REPOSITORY / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("measure_block_scaled_noise_kl")


def test_block_scaled_noise_hits_the_error_and_follows_each_blocks_magnitude() -> None:
    tool = _tool()
    weight = torch.randn(8, 64, generator=torch.Generator().manual_seed(0))
    weight[:, :32] *= 100
    weight[3, 32:] = 0
    noise = tool.block_scaled_noise(weight, 0.05, 32, torch.Generator().manual_seed(1))
    assert float(noise.norm() / weight.norm()) == pytest.approx(0.05, rel=1e-5)
    assert float(noise[3, 32:].abs().max()) == 0
    assert float(noise[:, :32].abs().mean() / noise[:, 32:].abs().mean()) > 30
    with pytest.raises(ValueError, match="blocks of 48"):
        tool.block_scaled_noise(weight, 0.05, 48, torch.Generator())


def test_markdown_reports_ratios_by_type_and_the_joint_runs() -> None:
    tool = _tool()
    report = {
        "protocol": {"teacher": {"source": "test/model"}}, "measured_at": "now", "measurement_sweep": "sweep",
        "level": 0.05, "block": 32, "projection_types": ["self_attn.q_proj", "mlp.down_proj"],
        "projections": {
            "0:self_attn.q_proj": {"gaussian_prediction": 0.02, "kl_above_control": 0.002},
            "1:self_attn.q_proj": {"gaussian_prediction": 0.02, "kl_above_control": 0.006},
            "0:mlp.down_proj": {"gaussian_prediction": 0.04, "kl_above_control": 0.02},
        },
        "sum_isolated_kl": 0.028, "gaussian_prediction": 0.08,
        "joint": [{"seed": 0, "joint_kl": 0.0266, "joint_over_isolated": 0.95}], "json": "block.json",
    }
    markdown = tool.render_markdown(report)
    assert "| self_attn.q_proj | 0.20 | 0.10–0.30 |" in markdown
    assert "| mlp.down_proj | 0.50 | 0.50–0.50 |" in markdown
    assert "(ratio 0.350)" in markdown and "isolated KLs = 0.950." in markdown
