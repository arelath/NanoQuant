from pathlib import Path

import pytest
import torch

from nanoquant.domain.models import (
    BlockId,
    LayerId,
)
from tools.compare_diagonal_objective import _extract_legacy_weighted_error, compare_layer


def test_extracts_and_executes_exact_legacy_function(tmp_path: Path) -> None:
    source = tmp_path / "compress_block.py"
    source.write_text(
        "import torch\n\n"
        "def _weighted_weight_error(pred: torch.Tensor, target: torch.Tensor, i_norm: torch.Tensor, "
        "o_norm: torch.Tensor, input_cholesky=None, chunk_rows: int = 512, target_norm=None):\n"
        "    i_weight = i_norm.float().clamp_min(1e-12)\n"
        "    o_weight = o_norm.float().clamp_min(1e-12)\n"
        "    error = ((pred.float() - target.float()).square() * o_weight[:, None] * i_weight[None, :]).sum()\n"
        "    norm = (target.float().square() * o_weight[:, None] * i_weight[None, :]).sum()\n"
        "    return error, norm, error / norm.clamp_min(1e-12)\n",
        encoding="utf-8",
    )

    oracle, start, end = _extract_legacy_weighted_error(source)

    assert (start, end) == (3, 8)
    error, norm, normalized = oracle(torch.ones((2, 2)), torch.zeros((2, 2)), torch.ones(2), torch.ones(2))
    assert float(error) == 4.0
    assert float(norm) == 0.0
    assert float(normalized) == pytest.approx(4e12)


def test_compare_layer_accepts_legacy_formula_with_zero_importance(tmp_path: Path) -> None:
    source = tmp_path / "compress_block.py"
    source.write_text(
        "def _weighted_weight_error(pred, target, i_norm, o_norm, input_cholesky=None, "
        "chunk_rows=512, target_norm=None):\n"
        "    i_weight = i_norm.float().clamp_min(1e-12)\n"
        "    o_weight = o_norm.float().clamp_min(1e-12)\n"
        "    error = ((pred.float() - target.float()).square() * o_weight[:, None] * i_weight[None, :]).sum()\n"
        "    norm = (target.float().square() * o_weight[:, None] * i_weight[None, :]).sum()\n"
        "    return error, norm, error / norm.clamp_min(1e-12)\n",
        encoding="utf-8",
    )
    oracle, _, _ = _extract_legacy_weighted_error(source)

    result = compare_layer(
        LayerId(BlockId(0), "self_attn.q_proj"),
        torch.ones((2, 2)),
        torch.zeros((2, 2)),
        torch.tensor([0.0, 2.0]),
        torch.tensor([0.0, 3.0]),
        oracle,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )

    assert result["passed"] is True
    assert result["input_importance_at_or_below_floor"] == 1
    assert result["output_importance_at_or_below_floor"] == 1
