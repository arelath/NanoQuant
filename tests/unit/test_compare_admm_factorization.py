from pathlib import Path

import pytest
import torch

from tools.compare_admm_factorization import _load_legacy_factorizer, _tensor_comparison


def test_load_legacy_factorizer_uses_standalone_source(tmp_path: Path) -> None:
    source = tmp_path / "admm_nq.py"
    source.write_text(
        "def factorize_admm_nanoquant(value, *args, **kwargs):\n"
        "    return {'W_final': value + 1}\n",
        encoding="utf-8",
    )

    factorizer = _load_legacy_factorizer(source)

    assert torch.equal(factorizer(torch.tensor([2.0]))["W_final"], torch.tensor([3.0]))


def test_tensor_comparison_reports_exact_and_relative_differences() -> None:
    exact = _tensor_comparison(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    different = _tensor_comparison(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 2.0]))

    assert exact == {
        "shape_equal": True,
        "exact": True,
        "agreement": 1.0,
        "maximum_absolute_difference": 0.0,
        "relative_l2_difference": 0.0,
    }
    assert different["exact"] is False
    assert different["agreement"] == 0.5
    assert different["maximum_absolute_difference"] == 1.0
    assert different["relative_l2_difference"] == pytest.approx(1 / 5**0.5)


def test_tensor_comparison_rejects_shape_parity() -> None:
    result = _tensor_comparison(torch.ones(2), torch.ones(2, 1))
    assert result == {"shape_equal": False, "left_shape": [2], "right_shape": [2, 1]}
