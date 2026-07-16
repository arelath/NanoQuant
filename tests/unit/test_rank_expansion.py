from __future__ import annotations

import pytest
import torch

from nanoquant.domain.rank_expansion import fit_residual_middle_scales


def test_residual_middle_fit_recovers_weighted_binary_correction() -> None:
    left = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
    right = torch.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]])
    pre = torch.tensor([0.5, 1.0, 1.5, 2.0])
    post = torch.tensor([0.75, 1.25, 1.5])
    expected_mid = torch.tensor([0.4, -0.7])
    residual = (left * post[:, None]) @ (right * expected_mid[:, None] * pre[None, :])

    result = fit_residual_middle_scales(
        residual,
        left,
        right,
        pre,
        post,
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([1.0, 3.0, 2.0]),
    )

    assert result.accepted
    assert result.after_error < result.before_error * 1e-8
    assert result.scale_mid.tolist() == pytest.approx(expected_mid.tolist(), abs=2e-5)
    torch.testing.assert_close(result.correction, residual, atol=2e-5, rtol=2e-5)


def test_residual_middle_fit_never_changes_protected_columns() -> None:
    generator = torch.Generator().manual_seed(4)
    residual = torch.randn((3, 5), generator=generator)
    left = torch.sign(torch.randn((3, 2), generator=generator))
    right = torch.sign(torch.randn((2, 5), generator=generator))

    result = fit_residual_middle_scales(
        residual,
        left,
        right,
        torch.ones(5),
        torch.ones(3),
        torch.ones(5),
        torch.ones(3),
        protected_columns=torch.tensor([1, 4]),
    )

    assert result.after_error <= result.before_error
    assert torch.count_nonzero(result.correction[:, [1, 4]]) == 0
