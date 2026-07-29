from __future__ import annotations

import pytest
import torch

from nanoquant.domain.covariance_refinement import refine_binary_factors_under_covariance
from nanoquant.domain.metrics import dense_hessian_squared_error
from nanoquant.domain.scale_fit import reconstruct


def test_covariance_refinement_improves_exact_metric_and_preserves_format() -> None:
    desired_left = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    desired_right = torch.tensor([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]])
    pre = torch.tensor([0.7, 1.1, 0.8])
    mid = torch.tensor([0.9, -0.6])
    post = torch.tensor([1.2, 0.5, -0.8])
    covariance = torch.tensor(
        [
            [1.5, 0.3, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.3],
        ]
    )
    output = torch.tensor([0.8, 1.4, 0.6])
    target = reconstruct(desired_left, desired_right, pre, mid, post)
    initial_left = desired_left.clone()
    initial_right = desired_right.clone()
    initial_left[0, 0] *= -1
    initial_right[1, 2] *= -1

    result = refine_binary_factors_under_covariance(
        target,
        initial_left,
        initial_right,
        pre,
        mid,
        post,
        covariance,
        output,
        left_steps=8,
        right_batches=8,
        right_batch_size=2,
    )

    assert result.after_error < result.before_error
    assert result.left_flips + result.right_flips > 0
    assert set(result.left_binary.unique().tolist()) <= {-1.0, 1.0}
    assert set(result.right_binary.unique().tolist()) <= {-1.0, 1.0}
    measured = float(
        dense_hessian_squared_error(target, result.reconstruction, covariance, output)
    )
    assert measured == pytest.approx(result.after_error, rel=1e-5)


def test_covariance_refinement_keeps_protected_pre_scales_zero() -> None:
    left = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    right = torch.tensor([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]])
    target = torch.randn((2, 3), generator=torch.Generator().manual_seed(4))

    result = refine_binary_factors_under_covariance(
        target,
        left,
        right,
        torch.ones(3),
        torch.ones(2),
        torch.ones(2),
        torch.eye(3),
        torch.ones(2),
        protected_columns=torch.tensor([1]),
        left_steps=2,
        right_batches=2,
    )

    assert result.scale_pre[1] == 0


def test_covariance_refinement_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        refine_binary_factors_under_covariance(
            torch.ones((2, 3)),
            torch.ones((2, 1)),
            torch.ones((1, 3)),
            torch.ones(3),
            torch.ones(1),
            torch.ones(2),
            torch.eye(2),
            torch.ones(2),
        )
