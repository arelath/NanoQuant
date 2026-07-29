from __future__ import annotations

import pytest
import torch

from nanoquant.domain.metrics import dense_hessian_squared_error
from nanoquant.domain.scale_fit import reconstruct
from tools.probe_covariance_binary import (
    _reconstruction_set,
    fit_covariance_scales,
    left_flip_deltas,
    refine_covariance_signs,
    right_flip_deltas,
)
from tools.probe_input_hadamard import block_groups


def _fixture() -> tuple[torch.Tensor, ...]:
    left = torch.tensor(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    right = torch.tensor(
        [
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]
    )
    pre = torch.tensor([0.7, 1.1, 0.8, 1.3])
    mid = torch.tensor([0.9, -0.6])
    post = torch.tensor([1.2, 0.5, -0.8])
    covariance = torch.tensor(
        [
            [1.5, 0.3, 0.1, 0.0],
            [0.3, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.3, 0.4],
            [0.0, 0.1, 0.4, 1.2],
        ]
    )
    output = torch.tensor([0.8, 1.4, 0.6])
    target = torch.tensor(
        [
            [0.2, -1.0, 0.7, -0.4],
            [-0.3, 0.1, 0.8, 0.6],
            [1.1, -0.2, -0.5, 0.9],
        ]
    )
    return target, left, right, pre, mid, post, covariance, output


def _error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    covariance: torch.Tensor,
    output: torch.Tensor,
) -> float:
    return float(dense_hessian_squared_error(target, prediction, covariance, output))


def test_left_flip_deltas_match_brute_force() -> None:
    target, left, right, pre, mid, post, covariance, output = _fixture()
    baseline = _error(target, reconstruct(left, right, pre, mid, post), covariance, output)
    actual = left_flip_deltas(target, left, right, pre, mid, post, covariance)

    for row in range(left.shape[0]):
        for rank in range(left.shape[1]):
            candidate = left.clone()
            candidate[row, rank] *= -1
            delta = _error(
                target,
                reconstruct(candidate, right, pre, mid, post),
                covariance,
                output,
            ) - baseline
            assert actual[row, rank] * output[row] == pytest.approx(delta, rel=1e-5, abs=1e-5)


def test_right_flip_deltas_match_brute_force() -> None:
    target, left, right, pre, mid, post, covariance, output = _fixture()
    baseline = _error(target, reconstruct(left, right, pre, mid, post), covariance, output)
    actual = right_flip_deltas(
        target,
        left,
        right,
        pre,
        mid,
        post,
        covariance,
        output,
    )

    for rank in range(right.shape[0]):
        for column in range(right.shape[1]):
            candidate = right.clone()
            candidate[rank, column] *= -1
            delta = _error(
                target,
                reconstruct(left, candidate, pre, mid, post),
                covariance,
                output,
            ) - baseline
            assert actual[rank, column] == pytest.approx(delta, rel=1e-5, abs=1e-5)


def test_covariance_scale_fit_is_monotone_and_recovers_known_target() -> None:
    _target, left, right, pre, mid, post, covariance, output = _fixture()
    target = reconstruct(left, right, pre, mid, post)
    initial_pre = pre * torch.tensor([1.3, 0.7, 1.2, 0.8])
    initial_mid = mid * torch.tensor([0.8, 1.4])
    initial_post = post * torch.tensor([1.1, 0.6, 1.3])

    result = fit_covariance_scales(
        target,
        left,
        right,
        initial_pre,
        initial_mid,
        initial_post,
        covariance,
        output,
        alternating_passes=4,
    )

    assert result.accepted is True
    assert result.after_error < result.before_error
    assert result.after_error / result.before_error < 1e-3


def test_covariance_sign_refinement_accepts_only_objective_improvements() -> None:
    _target, desired_left, desired_right, pre, mid, post, covariance, output = _fixture()
    target = reconstruct(desired_left, desired_right, pre, mid, post)
    initial_left = desired_left.clone()
    initial_right = desired_right.clone()
    initial_left[0, 0] *= -1
    initial_right[1, 3] *= -1

    result = refine_covariance_signs(
        target,
        initial_left,
        initial_right,
        pre,
        mid,
        post,
        covariance,
        output,
        left_steps=4,
        right_batches=4,
        right_batch_size=2,
    )

    assert result.after_error <= result.before_error
    assert result.left_flips + result.right_flips > 0
    assert result.after_error < result.before_error * 1e-5


def test_covariance_refinement_rejects_shape_and_setting_errors() -> None:
    target, left, right, pre, mid, post, covariance, output = _fixture()

    with pytest.raises(ValueError, match="dimensions"):
        fit_covariance_scales(
            target,
            left,
            right,
            pre[:-1],
            mid,
            post,
            covariance,
            output,
        )
    with pytest.raises(ValueError, match="settings"):
        refine_covariance_signs(
            target,
            left,
            right,
            pre,
            mid,
            post,
            covariance,
            output,
            left_steps=-1,
        )


def test_covariance_reconstruction_inventory_accepts_any_complete_block_count() -> None:
    groups = block_groups(4)
    results = {
        f"4:{group.label}": {
            "evaluation": {
                "original_error": 1.0,
                "original_target": 2.0,
            }
        }
        for group in groups
    }
    members = {
        f"4:{group.label}": tuple(
            (member, torch.ones((1, 1)), 1.0) for member in group.members
        )
        for group in groups
    }

    reconstruction = _reconstruction_set(results, members)

    assert len(reconstruction.layers) == 7
    with pytest.raises(ValueError, match="complete blocks"):
        _reconstruction_set(
            {key: value for key, value in results.items() if key != "4:down"},
            {key: value for key, value in members.items() if key != "4:down"},
        )
