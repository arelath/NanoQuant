from __future__ import annotations

import pytest
import torch

from nanoquant.domain.mlp_operator_refit import (
    coupled_mlp_output_normalized_rmse,
    fit_coupled_mlp_output_scales,
    fit_linear_input_scales,
    fit_linear_output_scales,
    linear_input_scale_normalized_rmse,
)


def test_coupled_mlp_scale_refit_recovers_channel_amplitudes() -> None:
    teacher_gate = torch.tensor(
        [[-1.0, 0.5], [0.2, 1.0], [1.5, -0.4]],
    )
    teacher_up = torch.tensor(
        [[0.7, -1.0], [1.2, 0.5], [-0.3, 1.5]],
    )
    candidate_gate = teacher_gate / torch.tensor([0.8, 1.2])
    candidate_up = teacher_up / torch.tensor([1.5, 0.6])

    result = fit_coupled_mlp_output_scales(
        teacher_gate,
        teacher_up,
        candidate_gate,
        candidate_up,
        minimum_gate_multiplier=0.5,
        maximum_gate_multiplier=1.5,
        gate_grid_points=101,
    )

    assert result.after_normalized_rmse < 1e-5
    assert result.after_normalized_rmse < result.before_normalized_rmse
    assert result.gate_multiplier.tolist() == pytest.approx([0.8, 1.2])
    assert result.up_multiplier.tolist() == pytest.approx([1.5, 0.6])


def test_coupled_mlp_metric_rejects_mismatched_outputs() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        coupled_mlp_output_normalized_rmse(
            torch.ones((2, 3)),
            torch.ones((2, 3)),
            torch.ones((2, 2)),
            torch.ones((2, 3)),
        )


def test_coupled_mlp_scale_refit_retains_identity_fallback() -> None:
    gate = torch.tensor([[-1.0, 0.5], [0.2, 1.0]])
    up = torch.tensor([[0.7, -1.0], [1.2, 0.5]])

    result = fit_coupled_mlp_output_scales(
        gate,
        up,
        gate,
        up,
        minimum_gate_multiplier=0.5,
        maximum_gate_multiplier=1.4,
        gate_grid_points=2,
    )

    assert result.after_normalized_rmse == 0
    assert torch.equal(result.gate_multiplier, torch.ones(2))
    assert torch.equal(result.up_multiplier, torch.ones(2))


def test_coupled_mlp_scale_refit_requires_identity_in_bounds() -> None:
    values = torch.ones((2, 2))

    with pytest.raises(ValueError, match="include the identity"):
        fit_coupled_mlp_output_scales(
            values,
            values,
            values,
            values,
            minimum_gate_multiplier=1.1,
            maximum_gate_multiplier=2.0,
        )


def test_linear_output_scale_refit_recovers_channel_amplitudes() -> None:
    target = torch.tensor(
        [[-1.0, 0.5], [0.2, 1.0], [1.5, -0.4]],
    )
    candidate = target / torch.tensor([0.8, 1.5])

    result = fit_linear_output_scales(target, candidate)

    assert result.after_normalized_rmse < 1e-6
    assert result.after_normalized_rmse < result.before_normalized_rmse
    assert result.multiplier.tolist() == pytest.approx([0.8, 1.5])


def test_linear_output_scale_refit_retains_identity_per_channel() -> None:
    target = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    candidate = target.clone()

    result = fit_linear_output_scales(target, candidate)

    assert result.after_normalized_rmse == 0
    assert torch.equal(result.multiplier, torch.ones(2))


def test_linear_input_scale_refit_recovers_channel_amplitudes() -> None:
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
    )
    weight = torch.tensor([[1.0, 0.25], [-0.5, 1.0]])
    expected_scale = torch.tensor([0.8, 1.5])
    target = (inputs * expected_scale).matmul(weight.mT)

    result = fit_linear_input_scales(
        target,
        inputs,
        weight,
        iterations=40,
        learning_rate=0.5,
    )

    assert result.after_normalized_rmse < 1e-5
    assert result.after_normalized_rmse < result.before_normalized_rmse
    assert result.multiplier.tolist() == pytest.approx(
        expected_scale.tolist(),
        abs=1e-5,
    )


def test_linear_input_scale_metric_rejects_misaligned_weight() -> None:
    with pytest.raises(ValueError, match="finite and aligned"):
        linear_input_scale_normalized_rmse(
            torch.ones((2, 3)),
            torch.ones((2, 2)),
            torch.ones((3, 3)),
        )


def test_linear_input_scale_refit_retains_identity_fallback() -> None:
    inputs = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    weight = torch.tensor([[0.75, -0.5], [0.25, 1.0]])
    target = inputs.matmul(weight.mT)

    result = fit_linear_input_scales(target, inputs, weight)

    assert result.after_normalized_rmse == 0
    assert result.accepted_iterations == 0
    assert torch.equal(result.multiplier, torch.ones(2))
