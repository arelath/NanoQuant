"""Zero-bit coupled output-scale refit for gated MLP input projections."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CoupledMlpScaleRefit:
    gate_multiplier: torch.Tensor
    up_multiplier: torch.Tensor
    before_normalized_rmse: float
    after_normalized_rmse: float


@dataclass(frozen=True, slots=True)
class LinearOutputScaleRefit:
    multiplier: torch.Tensor
    before_normalized_rmse: float
    after_normalized_rmse: float


@dataclass(frozen=True, slots=True)
class LinearInputScaleRefit:
    multiplier: torch.Tensor
    before_normalized_rmse: float
    after_normalized_rmse: float
    accepted_iterations: int


def _validate_outputs(*values: torch.Tensor) -> tuple[int, int]:
    if not values or values[0].ndim != 2 or min(values[0].shape) <= 0:
        raise ValueError("coupled MLP outputs must be non-empty matrices")
    shape = (int(values[0].shape[0]), int(values[0].shape[1]))
    if any(value.ndim != 2 or tuple(value.shape) != shape for value in values):
        raise ValueError("coupled MLP output matrices must share one shape")
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError("coupled MLP outputs must be finite")
    return shape


def linear_output_normalized_rmse(
    target: torch.Tensor,
    candidate: torch.Tensor,
    *,
    multiplier: torch.Tensor | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Measure a linear output after an optional positive channel scale."""

    _rows, channels = _validate_outputs(target, candidate)
    if epsilon <= 0:
        raise ValueError("linear output epsilon must be positive")
    scale = (
        torch.ones(channels, device=candidate.device)
        if multiplier is None
        else multiplier
    )
    if (
        scale.ndim != 1
        or scale.numel() != channels
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(
            "linear output multiplier must be a finite positive channel vector"
        )
    observed = candidate.float() * scale.to(
        device=candidate.device,
        dtype=torch.float32,
    ).reshape(1, -1)
    target32 = target.float()
    return math.sqrt(
        float((observed - target32).square().sum())
        / max(float(target32.square().sum()), epsilon)
    )


def fit_linear_output_scales(
    target: torch.Tensor,
    candidate: torch.Tensor,
    *,
    minimum_multiplier: float = 0.25,
    maximum_multiplier: float = 4.0,
    epsilon: float = 1e-12,
) -> LinearOutputScaleRefit:
    """Fit bounded output scales with an exact identity fallback."""

    _rows, channels = _validate_outputs(target, candidate)
    if (
        not 0 < minimum_multiplier <= 1 <= maximum_multiplier
        or epsilon <= 0
    ):
        raise ValueError(
            "linear output scale-search bounds must include the identity"
        )
    target32 = target.float()
    candidate32 = candidate.float()
    identity_error = (candidate32 - target32).square().sum(dim=0)
    fitted = (
        (target32 * candidate32).sum(dim=0)
        / candidate32.square().sum(dim=0).clamp_min(epsilon)
    ).clamp(minimum_multiplier, maximum_multiplier)
    fitted_error = (
        candidate32 * fitted.reshape(1, -1) - target32
    ).square().sum(dim=0)
    improved = fitted_error < identity_error
    multiplier = torch.where(
        improved,
        fitted,
        torch.ones(channels, device=candidate.device),
    )
    before = linear_output_normalized_rmse(
        target,
        candidate,
        epsilon=epsilon,
    )
    after = linear_output_normalized_rmse(
        target,
        candidate,
        multiplier=multiplier,
        epsilon=epsilon,
    )
    if after > before + 1e-8:
        raise RuntimeError("linear output scale refit regressed its fit objective")
    return LinearOutputScaleRefit(
        multiplier.detach().contiguous(),
        before,
        after,
    )


def linear_input_scale_normalized_rmse(
    target: torch.Tensor,
    candidate_inputs: torch.Tensor,
    candidate_weight: torch.Tensor,
    *,
    multiplier: torch.Tensor | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Measure a linear output after scaling its input channels."""

    if (
        target.ndim != 2
        or candidate_inputs.ndim != 2
        or candidate_weight.ndim != 2
        or target.shape[0] != candidate_inputs.shape[0]
        or target.shape[1] != candidate_weight.shape[0]
        or candidate_inputs.shape[1] != candidate_weight.shape[1]
        or min(target.shape) <= 0
        or not torch.isfinite(target).all()
        or not torch.isfinite(candidate_inputs).all()
        or not torch.isfinite(candidate_weight).all()
    ):
        raise ValueError("linear input-refit dimensions must be finite and aligned")
    if epsilon <= 0:
        raise ValueError("linear input-refit epsilon must be positive")
    channels = candidate_inputs.shape[1]
    scale = (
        torch.ones(channels, device=candidate_inputs.device)
        if multiplier is None
        else multiplier
    )
    if (
        scale.ndim != 1
        or scale.numel() != channels
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(
            "linear input multiplier must be a finite positive channel vector"
        )
    observed = (
        candidate_inputs.float()
        * scale.to(
            device=candidate_inputs.device,
            dtype=torch.float32,
        ).reshape(1, -1)
    ).matmul(candidate_weight.float().mT)
    target32 = target.float()
    return math.sqrt(
        float((observed - target32).square().sum())
        / max(float(target32.square().sum()), epsilon)
    )


def fit_linear_input_scales(
    target: torch.Tensor,
    candidate_inputs: torch.Tensor,
    candidate_weight: torch.Tensor,
    *,
    minimum_multiplier: float = 0.25,
    maximum_multiplier: float = 4.0,
    iterations: int = 20,
    learning_rate: float = 0.25,
    line_search_steps: int = 8,
    epsilon: float = 1e-12,
) -> LinearInputScaleRefit:
    """Fit bounded input scales with diagonal-preconditioned descent."""

    before = linear_input_scale_normalized_rmse(
        target,
        candidate_inputs,
        candidate_weight,
        epsilon=epsilon,
    )
    if (
        not 0 < minimum_multiplier <= 1 <= maximum_multiplier
        or iterations <= 0
        or learning_rate <= 0
        or line_search_steps <= 0
    ):
        raise ValueError(
            "linear input scale-search settings must include the identity"
        )
    target32 = target.float()
    inputs32 = candidate_inputs.float()
    weight32 = candidate_weight.float()
    multiplier = torch.ones(
        inputs32.shape[1],
        device=inputs32.device,
        dtype=torch.float32,
    )
    prediction = inputs32.matmul(weight32.mT)
    current_error = (prediction - target32).square().sum()
    diagonal = (
        inputs32.square().sum(dim=0)
        * weight32.square().sum(dim=0)
    ).clamp_min(epsilon)
    accepted_iterations = 0
    for _iteration in range(iterations):
        residual = target32 - prediction
        projected_residual = residual.matmul(weight32)
        direction = (
            inputs32 * projected_residual
        ).sum(dim=0) / diagonal
        step = learning_rate
        accepted = False
        for _line_search in range(line_search_steps):
            proposed = (multiplier + step * direction).clamp(
                minimum_multiplier,
                maximum_multiplier,
            )
            proposed_prediction = (
                inputs32 * proposed.reshape(1, -1)
            ).matmul(weight32.mT)
            proposed_error = (
                proposed_prediction - target32
            ).square().sum()
            if float(proposed_error) < float(current_error):
                multiplier = proposed
                prediction = proposed_prediction
                current_error = proposed_error
                accepted = True
                accepted_iterations += 1
                break
            step *= 0.5
        if not accepted:
            break
    after = math.sqrt(
        float(current_error)
        / max(float(target32.square().sum()), epsilon)
    )
    if after > before + 1e-8:
        raise RuntimeError("linear input scale refit regressed its fit objective")
    return LinearInputScaleRefit(
        multiplier.detach().contiguous(),
        before,
        after,
        accepted_iterations,
    )


def coupled_mlp_output_normalized_rmse(
    teacher_gate: torch.Tensor,
    teacher_up: torch.Tensor,
    candidate_gate: torch.Tensor,
    candidate_up: torch.Tensor,
    *,
    gate_multiplier: torch.Tensor | None = None,
    up_multiplier: torch.Tensor | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Measure error after the multiplicative gated activation."""

    _rows, channels = _validate_outputs(
        teacher_gate,
        teacher_up,
        candidate_gate,
        candidate_up,
    )
    if epsilon <= 0:
        raise ValueError("coupled MLP epsilon must be positive")
    gate_scale = (
        torch.ones(channels, device=candidate_gate.device)
        if gate_multiplier is None
        else gate_multiplier
    )
    up_scale = (
        torch.ones(channels, device=candidate_up.device)
        if up_multiplier is None
        else up_multiplier
    )
    if (
        gate_scale.ndim != 1
        or up_scale.ndim != 1
        or gate_scale.numel() != channels
        or up_scale.numel() != channels
        or not torch.isfinite(gate_scale).all()
        or not torch.isfinite(up_scale).all()
        or (gate_scale <= 0).any()
        or (up_scale <= 0).any()
    ):
        raise ValueError(
            "coupled MLP multipliers must be finite positive channel vectors"
        )
    gate_scale = gate_scale.to(
        device=candidate_gate.device,
        dtype=torch.float32,
    )
    up_scale = up_scale.to(
        device=candidate_up.device,
        dtype=torch.float32,
    )
    target = F.silu(teacher_gate.float()) * teacher_up.float()
    observed = F.silu(
        candidate_gate.float() * gate_scale.float().reshape(1, -1)
    ) * (candidate_up.float() * up_scale.float().reshape(1, -1))
    return math.sqrt(
        float((observed - target).square().sum())
        / max(float(target.square().sum()), epsilon)
    )


def fit_coupled_mlp_output_scales(
    teacher_gate: torch.Tensor,
    teacher_up: torch.Tensor,
    candidate_gate: torch.Tensor,
    candidate_up: torch.Tensor,
    *,
    minimum_gate_multiplier: float = 0.5,
    maximum_gate_multiplier: float = 1.5,
    gate_grid_points: int = 41,
    minimum_up_multiplier: float = 0.25,
    maximum_up_multiplier: float = 4.0,
    epsilon: float = 1e-12,
) -> CoupledMlpScaleRefit:
    """Grid-search gate scales and solve the paired up scale per channel."""

    _rows, channels = _validate_outputs(
        teacher_gate,
        teacher_up,
        candidate_gate,
        candidate_up,
    )
    if (
        not 0 < minimum_gate_multiplier <= 1 <= maximum_gate_multiplier
        or gate_grid_points < 2
        or not 0 < minimum_up_multiplier <= 1 <= maximum_up_multiplier
        or epsilon <= 0
    ):
        raise ValueError(
            "coupled MLP scale-search bounds must include the identity"
        )
    target = F.silu(teacher_gate.float()) * teacher_up.float()
    gate = candidate_gate.float()
    up = candidate_up.float()
    best_error = (F.silu(gate) * up - target).square().sum(dim=0)
    best_gate = torch.ones_like(best_error)
    best_up = torch.ones_like(best_error)
    for gate_scale in torch.linspace(
        minimum_gate_multiplier,
        maximum_gate_multiplier,
        gate_grid_points,
        device=gate.device,
    ):
        basis = F.silu(gate * gate_scale) * up
        up_scale = (
            (target * basis).sum(dim=0)
            / basis.square().sum(dim=0).clamp_min(epsilon)
        ).clamp(minimum_up_multiplier, maximum_up_multiplier)
        error = (
            basis * up_scale.reshape(1, -1) - target
        ).square().sum(dim=0)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_gate = torch.where(improved, gate_scale, best_gate)
        best_up = torch.where(improved, up_scale, best_up)
    before = coupled_mlp_output_normalized_rmse(
        teacher_gate,
        teacher_up,
        candidate_gate,
        candidate_up,
        epsilon=epsilon,
    )
    after = coupled_mlp_output_normalized_rmse(
        teacher_gate,
        teacher_up,
        candidate_gate,
        candidate_up,
        gate_multiplier=best_gate,
        up_multiplier=best_up,
        epsilon=epsilon,
    )
    if after > before + 1e-8:
        raise RuntimeError("coupled MLP scale refit regressed its fit objective")
    return CoupledMlpScaleRefit(
        best_gate.detach().contiguous(),
        best_up.detach().contiguous(),
        before,
        after,
    )
