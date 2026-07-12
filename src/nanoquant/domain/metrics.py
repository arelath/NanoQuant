"""Pure reconstruction metrics."""

from __future__ import annotations

import torch

from .models import ReconstructionMetrics


def raw_squared_error(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    _same_shape(target, prediction)
    return (prediction.float() - target.float()).square().sum()


def per_element_squared_error(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Mean raw squared error, with the denominator explicit in the function name."""
    _same_shape(target, prediction)
    if target.numel() == 0:
        raise ValueError("per-element error is undefined for an empty tensor")
    return raw_squared_error(target, prediction) / target.numel()


def weighted_squared_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> torch.Tensor:
    _same_shape(target, prediction)
    _importance_shapes(target, input_importance, output_importance)
    delta = prediction.float() - target.float()
    weights = output_importance.detach().float().reshape(-1, 1) * input_importance.detach().float().reshape(1, -1)
    return (delta.square() * weights).sum()


def dense_hessian_squared_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
) -> torch.Tensor:
    _same_shape(target, prediction)
    if covariance.shape != (target.shape[1], target.shape[1]):
        raise ValueError("covariance shape does not match input dimension")
    delta = prediction.float() - target.float()
    row_errors = ((delta @ covariance.float()) * delta).sum(dim=1)
    return (row_errors * output_importance.detach().float().reshape(-1)).sum()


def normalized(error: torch.Tensor, target_norm: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    return error / target_norm.clamp_min(epsilon)


def reconstruction_metrics(
    target: torch.Tensor,
    export_prediction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    objective_mode: str = "diagonal",
    latent_prediction: torch.Tensor | None = None,
    unwhitened_prediction: torch.Tensor | None = None,
) -> ReconstructionMetrics:
    target_weighted = weighted_squared_error(target, torch.zeros_like(target), input_importance, output_importance)
    target_raw = raw_squared_error(target, torch.zeros_like(target))
    export_weighted = weighted_squared_error(target, export_prediction, input_importance, output_importance)
    raw = raw_squared_error(target, export_prediction)
    unwhitened = unwhitened_prediction if unwhitened_prediction is not None else export_prediction
    unwhitened_error = weighted_squared_error(target, unwhitened, input_importance, output_importance)
    latent_error = (
        weighted_squared_error(target, latent_prediction, input_importance, output_importance)
        if latent_prediction is not None
        else None
    )
    return ReconstructionMetrics(
        objective_mode=objective_mode,
        target_weighted_norm_squared=float(target_weighted),
        latent_weighted_error=None if latent_error is None else float(latent_error),
        latent_weighted_normalized_error=None
        if latent_error is None
        else float(normalized(latent_error, target_weighted)),
        unwhitened_weighted_error=float(unwhitened_error),
        unwhitened_weighted_normalized_error=float(normalized(unwhitened_error, target_weighted)),
        export_weighted_error=float(export_weighted),
        export_weighted_normalized_error=float(normalized(export_weighted, target_weighted)),
        raw_error=float(raw),
        raw_normalized_error=float(normalized(raw, target_raw)),
    )


def _same_shape(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("target and prediction must be same-shaped matrices")


def _importance_shapes(weight: torch.Tensor, input_importance: torch.Tensor, output_importance: torch.Tensor) -> None:
    if input_importance.numel() != weight.shape[1] or output_importance.numel() != weight.shape[0]:
        raise ValueError("importance vector lengths do not match weight dimensions")
