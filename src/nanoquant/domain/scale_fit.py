"""Alternating weighted least-squares scale fitting with rollback."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MaterializedScaleFitResult:
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    reconstruction: torch.Tensor
    before_error: float
    after_error: float
    accepted: bool
    rollback_reason: str | None


def reconstruct(
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
) -> torch.Tensor:
    return (left_binary.float() * scale_post.float().reshape(-1, 1)) @ (
        right_binary.float() * scale_mid.float().reshape(-1, 1) * scale_pre.float().reshape(1, -1)
    )


def _weighted_error(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    total = torch.zeros((), device=target.device, dtype=torch.float32)
    weighted_right = right * (mid[:, None] * pre[None, :])
    for start in range(0, target.shape[0], chunk_rows):
        end = min(start + chunk_rows, target.shape[0])
        prediction = (left[start:end] * post[start:end, None]) @ weighted_right
        difference = prediction - target[start:end]
        total += (
            difference.square()
            * output_importance[start:end, None]
            * input_importance[None, :]
        ).sum()
    return total


def _weighted_prediction_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    total = torch.zeros((), device=target.device, dtype=torch.float32)
    for start in range(0, target.shape[0], chunk_rows):
        end = min(start + chunk_rows, target.shape[0])
        difference = prediction[start:end].float() - target[start:end]
        total += (
            difference.square()
            * output_importance[start:end, None]
            * input_importance[None, :]
        ).sum()
    return total


def _fit_post(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    input_importance: torch.Tensor,
    epsilon: float,
    chunk_rows: int,
) -> torch.Tensor:
    result = torch.empty(target.shape[0], device=target.device, dtype=torch.float32)
    weighted_right = right * (mid[:, None] * pre[None, :])
    for start in range(0, target.shape[0], chunk_rows):
        end = min(start + chunk_rows, target.shape[0])
        base = left[start:end] @ weighted_right
        numerator = (base * target[start:end] * input_importance[None, :]).sum(dim=1)
        denominator = (base.square() * input_importance[None, :]).sum(dim=1).clamp_min(epsilon)
        result[start:end] = numerator / denominator
    return torch.nan_to_num(result)


def _fit_pre(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    output_importance: torch.Tensor,
    epsilon: float,
    chunk_rows: int,
) -> torch.Tensor:
    numerator = torch.zeros(target.shape[1], device=target.device, dtype=torch.float32)
    denominator = torch.zeros_like(numerator)
    for start in range(0, target.shape[0], chunk_rows):
        end = min(start + chunk_rows, target.shape[0])
        weighted_left = left[start:end] * (post[start:end, None] * mid[None, :])
        base = weighted_left @ right
        row_weight = output_importance[start:end, None]
        numerator += (base * target[start:end] * row_weight).sum(dim=0)
        denominator += (base.square() * row_weight).sum(dim=0)
    return torch.nan_to_num(numerator / denominator.clamp_min(epsilon))


def _fit_mid(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    scaled_left = left * post[:, None]
    scaled_right = right * pre[None, :]
    left_gram = scaled_left.mT @ (scaled_left * output_importance[:, None])
    weighted_right = scaled_right * input_importance.sqrt()[None, :]
    right_gram = weighted_right @ weighted_right.mT
    cross = scaled_left.mT @ (target * input_importance[None, :] * output_importance[:, None])
    rhs = (cross * scaled_right).sum(dim=1)
    system = left_gram * right_gram
    system = 0.5 * (system + system.mT)
    ridge = torch.clamp(system.diagonal().mean().abs() * 1e-6, min=epsilon)
    system.diagonal().add_(ridge)
    cholesky, info = torch.linalg.cholesky_ex(system, upper=False)
    result = (
        torch.cholesky_solve(rhs[:, None], cholesky, upper=False).squeeze(1)
        if int(info.item()) == 0
        else torch.linalg.lstsq(system, rhs[:, None]).solution.squeeze(1)
    )
    return torch.nan_to_num(result)


def fit_scales(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    alternating_passes: int = 2,
    epsilon: float = 1e-8,
    protected_columns: torch.Tensor | None = None,
    rollback_on_regression: bool = True,
    chunk_rows: int = 512,
) -> MaterializedScaleFitResult:
    if alternating_passes < 0 or epsilon <= 0 or chunk_rows <= 0:
        raise ValueError("scale-fit settings are invalid")
    left = torch.sign(left_binary.detach().float())
    right = torch.sign(right_binary.detach().float())
    pre = scale_pre.detach().float().reshape(-1).clone()
    mid = scale_mid.detach().float().reshape(-1).clone()
    post = scale_post.detach().float().reshape(-1).clone()
    target32 = target.detach().float()
    input_weight = input_importance.detach().float().reshape(-1).clamp_min(epsilon)
    output_weight = output_importance.detach().float().reshape(-1).clamp_min(epsilon)
    original = reconstruct(left, right, pre, mid, post).to(target.dtype)
    before_tensor = _weighted_error(
        target32, left, right, pre, mid, post, input_weight, output_weight, chunk_rows
    )
    protected = None if protected_columns is None else protected_columns.detach().long().reshape(-1)
    best_error = before_tensor
    best_pre = pre.clone()
    best_mid = mid.clone()
    best_post = post.clone()
    for _ in range(alternating_passes):
        post = _fit_post(target32, left, right, pre, mid, input_weight, epsilon, chunk_rows)
        pre = _fit_pre(target32, left, right, mid, post, output_weight, epsilon, chunk_rows)
        if protected is not None:
            pre[protected] = 0
        mid = _fit_mid(target32, left, right, pre, post, input_weight, output_weight, epsilon)
        current_error = _weighted_error(
            target32, left, right, pre, mid, post, input_weight, output_weight, chunk_rows
        )
        if bool(torch.isfinite(current_error)) and float(current_error) < float(best_error):
            best_error = current_error
            best_pre = pre.clone()
            best_mid = mid.clone()
            best_post = post.clone()
    candidate = reconstruct(left, right, best_pre, best_mid, best_post).to(target.dtype)
    export_before = float(
        _weighted_prediction_error(
            target32, original, input_weight, output_weight, chunk_rows
        )
    )
    export_after = float(
        _weighted_prediction_error(
            target32, candidate, input_weight, output_weight, chunk_rows
        )
    )
    finite = torch.isfinite(candidate).all().item() and math_is_finite(export_after)
    if not finite or (rollback_on_regression and export_after > export_before):
        reason = "non_finite_candidate" if not finite else "export_weighted_objective_regressed"
        return MaterializedScaleFitResult(
            scale_pre.detach().clone(),
            scale_mid.detach().clone(),
            scale_post.detach().clone(),
            original,
            export_before,
            export_before,
            False,
            reason,
        )
    return MaterializedScaleFitResult(
        best_pre,
        best_mid,
        best_post,
        candidate,
        export_before,
        export_after,
        True,
        None,
    )


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
