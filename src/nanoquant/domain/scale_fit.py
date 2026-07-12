"""Alternating weighted least-squares scale fitting with rollback."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .metrics import weighted_squared_error


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
) -> MaterializedScaleFitResult:
    if alternating_passes < 0 or epsilon <= 0:
        raise ValueError("scale-fit settings are invalid")
    left = torch.sign(left_binary.detach().float())
    right = torch.sign(right_binary.detach().float())
    pre = scale_pre.detach().float().reshape(-1).clone()
    mid = scale_mid.detach().float().reshape(-1).clone()
    post = scale_post.detach().float().reshape(-1).clone()
    original = reconstruct(left, right, pre, mid, post)
    before = float(weighted_squared_error(target, original, input_importance, output_importance))
    protected = None if protected_columns is None else protected_columns.detach().long().reshape(-1)
    for _ in range(alternating_passes):
        weighted_right = right * mid.reshape(-1, 1) * pre.reshape(1, -1)
        base = left @ weighted_right
        numerator = (base * target.float() * input_importance.float().reshape(1, -1)).sum(dim=1)
        denominator = (base.square() * input_importance.float().reshape(1, -1)).sum(dim=1).clamp_min(epsilon)
        post = torch.nan_to_num(numerator / denominator)
        weighted_left = left * post.reshape(-1, 1) * mid.reshape(1, -1)
        base = weighted_left @ right
        numerator = (base * target.float() * output_importance.float().reshape(-1, 1)).sum(dim=0)
        denominator = (base.square() * output_importance.float().reshape(-1, 1)).sum(dim=0).clamp_min(epsilon)
        pre = torch.nan_to_num(numerator / denominator)
        if protected is not None:
            pre[protected] = 0
        left_scaled = left * post.reshape(-1, 1)
        right_scaled = right * pre.reshape(1, -1)
        left_gram = left_scaled.mT @ (left_scaled * output_importance.float().reshape(-1, 1))
        right_weighted = right_scaled * input_importance.float().sqrt().reshape(1, -1)
        system = left_gram * (right_weighted @ right_weighted.mT)
        system = 0.5 * (system + system.mT)
        system.diagonal().add_(system.diagonal().mean().abs() * 1e-6 + epsilon)
        cross = left_scaled.mT @ (target.float() * output_importance.float().reshape(-1, 1))
        rhs = (cross * (right_scaled * input_importance.float().reshape(1, -1))).sum(dim=1)
        mid = torch.nan_to_num(torch.linalg.lstsq(system, rhs.reshape(-1, 1)).solution.reshape(-1))
    candidate = reconstruct(left, right, pre, mid, post).to(target.dtype)
    after = float(weighted_squared_error(target, candidate, input_importance, output_importance))
    finite = torch.isfinite(candidate).all().item() and math_is_finite(after)
    if not finite or (rollback_on_regression and after > before):
        reason = "non_finite_candidate" if not finite else "weighted_objective_regressed"
        return MaterializedScaleFitResult(
            scale_pre.detach().clone(),
            scale_mid.detach().clone(),
            scale_post.detach().clone(),
            original.to(target.dtype),
            before,
            before,
            False,
            reason,
        )
    return MaterializedScaleFitResult(pre, mid, post, candidate, before, after, True, None)


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
