"""Dense-covariance refinement for the existing binary-factor representation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nanoquant.domain.metrics import dense_hessian_squared_error
from nanoquant.domain.scale_fit import reconstruct


@dataclass(frozen=True, slots=True)
class CovarianceBinaryRefinement:
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    reconstruction: torch.Tensor
    before_error: float
    after_error: float
    left_flips: int
    right_flips: int


def _error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
) -> float:
    return float(dense_hessian_squared_error(target, prediction, covariance, output_importance))


def _solve(system: torch.Tensor, rhs: torch.Tensor, epsilon: float) -> torch.Tensor:
    value = 0.5 * (system.float() + system.float().mT)
    value.diagonal().add_((value.diagonal().mean().abs() * 1e-6).clamp_min(epsilon))
    factor, info = torch.linalg.cholesky_ex(value)
    result = (
        torch.cholesky_solve(rhs.float().reshape(-1, 1), factor).reshape(-1)
        if int(info.max()) == 0
        else torch.linalg.lstsq(value, rhs.float().reshape(-1, 1)).solution.reshape(-1)
    )
    return torch.nan_to_num(result)


def _fit_scales(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    covariance: torch.Tensor,
    output: torch.Tensor,
    protected_columns: torch.Tensor | None,
    passes: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    best_prediction = reconstruct(left, right, pre, mid, post)
    best_error = _error(target, best_prediction, covariance, output)
    best = (pre.clone(), mid.clone(), post.clone(), best_prediction)
    for _ in range(passes):
        base = (left * mid.reshape(1, -1)) @ (right * pre.reshape(1, -1))
        base_covariance = base @ covariance
        post = torch.nan_to_num(
            (base_covariance * target).sum(dim=1)
            / (base_covariance * base).sum(dim=1).clamp_min(epsilon)
        )

        weighted_left = left * (post.reshape(-1, 1) * mid.reshape(1, -1))
        base = weighted_left @ right
        pre_system = (base.mT @ (base * output.reshape(-1, 1))) * covariance
        target_covariance = (target * output.reshape(-1, 1)) @ covariance
        pre = _solve(pre_system, (base * target_covariance).sum(dim=0), epsilon)
        if protected_columns is not None:
            pre[protected_columns] = 0

        scaled_left = left * post.reshape(-1, 1)
        scaled_right = right * pre.reshape(1, -1)
        left_gram = scaled_left.mT @ (scaled_left * output.reshape(-1, 1))
        right_covariance = scaled_right @ covariance
        mid_system = left_gram * (right_covariance @ scaled_right.mT)
        cross = scaled_left.mT @ target_covariance
        mid = _solve(mid_system, (cross * scaled_right).sum(dim=1), epsilon)

        candidate = reconstruct(left, right, pre, mid, post)
        candidate_error = _error(target, candidate, covariance, output)
        if math.isfinite(candidate_error) and candidate_error < best_error:
            best_error = candidate_error
            best = (pre.clone(), mid.clone(), post.clone(), candidate)
    return (*best[:3], best[3], best_error)


def refine_binary_factors_under_covariance(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    protected_columns: torch.Tensor | None = None,
    scale_passes: int = 2,
    left_steps: int = 32,
    right_batches: int = 16,
    right_batch_size: int = 128,
    epsilon: float = 1e-8,
) -> CovarianceBinaryRefinement:
    """Refine signs and separable scales without changing rank or storage."""

    if target.ndim != 2 or left_binary.ndim != 2 or right_binary.ndim != 2:
        raise ValueError("covariance refinement target and factors must be matrices")
    if (
        left_binary.shape[0] != target.shape[0]
        or right_binary.shape[1] != target.shape[1]
        or left_binary.shape[1] != right_binary.shape[0]
        or covariance.shape != (target.shape[1], target.shape[1])
        or output_importance.numel() != target.shape[0]
    ):
        raise ValueError("covariance refinement dimensions do not match")
    if scale_passes < 0 or left_steps < 0 or right_batches < 0 or right_batch_size <= 0:
        raise ValueError("covariance refinement settings are invalid")

    target32 = target.detach().float()
    left = torch.sign(left_binary.detach().float()).clone()
    right = torch.sign(right_binary.detach().float()).clone()
    pre = scale_pre.detach().float().reshape(-1).clone()
    mid = scale_mid.detach().float().reshape(-1).clone()
    post = scale_post.detach().float().reshape(-1).clone()
    metric = 0.5 * (covariance.detach().float() + covariance.detach().float().mT)
    output = output_importance.detach().float().reshape(-1).clamp_min(epsilon)
    protected = (
        None
        if protected_columns is None
        else protected_columns.detach().long().reshape(-1)
    )
    if protected is not None:
        pre[protected] = 0
    initial = reconstruct(left, right, pre, mid, post)
    before_error = _error(target32, initial, metric, output)
    pre, mid, post, _prediction, _scale_error = _fit_scales(
        target32,
        left,
        right,
        pre,
        mid,
        post,
        metric,
        output,
        protected,
        scale_passes,
        epsilon,
    )

    scaled_right = right * (mid.reshape(-1, 1) * pre.reshape(1, -1))
    right_covariance = scaled_right @ metric
    left_gram = right_covariance @ scaled_right.mT
    left_cross = (target32 @ metric) @ scaled_right.mT
    coefficients = left * post.reshape(-1, 1)
    state = coefficients @ left_gram - left_cross
    rows = torch.arange(left.shape[0], device=left.device)
    left_flips = 0
    for _ in range(left_steps):
        deltas = (
            -4 * post.reshape(-1, 1) * left * state
            + 4 * post.square().reshape(-1, 1) * left_gram.diagonal().reshape(1, -1)
        )
        best_delta, indices = deltas.min(dim=1)
        active = best_delta < 0
        if not bool(active.any()):
            break
        active_rows = rows[active]
        active_indices = indices[active]
        changes = -2 * coefficients[active_rows, active_indices]
        left[active_rows, active_indices] = -left[active_rows, active_indices]
        coefficients[active_rows, active_indices] += changes
        state[active_rows] += left_gram.index_select(0, active_indices) * changes.reshape(-1, 1)
        left_flips += int(active.sum())

    scaled_left = left * (post.reshape(-1, 1) * mid.reshape(1, -1))
    right_gram = scaled_left.mT @ (scaled_left * output.reshape(-1, 1))
    right_cross = scaled_left.mT @ (target32 * output.reshape(-1, 1))
    scaled_right = right * pre.reshape(1, -1)
    right_flips = 0
    for _ in range(right_batches):
        gradient_half = (right_gram @ scaled_right - right_cross) @ metric
        deltas = (
            -4 * scaled_right * gradient_half
            + 4
            * scaled_right.square()
            * right_gram.diagonal().reshape(-1, 1)
            * metric.diagonal().reshape(1, -1)
        )
        values, flat_indices = torch.topk(
            deltas.reshape(-1),
            min(right_batch_size, deltas.numel()),
            largest=False,
        )
        flat_indices = flat_indices[values < 0]
        accepted = False
        while flat_indices.numel() > 0:
            rank_indices = torch.div(flat_indices, right.shape[1], rounding_mode="floor")
            column_indices = flat_indices.remainder(right.shape[1])
            changes = -2 * scaled_right[rank_indices, column_indices]
            first = (2 * changes * gradient_half[rank_indices, column_indices]).sum()
            second = (
                right_gram[rank_indices[:, None], rank_indices[None, :]]
                * metric[column_indices[:, None], column_indices[None, :]]
                * changes.reshape(-1, 1)
                * changes.reshape(1, -1)
            ).sum()
            if float(first + second) < 0:
                right[rank_indices, column_indices] = -right[rank_indices, column_indices]
                scaled_right[rank_indices, column_indices] += changes
                right_flips += int(flat_indices.numel())
                accepted = True
                break
            flat_indices = flat_indices[: flat_indices.numel() // 2]
        if not accepted:
            break

    pre, mid, post, candidate, after_error = _fit_scales(
        target32,
        left,
        right,
        pre,
        mid,
        post,
        metric,
        output,
        protected,
        scale_passes,
        epsilon,
    )
    if not math.isfinite(after_error) or after_error > before_error:
        return CovarianceBinaryRefinement(
            torch.sign(left_binary.detach()).clone(),
            torch.sign(right_binary.detach()).clone(),
            scale_pre.detach().clone(),
            scale_mid.detach().clone(),
            scale_post.detach().clone(),
            initial.to(target.dtype),
            before_error,
            before_error,
            0,
            0,
        )
    return CovarianceBinaryRefinement(
        left.to(left_binary.dtype),
        right.to(right_binary.dtype),
        pre,
        mid,
        post,
        candidate.to(target.dtype),
        before_error,
        after_error,
        left_flips,
        right_flips,
    )
