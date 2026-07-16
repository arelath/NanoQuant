"""Pure weighted fitting for additive binary rank expansion."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ResidualMiddleFitResult:
    """Best additive middle scales for fixed binary factors and outer scales."""

    scale_mid: torch.Tensor
    correction: torch.Tensor
    before_error: float
    after_error: float
    accepted: bool


def fit_residual_middle_scales(
    residual: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    protected_columns: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> ResidualMiddleFitResult:
    """Fit only new rank coefficients, preserving the existing reconstruction exactly.

    The added correction has the standard NanoQuant form
    ``diag(post) @ left @ diag(mid) @ right @ diag(pre)``. Existing factors,
    scales, outliers, and bias are not inputs and therefore cannot be changed.
    A zero middle-scale vector is always feasible, so a finite regressing solve
    is rolled back to the exact no-op correction.
    """

    if residual.ndim != 2 or left_binary.ndim != 2 or right_binary.ndim != 2:
        raise ValueError("rank expansion tensors must be matrices")
    if left_binary.shape[0] != residual.shape[0] or right_binary.shape[1] != residual.shape[1]:
        raise ValueError("rank expansion factor dimensions differ from the residual")
    if left_binary.shape[1] != right_binary.shape[0] or left_binary.shape[1] == 0:
        raise ValueError("rank expansion factors must have one positive shared rank")
    if scale_pre.numel() != residual.shape[1] or scale_post.numel() != residual.shape[0]:
        raise ValueError("rank expansion outer scales differ from the residual")
    if input_importance.numel() != residual.shape[1] or output_importance.numel() != residual.shape[0]:
        raise ValueError("rank expansion importance dimensions differ from the residual")
    if epsilon <= 0:
        raise ValueError("rank expansion epsilon must be positive")

    target = residual.detach().float()
    left = torch.sign(left_binary.detach().float())
    right = torch.sign(right_binary.detach().float())
    pre = scale_pre.detach().float().reshape(-1).clone()
    post = scale_post.detach().float().reshape(-1)
    if protected_columns is not None:
        indexes = protected_columns.detach().long().reshape(-1)
        if indexes.numel() and (int(indexes.min()) < 0 or int(indexes.max()) >= pre.numel()):
            raise ValueError("rank expansion protected column is outside the input dimension")
        pre.index_fill_(0, indexes, 0)
    input_weight = input_importance.detach().float().reshape(-1).clamp_min(epsilon)
    output_weight = output_importance.detach().float().reshape(-1).clamp_min(epsilon)

    scaled_left = left * post[:, None]
    scaled_right = right * pre[None, :]
    left_gram = scaled_left.mT @ (scaled_left * output_weight[:, None])
    weighted_right = scaled_right * input_weight.sqrt()[None, :]
    right_gram = weighted_right @ weighted_right.mT
    system = left_gram * right_gram
    system = 0.5 * (system + system.mT)
    cross = scaled_left.mT @ (target * input_weight[None, :] * output_weight[:, None])
    rhs = (cross * scaled_right).sum(dim=1)
    ridge = torch.clamp(system.diagonal().mean().abs() * 1e-6, min=epsilon)
    system.diagonal().add_(ridge)
    cholesky, info = torch.linalg.cholesky_ex(system, upper=False)
    middle = (
        torch.cholesky_solve(rhs[:, None], cholesky, upper=False).squeeze(1)
        if int(info.item()) == 0
        else torch.linalg.lstsq(system, rhs[:, None]).solution.squeeze(1)
    )
    middle = torch.nan_to_num(middle)
    correction = scaled_left @ (scaled_right * middle[:, None])
    weighted = input_weight[None, :] * output_weight[:, None]
    before = float((target.square() * weighted).sum())
    after = float(((target - correction).square() * weighted).sum())
    finite = bool(torch.isfinite(middle).all() and torch.isfinite(correction).all())
    if not finite or after > before:
        middle = torch.zeros_like(middle)
        correction = torch.zeros_like(target)
        after = before
        accepted = False
    else:
        accepted = True
    return ResidualMiddleFitResult(middle, correction, before, after, accepted)


__all__ = ["ResidualMiddleFitResult", "fit_residual_middle_scales"]
