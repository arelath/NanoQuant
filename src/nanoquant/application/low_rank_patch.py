"""Closed-form activation-weighted low-rank residual patches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FittedLowRankPatch:
    rank: int
    left: torch.Tensor
    right: torch.Tensor
    fit_error_before: float
    fit_error_after: float
    held_out_error_before: float
    held_out_error_after: float
    accepted: bool
    rejection_reason: str | None


def activation_error(
    residual: torch.Tensor,
    covariance: torch.Tensor,
    input_mean: torch.Tensor,
    bias: torch.Tensor | None,
) -> float:
    """Measure expected squared output error from uncentered input moments."""

    value = ((residual @ covariance) * residual).sum()
    if bias is not None:
        output_mean = residual @ input_mean
        value = value - 2 * torch.dot(bias, output_mean) + bias.square().sum()
    return max(0.0, float(value))


def fit_low_rank_patch_family(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    fit_covariance: torch.Tensor,
    held_out_covariance: torch.Tensor,
    fit_input_mean: torch.Tensor,
    held_out_input_mean: torch.Tensor,
    *,
    ranks: tuple[int, ...],
    ridge_fraction: float,
    storage_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    require_held_out_acceptance: bool = True,
) -> tuple[FittedLowRankPatch, ...]:
    """Fit several prefix ranks from one activation-weighted residual SVD."""

    if not ranks or tuple(sorted(set(ranks))) != ranks or ranks[0] <= 0:
        raise ValueError("low-rank patch ranks must be unique, positive, and increasing")
    if target.ndim != 2 or reconstruction.shape != target.shape:
        raise ValueError("low-rank patch target and reconstruction must be equally shaped matrices")
    out_features, in_features = target.shape
    expected_covariance = (in_features, in_features)
    if fit_covariance.shape != expected_covariance or held_out_covariance.shape != expected_covariance:
        raise ValueError("low-rank patch covariance shape differs from the input dimension")
    if fit_input_mean.shape != (in_features,) or held_out_input_mean.shape != (in_features,):
        raise ValueError("low-rank patch input mean shape differs from the input dimension")
    if bias is not None and bias.shape != (out_features,):
        raise ValueError("low-rank patch bias shape differs from the output dimension")
    if ridge_fraction < 0:
        raise ValueError("low-rank patch ridge fraction must not be negative")
    if storage_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("low-rank patch storage dtype must be float16 or bfloat16")

    residual = target.float() - reconstruction.float()
    covariance = fit_covariance.float()
    ridge = ridge_fraction * float(torch.trace(covariance)) / in_features
    damped = covariance + torch.eye(
        in_features,
        dtype=covariance.dtype,
        device=covariance.device,
    ) * ridge
    cholesky = torch.linalg.cholesky(damped)
    transformed = cholesky.mT @ residual.mT
    singular_left, singular_values, singular_right = torch.linalg.svd(
        transformed,
        full_matrices=False,
    )
    fit_before = activation_error(
        residual,
        covariance,
        fit_input_mean.float(),
        None if bias is None else bias.float(),
    )
    held_before = activation_error(
        residual,
        held_out_covariance.float(),
        held_out_input_mean.float(),
        None if bias is None else bias.float(),
    )

    results: list[FittedLowRankPatch] = []
    for requested_rank in ranks:
        rank = min(requested_rank, singular_values.numel())
        patch_left = singular_right[:rank].mT * singular_values[:rank]
        solved = torch.linalg.solve_triangular(
            cholesky.mT,
            singular_left[:, :rank],
            upper=True,
        )
        patch_right = solved.mT
        stored_left = patch_left.to(storage_dtype)
        stored_right = patch_right.to(storage_dtype)
        stored_patch = stored_left.float() @ stored_right.float()
        fit_after = activation_error(
            residual - stored_patch,
            covariance,
            fit_input_mean.float(),
            None if bias is None else bias.float(),
        )
        held_after = activation_error(
            residual - stored_patch,
            held_out_covariance.float(),
            held_out_input_mean.float(),
            None if bias is None else bias.float(),
        )
        accepted = fit_after < fit_before and (
            not require_held_out_acceptance or held_after < held_before
        )
        results.append(
            FittedLowRankPatch(
                rank,
                stored_left.detach().cpu().contiguous(),
                stored_right.detach().cpu().contiguous(),
                fit_before,
                fit_after,
                held_before,
                held_after,
                accepted,
                None
                if accepted
                else (
                    "held_out_error_did_not_improve"
                    if held_after >= held_before
                    else "fit_error_did_not_improve"
                ),
            )
        )
    return tuple(results)


__all__ = [
    "FittedLowRankPatch",
    "activation_error",
    "fit_low_rank_patch_family",
]
