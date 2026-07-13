"""Deterministic, side-effect-free NanoQuant ADMM factorization."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ADMMTracePoint:
    iteration: int
    rho: float
    primal_residual: float
    dual_residual: float


@dataclass(frozen=True, slots=True)
class ADMMResult:
    left_latent: torch.Tensor
    right_latent: torch.Tensor
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    reconstruction: torch.Tensor
    iterations_completed: int
    stopped_early: bool
    trace: tuple[ADMMTracePoint, ...]


def cubic_schedule(progress: float) -> float:
    return min(1.0, max(0.0, progress)) ** 3


def linear_schedule(progress: float) -> float:
    return min(1.0, max(0.0, progress))


def logistic_schedule(progress: float, steepness: float = 5.0) -> float:
    progress = min(1.0, max(0.0, progress))
    return 1.0 / (1.0 + math.exp(-steepness * (progress - 0.5)))


def exponential_schedule(progress: float, steepness: float = 5.0) -> float:
    progress = min(1.0, max(0.0, progress))
    return math.expm1(steepness * progress) / math.expm1(steepness)


SCHEDULES: dict[str, Callable[[float], float]] = {
    "cubic": cubic_schedule,
    "linear": linear_schedule,
    "logistic": logistic_schedule,
    "exp_growth": exponential_schedule,
}


def _sign(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))


def _power_iteration(
    value: torch.Tensor, iterations: int, generator: torch.Generator, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vector = torch.randn(value.shape[1], dtype=value.dtype, device=value.device, generator=generator)
    vector = vector / vector.norm().clamp_min(epsilon)
    for _ in range(iterations):
        left = value @ vector
        left = left / left.norm().clamp_min(epsilon)
        vector = value.mT @ left
        vector = vector / vector.norm().clamp_min(epsilon)
    unnormalized = value @ vector
    singular = unnormalized.norm().clamp_min(epsilon)
    return unnormalized / singular, singular, vector


def _rank_one_sign_projection(
    value: torch.Tensor, iterations: int, generator: torch.Generator, epsilon: float
) -> torch.Tensor:
    left, right, signs = _svid_components(value, iterations, generator, epsilon)
    return torch.outer(left, right) * signs


def _svid_components(
    value: torch.Tensor, iterations: int, generator: torch.Generator, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signs = _sign(value)
    left, singular, right = _power_iteration(value.abs(), iterations, generator, epsilon)
    return left * singular, right, signs


def _solve(
    design: torch.Tensor,
    target: torch.Tensor,
    projected: torch.Tensor,
    dual: torch.Tensor,
    rho: float,
    regularization: float,
    epsilon: float,
) -> torch.Tensor:
    design32 = design.float()
    system = design32.mT @ design32
    system = 0.5 * (system + system.mT)
    stabilizer = (rho * system.diagonal().mean().abs() + regularization).clamp_min(epsilon)
    system.diagonal().add_(stabilizer)
    rhs = design32.mT @ target.float()
    # rhs is float32, so add_ promotes the bf16 factor inputs exactly while
    # avoiding two standalone conversion kernels and their temporary tensors.
    rhs.add_(projected, alpha=rho)
    rhs.add_(dual, alpha=-rho)
    factor, info = torch.linalg.cholesky_ex(system)
    solution = torch.cholesky_solve(rhs, factor) if int(info.max()) == 0 else torch.linalg.solve(system, rhs)
    return solution.to(design.dtype)


def factorize_admm(
    weight: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    rank: int,
    generator: torch.Generator,
    *,
    outer_iterations: int = 400,
    inner_iterations: int = 5,
    regularization: float = 3e-2,
    penalty_schedule: str = "cubic",
    convergence_check_interval: int = 100,
    early_stop_tolerance: float | None = None,
    epsilon: float = 1e-12,
) -> ADMMResult:
    if weight.ndim != 2 or rank <= 0 or rank > min(weight.shape):
        raise ValueError("weight must be a matrix and rank within its dimensions")
    if input_importance.numel() != weight.shape[1] or output_importance.numel() != weight.shape[0]:
        raise ValueError("importance dimensions do not match weight")
    if outer_iterations < 0 or inner_iterations <= 0 or convergence_check_interval <= 0:
        raise ValueError("iteration settings are invalid")
    try:
        schedule = SCHEDULES[penalty_schedule]
    except KeyError as exc:
        raise ValueError(f"unknown penalty schedule: {penalty_schedule}") from exc
    dtype = weight.dtype if weight.is_floating_point() else torch.float32
    target = weight.detach().to(dtype=dtype)
    input_scale = input_importance.detach().float().sqrt().clamp_min(epsilon)
    output_scale = output_importance.detach().float().sqrt().clamp_min(epsilon).reshape(-1, 1)
    normalized = target * input_scale.reshape(1, -1) * output_scale
    left = torch.randn((weight.shape[0], rank), dtype=dtype, device=weight.device, generator=generator)
    right = torch.randn((rank, weight.shape[1]), dtype=dtype, device=weight.device, generator=generator)
    left_projected = _rank_one_sign_projection(left, inner_iterations, generator, epsilon)
    right_projected = _rank_one_sign_projection(right, inner_iterations, generator, epsilon)
    left_dual = left - left_projected
    right_dual = right - right_projected
    trace: list[ADMMTracePoint] = []
    stopped = False
    completed = 0
    for iteration in range(outer_iterations):
        rho = schedule(iteration / max(1, outer_iterations))
        right_norm = right_projected.norm(dim=1).clamp_min(epsilon)
        left = _solve(
            right_projected.mT / right_norm,
            normalized.mT,
            left_projected.mT,
            left_dual.mT,
            rho,
            regularization,
            epsilon,
        ).mT
        left_norm = left_projected.norm(dim=0).clamp_min(epsilon)
        right = _solve(
            left_projected / left_norm, normalized, right_projected, right_dual, rho, regularization, epsilon
        )
        previous_left = left_projected
        previous_right = right_projected
        left_projected = _rank_one_sign_projection(left + left_dual, inner_iterations, generator, epsilon)
        right_projected = _rank_one_sign_projection(right + right_dual, inner_iterations, generator, epsilon)
        left_dual.add_(left - left_projected)
        right_dual.add_(right - right_projected)
        completed = iteration + 1
        if iteration == 0 or completed % convergence_check_interval == 0 or completed == outer_iterations:
            primal = float((left - left_projected).norm() + (right - right_projected).norm())
            dual = float(rho * ((left_projected - previous_left).norm() + (right_projected - previous_right).norm()))
            trace.append(ADMMTracePoint(completed, rho, primal, dual))
            if early_stop_tolerance is not None and primal <= early_stop_tolerance and dual <= early_stop_tolerance:
                stopped = True
                break
    left_unbalanced = left_projected / output_scale
    right_unbalanced = right_projected / input_scale
    balance = (right_unbalanced.norm().clamp_min(epsilon) / left_unbalanced.norm().clamp_min(epsilon)).sqrt()
    left_export = left_unbalanced * balance
    right_export = right_unbalanced / balance
    # Legacy NanoQuant tunes the primal-plus-dual variables, while export is
    # derived from the SVID-projected variables. Their signs agree initially,
    # but their margins to zero carry essential STE optimization state.
    left_latent = ((left + left_dual) / output_scale) * balance
    right_latent = ((right + right_dual) / input_scale) / balance
    scale_factor = left_projected.norm(dim=0).clamp_min(epsilon).reciprocal()
    left_export = left_export * scale_factor
    right_u, scale_pre, right_binary = _svid_components(
        right_export.float(), inner_iterations, generator, epsilon
    )
    left_u, scale_post, left_sign = _svid_components(
        left_export.mT.float(), inner_iterations, generator, epsilon
    )
    left_binary = left_sign.mT.to(dtype).clone(memory_format=torch.contiguous_format)
    right_binary = right_binary.to(dtype).contiguous()
    scale_pre = scale_pre.to(dtype)
    scale_mid = (right_u * left_u).to(dtype)
    scale_post = scale_post.to(dtype)
    reconstruction = (left_binary * scale_post.reshape(-1, 1)) @ (
        right_binary * scale_mid.reshape(-1, 1) * scale_pre.reshape(1, -1)
    )
    return ADMMResult(
        left_latent.clone().contiguous(),
        right_latent.clone().contiguous(),
        left_binary.contiguous(),
        right_binary.contiguous(),
        scale_pre.contiguous(),
        scale_mid.contiguous(),
        scale_post.contiguous(),
        reconstruction.contiguous(),
        completed,
        stopped,
        tuple(trace),
    )
