"""Progressively fix common sign-word positions during over-complete ADMM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .factorization import SCHEDULES, ADMMResult, ADMMTracePoint
from .sign_word_codebook import _rank_one_magnitudes, _sign, _solve


@dataclass(frozen=True, slots=True)
class ProgressiveSignFixingCost:
    variable_bits: int
    scale_bits: int
    metadata_bits: int
    word_count: int

    @property
    def total(self) -> int:
        return self.variable_bits + self.scale_bits + self.metadata_bits


@dataclass(frozen=True, slots=True)
class FixedBitDecision:
    iteration: int
    position: int
    value: int
    majority_fraction: float


@dataclass(frozen=True, slots=True)
class ProgressiveSignConstraint:
    fixed_mask: torch.Tensor
    fixed_values: torch.Tensor
    decisions: tuple[FixedBitDecision, ...] = ()

    def __post_init__(self) -> None:
        if tuple(self.fixed_mask.shape) != (32,) or self.fixed_mask.dtype != torch.bool:
            raise ValueError("fixed mask must be a 32-element boolean tensor")
        if tuple(self.fixed_values.shape) != (32,):
            raise ValueError("fixed values must contain 32 signs")
        if not torch.all((self.fixed_values == 1) | (self.fixed_values == -1)):
            raise ValueError("fixed values must be signs")

    @property
    def fixed_count(self) -> int:
        return int(self.fixed_mask.sum())


@dataclass(frozen=True, slots=True)
class ProgressiveSignFixingADMMResult:
    factors: ADMMResult
    left_constraint: ProgressiveSignConstraint
    right_constraint: ProgressiveSignConstraint


def progressive_sign_fixing_bit_cost(
    out_features: int,
    in_features: int,
    rank: int,
    *,
    variable_bits_per_word: int,
    scale_width: int = 16,
    metadata_bits_per_factor: int = 64,
) -> ProgressiveSignFixingCost:
    """Charge variable payload bits, scales, and mask/template metadata."""

    if min(out_features, in_features, rank) <= 0:
        raise ValueError("progressive-fixing dimensions and rank must be positive")
    if not 0 <= variable_bits_per_word <= 32:
        raise ValueError("variable bits per word must lie in [0, 32]")
    if scale_width < 0 or metadata_bits_per_factor < 0:
        raise ValueError("storage widths must not be negative")
    words = out_features * math.ceil(rank / 32) + rank * math.ceil(in_features / 32)
    return ProgressiveSignFixingCost(
        variable_bits=words * variable_bits_per_word,
        scale_bits=scale_width * (out_features + in_features + rank),
        metadata_bits=2 * metadata_bits_per_factor,
        word_count=words,
    )


def maximum_progressive_rank_for_budget(
    out_features: int,
    in_features: int,
    target_bits: int,
    *,
    variable_bits_per_word: int,
    rank_multiple: int = 32,
    scale_width: int = 16,
) -> int:
    if target_bits <= 0 or rank_multiple <= 0:
        raise ValueError("rank budget and multiple must be positive")
    accepted = 0
    rank = rank_multiple
    while True:
        cost = progressive_sign_fixing_bit_cost(
            out_features,
            in_features,
            rank,
            variable_bits_per_word=variable_bits_per_word,
            scale_width=scale_width,
        )
        if cost.total > target_bits:
            break
        accepted = rank
        rank += rank_multiple
    if accepted <= 0:
        raise ValueError("target budget cannot fund one aligned progressive rank")
    return accepted


def empty_progressive_constraint(
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> ProgressiveSignConstraint:
    return ProgressiveSignConstraint(
        torch.zeros(32, dtype=torch.bool, device=device),
        torch.ones(32, dtype=dtype, device=device),
    )


def _word_view(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = value.shape
    words = math.ceil(columns / 32)
    padded_columns = words * 32
    padded = torch.ones(
        (rows, padded_columns),
        dtype=value.dtype,
        device=value.device,
    )
    padded[:, :columns] = value
    valid = torch.arange(padded_columns, device=value.device).reshape(words, 32) < columns
    valid = valid.unsqueeze(0).expand(rows, -1, -1)
    return padded.reshape(rows, words, 32), valid


def choose_next_fixed_bit(
    value: torch.Tensor,
    constraint: ProgressiveSignConstraint,
    iteration: int,
) -> ProgressiveSignConstraint:
    """Fix the remaining bit position with the strongest global majority."""

    if constraint.fixed_count >= 32:
        raise ValueError("all sign-word positions are already fixed")
    signs, valid = _word_view(_sign(value.float()))
    sums = (signs * valid).sum(dim=(0, 1))
    counts = valid.sum(dim=(0, 1)).clamp_min(1)
    bias = sums.abs() / counts
    bias = bias.masked_fill(constraint.fixed_mask, -1)
    position = int(bias.argmax())
    selected_sum = float(sums[position])
    selected_count = int(counts[position])
    fixed_value = 1 if selected_sum >= 0 else -1
    majority = (selected_count + abs(selected_sum)) / (2 * selected_count)
    mask = constraint.fixed_mask.clone()
    values = constraint.fixed_values.clone()
    mask[position] = True
    values[position] = fixed_value
    return ProgressiveSignConstraint(
        mask,
        values,
        constraint.decisions
        + (FixedBitDecision(iteration, position, fixed_value, majority),),
    )


def apply_progressive_constraint(
    signs: torch.Tensor,
    constraint: ProgressiveSignConstraint,
) -> torch.Tensor:
    """Overwrite globally fixed word positions and preserve all other signs."""

    words, _ = _word_view(signs)
    values = constraint.fixed_values.to(device=signs.device, dtype=signs.dtype)
    mask = constraint.fixed_mask.to(device=signs.device)
    words[:, :, mask] = values[mask]
    return words.reshape(signs.shape[0], -1)[:, : signs.shape[1]].contiguous()


def _project(
    value: torch.Tensor,
    constraint: ProgressiveSignConstraint,
    *,
    inner_iterations: int,
    generator: torch.Generator,
    epsilon: float,
) -> torch.Tensor:
    row_magnitude, column_magnitude = _rank_one_magnitudes(
        value,
        inner_iterations,
        generator,
        epsilon,
    )
    signs = apply_progressive_constraint(_sign(value), constraint)
    return torch.outer(row_magnitude, column_magnitude) * signs


def factorize_progressive_sign_fixing_admm(
    weight: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    rank: int,
    generator: torch.Generator,
    *,
    variable_bits_per_word: int,
    outer_iterations: int = 800,
    inner_iterations: int = 5,
    regularization: float = 3e-2,
    penalty_schedule: str = "cubic",
    convergence_check_interval: int = 100,
    fixing_warmup_fraction: float = 0.25,
    fixing_fraction: float = 0.5,
    epsilon: float = 1e-12,
) -> ProgressiveSignFixingADMMResult:
    """Fit factors while progressively fixing the most common word positions."""

    if weight.ndim != 2 or rank <= 0:
        raise ValueError("weight must be a matrix and rank positive")
    if input_importance.numel() != weight.shape[1] or output_importance.numel() != weight.shape[0]:
        raise ValueError("importance dimensions do not match weight")
    if not 0 <= variable_bits_per_word <= 32:
        raise ValueError("variable bits per word must lie in [0, 32]")
    if outer_iterations <= 0 or inner_iterations <= 0 or convergence_check_interval <= 0:
        raise ValueError("iteration settings must be positive")
    if not 0 <= fixing_warmup_fraction < fixing_fraction <= 1:
        raise ValueError("fixing schedule fractions are invalid")
    try:
        schedule = SCHEDULES[penalty_schedule]
    except KeyError as exc:
        raise ValueError(f"unknown penalty schedule: {penalty_schedule}") from exc

    dtype = torch.float32
    target = weight.detach().float()
    input_scale = input_importance.detach().float().sqrt().clamp_min(epsilon)
    output_scale = output_importance.detach().float().sqrt().clamp_min(epsilon).reshape(-1, 1)
    normalized = target * input_scale.reshape(1, -1) * output_scale
    left = torch.randn(
        (weight.shape[0], rank),
        dtype=dtype,
        device=weight.device,
        generator=generator,
    )
    right = torch.randn(
        (rank, weight.shape[1]),
        dtype=dtype,
        device=weight.device,
        generator=generator,
    )
    left_constraint = empty_progressive_constraint(weight.device, dtype)
    right_constraint = empty_progressive_constraint(weight.device, dtype)
    left_projected = _project(
        left,
        left_constraint,
        inner_iterations=inner_iterations,
        generator=generator,
        epsilon=epsilon,
    )
    right_projected = _project(
        right,
        right_constraint,
        inner_iterations=inner_iterations,
        generator=generator,
        epsilon=epsilon,
    )
    left_dual = left - left_projected
    right_dual = right - right_projected
    fixed_target = 32 - variable_bits_per_word
    fixing_start = math.floor(outer_iterations * fixing_warmup_fraction)
    fixing_stop = max(
        fixing_start + fixed_target,
        math.floor(outer_iterations * fixing_fraction),
    )
    trace: list[ADMMTracePoint] = []

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
            left_projected / left_norm,
            normalized,
            right_projected,
            right_dual,
            rho,
            regularization,
            epsilon,
        )
        previous_left = left_projected
        previous_right = right_projected
        fixing_progress = max(0, iteration + 1 - fixing_start)
        fixing_duration = fixing_stop - fixing_start
        desired_fixed = min(
            fixed_target,
            math.floor(fixing_progress * fixed_target / fixing_duration),
        )
        while left_constraint.fixed_count < desired_fixed:
            left_constraint = choose_next_fixed_bit(
                left + left_dual,
                left_constraint,
                iteration + 1,
            )
            right_constraint = choose_next_fixed_bit(
                right + right_dual,
                right_constraint,
                iteration + 1,
            )
        left_projected = _project(
            left + left_dual,
            left_constraint,
            inner_iterations=inner_iterations,
            generator=generator,
            epsilon=epsilon,
        )
        right_projected = _project(
            right + right_dual,
            right_constraint,
            inner_iterations=inner_iterations,
            generator=generator,
            epsilon=epsilon,
        )
        if left_constraint.fixed_count:
            # Once the feasible set becomes a strict coordinate subspace,
            # accumulated nonconvex ADMM multipliers can grow without bound
            # against globally fixed signs.  Continue as projected alternating
            # ridge solves: the constraint and penalty remain active, while no
            # stale dual is carried across progressive fixing steps.
            left_dual.zero_()
            right_dual.zero_()
        else:
            left_dual.add_(left - left_projected)
            right_dual.add_(right - right_projected)
        completed = iteration + 1
        if iteration == 0 or completed % convergence_check_interval == 0 or completed == outer_iterations:
            primal = float((left - left_projected).norm() + (right - right_projected).norm())
            dual_residual = float(
                rho * ((left_projected - previous_left).norm() + (right_projected - previous_right).norm())
            )
            trace.append(ADMMTracePoint(completed, rho, primal, dual_residual))

    if left_constraint.fixed_count != fixed_target or right_constraint.fixed_count != fixed_target:
        raise RuntimeError("progressive fixing did not reach its target")
    left_unbalanced = left_projected / output_scale
    right_unbalanced = right_projected / input_scale
    balance = (right_unbalanced.norm().clamp_min(epsilon) / left_unbalanced.norm().clamp_min(epsilon)).sqrt()
    left_export = left_unbalanced * balance
    right_export = right_unbalanced / balance
    left_latent = ((left + left_dual) / output_scale) * balance
    right_latent = ((right + right_dual) / input_scale) / balance
    scale_factor = left_projected.norm(dim=0).clamp_min(epsilon).reciprocal()
    left_export = left_export * scale_factor
    right_u, scale_pre = _rank_one_magnitudes(
        right_export,
        inner_iterations,
        generator,
        epsilon,
    )
    left_u, scale_post = _rank_one_magnitudes(
        left_export.mT,
        inner_iterations,
        generator,
        epsilon,
    )
    left_binary = apply_progressive_constraint(_sign(left_export), left_constraint)
    right_binary = apply_progressive_constraint(_sign(right_export), right_constraint)
    scale_mid = right_u * left_u
    reconstruction = (left_binary * scale_post.reshape(-1, 1)) @ (
        right_binary * scale_mid.reshape(-1, 1) * scale_pre.reshape(1, -1)
    )
    factors = ADMMResult(
        left_latent.clone().contiguous(),
        right_latent.clone().contiguous(),
        left_binary.contiguous(),
        right_binary.contiguous(),
        scale_pre.contiguous(),
        scale_mid.contiguous(),
        scale_post.contiguous(),
        reconstruction.contiguous(),
        outer_iterations,
        False,
        tuple(trace),
    )
    return ProgressiveSignFixingADMMResult(
        factors,
        left_constraint,
        right_constraint,
    )
