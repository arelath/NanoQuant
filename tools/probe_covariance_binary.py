"""Probe covariance-aware refinement of the existing binary-factor format.

The production format reconstructs a weight as

    diag(post) @ left_binary @ diag(mid) @ right_binary @ diag(pre)

This analysis-only probe keeps that representation, rank, and factor bits
unchanged.  It starts from the ordinary diagonal-objective ADMM result, then
alternates exact dense-covariance scale solves with bounded sign-coordinate
descent.  The retained format therefore needs no whitening matrix or runtime
transform.

The full pinned-model orchestration is intentionally added only after the
small-matrix objective and sign-delta identities pass focused tests.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_covariance_headroom import (
    _capture_covariances,
    _group_importance,
    _input_capture_specs,
    _materialize_group_weight,
)
from probe_factor_grouping import (
    PINNED_MODEL_REVISION,
    PROJECTION_PATHS,
    GroupSpec,
    MemberSpec,
    ProbeProtocol,
    _logical_seed,
    _planned_group_rank,
    group_shape,
    load_calibration_profiles,
)
from probe_importance_shrinkage import (
    _capture_outputs,
    _dtype,
    _isolated_block_outputs,
    _paired_summary,
    _parse_ints,
)
from probe_input_hadamard import (
    BASELINE_KEY,
    TRANSFORMED_GROUPS,
    _aggregate_groups,
    _covariance_key,
    _evaluate_prediction,
    _member_reconstructions,
    block_groups,
)
from safetensors import safe_open

from nanoquant.config.codec import to_dict
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.metrics import dense_hessian_squared_error
from nanoquant.domain.models import BitCost, BlockId, LayerId
from nanoquant.domain.objectives import regularize_covariance
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.scale_fit import fit_scales, reconstruct
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

MODEL_SOURCE = "google/gemma-3-1b-it"
CANDIDATE_KEY = "covariance-refined"


@dataclass(frozen=True, slots=True)
class CovarianceScaleFitResult:
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    reconstruction: torch.Tensor
    before_error: float
    after_error: float
    accepted: bool
    rollback_reason: str | None


@dataclass(frozen=True, slots=True)
class SignRefinementResult:
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    reconstruction: torch.Tensor
    before_error: float
    after_error: float
    left_flips: int
    right_flips: int


def _validate_factor_shapes(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
) -> None:
    if target.ndim != 2 or left.ndim != 2 or right.ndim != 2:
        raise ValueError("target and binary factors must be matrices")
    if left.shape[0] != target.shape[0] or right.shape[1] != target.shape[1]:
        raise ValueError("binary factor dimensions do not match target")
    if left.shape[1] != right.shape[0]:
        raise ValueError("binary factor ranks do not match")
    if pre.numel() != target.shape[1] or mid.numel() != left.shape[1] or post.numel() != target.shape[0]:
        raise ValueError("scale dimensions do not match factors")
    if covariance.shape != (target.shape[1], target.shape[1]):
        raise ValueError("covariance dimensions do not match target")
    if output_importance.numel() != target.shape[0]:
        raise ValueError("output importance dimensions do not match target")


def _error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
) -> float:
    return float(
        dense_hessian_squared_error(
            target,
            prediction,
            covariance,
            output_importance,
        )
    )


def _stable_psd_solve(
    system: torch.Tensor,
    rhs: torch.Tensor,
    *,
    epsilon: float,
    ridge_fraction: float,
) -> torch.Tensor:
    value = 0.5 * (system.float() + system.float().mT)
    ridge = (value.diagonal().mean().abs() * ridge_fraction).clamp_min(epsilon)
    value.diagonal().add_(ridge)
    factor, info = torch.linalg.cholesky_ex(value)
    if int(info.max()) == 0:
        solution = torch.cholesky_solve(rhs.float().reshape(-1, 1), factor).reshape(-1)
    else:
        solution = torch.linalg.lstsq(value, rhs.float().reshape(-1, 1)).solution.reshape(-1)
    return torch.nan_to_num(solution)


def _fit_post(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    covariance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    base = (left * mid.reshape(1, -1)) @ (right * pre.reshape(1, -1))
    base_covariance = base @ covariance
    numerator = (base_covariance * target).sum(dim=1)
    denominator = (base_covariance * base).sum(dim=1).clamp_min(epsilon)
    return torch.nan_to_num(numerator / denominator)


def _fit_pre(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    epsilon: float,
    ridge_fraction: float,
) -> torch.Tensor:
    weighted_left = left * (post.reshape(-1, 1) * mid.reshape(1, -1))
    base = weighted_left @ right
    base_gram = base.mT @ (base * output_importance.reshape(-1, 1))
    system = base_gram * covariance
    target_covariance = (target * output_importance.reshape(-1, 1)) @ covariance
    rhs = (base * target_covariance).sum(dim=0)
    return _stable_psd_solve(
        system,
        rhs,
        epsilon=epsilon,
        ridge_fraction=ridge_fraction,
    )


def _fit_mid(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    epsilon: float,
    ridge_fraction: float,
) -> torch.Tensor:
    scaled_left = left * post.reshape(-1, 1)
    scaled_right = right * pre.reshape(1, -1)
    left_gram = scaled_left.mT @ (scaled_left * output_importance.reshape(-1, 1))
    right_covariance = scaled_right @ covariance
    right_gram = right_covariance @ scaled_right.mT
    system = left_gram * right_gram
    target_covariance = (target * output_importance.reshape(-1, 1)) @ covariance
    cross = scaled_left.mT @ target_covariance
    rhs = (cross * scaled_right).sum(dim=1)
    return _stable_psd_solve(
        system,
        rhs,
        epsilon=epsilon,
        ridge_fraction=ridge_fraction,
    )


def fit_covariance_scales(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    alternating_passes: int = 2,
    epsilon: float = 1e-8,
    ridge_fraction: float = 1e-6,
    rollback_on_regression: bool = True,
) -> CovarianceScaleFitResult:
    """Fit all three scale vectors under a dense input covariance."""

    if alternating_passes < 0 or epsilon <= 0 or ridge_fraction < 0:
        raise ValueError("covariance scale-fit settings are invalid")
    _validate_factor_shapes(
        target,
        left_binary,
        right_binary,
        scale_pre,
        scale_mid,
        scale_post,
        covariance,
        output_importance,
    )
    target32 = target.detach().float()
    left = torch.sign(left_binary.detach().float())
    right = torch.sign(right_binary.detach().float())
    pre = scale_pre.detach().float().reshape(-1).clone()
    mid = scale_mid.detach().float().reshape(-1).clone()
    post = scale_post.detach().float().reshape(-1).clone()
    metric = 0.5 * (covariance.detach().float() + covariance.detach().float().mT)
    output = output_importance.detach().float().reshape(-1).clamp_min(epsilon)
    original = reconstruct(left, right, pre, mid, post)
    before_error = _error(target32, original, metric, output)
    best_error = before_error
    best = (pre.clone(), mid.clone(), post.clone(), original)
    for _ in range(alternating_passes):
        post = _fit_post(target32, left, right, pre, mid, metric, epsilon)
        pre = _fit_pre(
            target32,
            left,
            right,
            mid,
            post,
            metric,
            output,
            epsilon=epsilon,
            ridge_fraction=ridge_fraction,
        )
        mid = _fit_mid(
            target32,
            left,
            right,
            pre,
            post,
            metric,
            output,
            epsilon=epsilon,
            ridge_fraction=ridge_fraction,
        )
        candidate = reconstruct(left, right, pre, mid, post)
        candidate_error = _error(target32, candidate, metric, output)
        if math.isfinite(candidate_error) and candidate_error < best_error:
            best_error = candidate_error
            best = (pre.clone(), mid.clone(), post.clone(), candidate)
    best_pre, best_mid, best_post, candidate = best
    finite = bool(torch.isfinite(candidate).all()) and math.isfinite(best_error)
    if not finite or (rollback_on_regression and best_error > before_error):
        reason = "non_finite_candidate" if not finite else "covariance_objective_regressed"
        return CovarianceScaleFitResult(
            scale_pre.detach().clone(),
            scale_mid.detach().clone(),
            scale_post.detach().clone(),
            original.to(target.dtype),
            before_error,
            before_error,
            False,
            reason,
        )
    return CovarianceScaleFitResult(
        best_pre,
        best_mid,
        best_post,
        candidate.to(target.dtype),
        before_error,
        best_error,
        True,
        None,
    )


def left_flip_deltas(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    """Return the exact per-bit objective delta with every other bit fixed."""

    left = torch.sign(left_binary.float())
    right = torch.sign(right_binary.float())
    pre = scale_pre.float().reshape(-1)
    mid = scale_mid.float().reshape(-1)
    post = scale_post.float().reshape(-1)
    scaled_right = right * (mid.reshape(-1, 1) * pre.reshape(1, -1))
    right_covariance = scaled_right @ covariance.float()
    gram = right_covariance @ scaled_right.mT
    cross = (target.float() @ covariance.float()) @ scaled_right.mT
    coefficients = left * post.reshape(-1, 1)
    state = coefficients @ gram - cross
    return (
        -4 * post.reshape(-1, 1) * left * state
        + 4 * post.square().reshape(-1, 1) * gram.diagonal().reshape(1, -1)
    )


def right_flip_deltas(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
) -> torch.Tensor:
    """Return exact individual right-bit deltas with every other bit fixed."""

    left = torch.sign(left_binary.float())
    right = torch.sign(right_binary.float())
    pre = scale_pre.float().reshape(-1)
    mid = scale_mid.float().reshape(-1)
    post = scale_post.float().reshape(-1)
    scaled_left = left * (post.reshape(-1, 1) * mid.reshape(1, -1))
    output = output_importance.float().reshape(-1)
    gram = scaled_left.mT @ (scaled_left * output.reshape(-1, 1))
    cross = scaled_left.mT @ (target.float() * output.reshape(-1, 1))
    scaled_right = right * pre.reshape(1, -1)
    gradient_half = (gram @ scaled_right - cross) @ covariance.float()
    return (
        -4 * scaled_right * gradient_half
        + 4
        * scaled_right.square()
        * gram.diagonal().reshape(-1, 1)
        * covariance.float().diagonal().reshape(1, -1)
    )


def refine_covariance_signs(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    left_steps: int = 8,
    right_batches: int = 8,
    right_batch_size: int = 128,
    improvement_tolerance: float = 0.0,
) -> SignRefinementResult:
    """Run bounded exact left-coordinate and accepted right-batch sign descent."""

    if left_steps < 0 or right_batches < 0 or right_batch_size <= 0 or improvement_tolerance < 0:
        raise ValueError("sign-refinement settings are invalid")
    _validate_factor_shapes(
        target,
        left_binary,
        right_binary,
        scale_pre,
        scale_mid,
        scale_post,
        covariance,
        output_importance,
    )
    target32 = target.detach().float()
    left = torch.sign(left_binary.detach().float()).clone()
    right = torch.sign(right_binary.detach().float()).clone()
    pre = scale_pre.detach().float().reshape(-1)
    mid = scale_mid.detach().float().reshape(-1)
    post = scale_post.detach().float().reshape(-1)
    metric = 0.5 * (covariance.detach().float() + covariance.detach().float().mT)
    output = output_importance.detach().float().reshape(-1).clamp_min(1e-12)
    initial = reconstruct(left, right, pre, mid, post)
    before_error = _error(target32, initial, metric, output)

    scaled_right = right * (mid.reshape(-1, 1) * pre.reshape(1, -1))
    right_covariance = scaled_right @ metric
    left_gram = right_covariance @ scaled_right.mT
    left_cross = (target32 @ metric) @ scaled_right.mT
    coefficients = left * post.reshape(-1, 1)
    left_state = coefficients @ left_gram - left_cross
    row_indices = torch.arange(left.shape[0], device=left.device)
    left_flips = 0
    for _ in range(left_steps):
        deltas = (
            -4 * post.reshape(-1, 1) * left * left_state
            + 4 * post.square().reshape(-1, 1) * left_gram.diagonal().reshape(1, -1)
        )
        best_delta, indices = deltas.min(dim=1)
        active = best_delta < -improvement_tolerance
        if not bool(active.any()):
            break
        active_rows = row_indices[active]
        active_indices = indices[active]
        delta_coefficient = -2 * coefficients[active_rows, active_indices]
        left[active_rows, active_indices] = -left[active_rows, active_indices]
        coefficients[active_rows, active_indices] = (
            coefficients[active_rows, active_indices] + delta_coefficient
        )
        left_state[active_rows] = left_state[active_rows] + (
            left_gram.index_select(0, active_indices) * delta_coefficient.reshape(-1, 1)
        )
        left_flips += int(active.sum())

    scaled_left = left * (post.reshape(-1, 1) * mid.reshape(1, -1))
    right_gram = scaled_left.mT @ (scaled_left * output.reshape(-1, 1))
    right_cross = scaled_left.mT @ (target32 * output.reshape(-1, 1))
    scaled_right = right * pre.reshape(1, -1)
    right_flips = 0
    for _ in range(right_batches):
        gradient_half = (right_gram @ scaled_right - right_cross) @ metric
        individual = (
            -4 * scaled_right * gradient_half
            + 4
            * scaled_right.square()
            * right_gram.diagonal().reshape(-1, 1)
            * metric.diagonal().reshape(1, -1)
        )
        flat = individual.reshape(-1)
        count = min(right_batch_size, flat.numel())
        values, flat_indices = torch.topk(flat, count, largest=False)
        eligible = values < -improvement_tolerance
        flat_indices = flat_indices[eligible]
        if flat_indices.numel() == 0:
            break
        accepted = False
        while flat_indices.numel() > 0:
            rank_indices = torch.div(flat_indices, right.shape[1], rounding_mode="floor")
            column_indices = flat_indices.remainder(right.shape[1])
            changes = -2 * scaled_right[rank_indices, column_indices]
            first_order = (
                2 * changes * gradient_half[rank_indices, column_indices]
            ).sum()
            gram_pairs = right_gram[rank_indices[:, None], rank_indices[None, :]]
            covariance_pairs = metric[column_indices[:, None], column_indices[None, :]]
            second_order = (
                gram_pairs
                * covariance_pairs
                * changes.reshape(-1, 1)
                * changes.reshape(1, -1)
            ).sum()
            exact_delta = float(first_order + second_order)
            if math.isfinite(exact_delta) and exact_delta < -improvement_tolerance:
                right[rank_indices, column_indices] = -right[rank_indices, column_indices]
                scaled_right[rank_indices, column_indices] = (
                    scaled_right[rank_indices, column_indices] + changes
                )
                right_flips += int(flat_indices.numel())
                accepted = True
                break
            flat_indices = flat_indices[: flat_indices.numel() // 2]
        if not accepted:
            break

    candidate = reconstruct(left, right, pre, mid, post)
    after_error = _error(target32, candidate, metric, output)
    if not math.isfinite(after_error) or after_error > before_error:
        return SignRefinementResult(
            torch.sign(left_binary.detach()).clone(),
            torch.sign(right_binary.detach()).clone(),
            initial.to(target.dtype),
            before_error,
            before_error,
            0,
            0,
        )
    return SignRefinementResult(
        left.to(left_binary.dtype),
        right.to(right_binary.dtype),
        candidate.to(target.dtype),
        before_error,
        after_error,
        left_flips,
        right_flips,
    )


def _factorization_parameters(protocol: ProbeProtocol) -> AdmmParameters:
    return AdmmParameters(
        outer_iterations=protocol.outer_iterations,
        inner_iterations=protocol.inner_iterations,
        regularization=protocol.regularization,
        penalty_schedule=protocol.penalty_schedule,
        convergence_check_interval=protocol.convergence_check_interval,
        transpose_wide=protocol.transpose_wide,
    )


def _base_result_payload(
    handle: Any,
    group: GroupSpec,
    protocol: ProbeProtocol,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_importance: torch.Tensor,
    diagonal_input: torch.Tensor,
    fit_covariance: torch.Tensor | None,
    held_out_covariance: torch.Tensor | None,
    factorization: dict[str, Any],
) -> tuple[dict[str, Any], tuple[tuple[MemberSpec, torch.Tensor, float], ...]]:
    out_features, in_features, source_elements = group_shape(handle, group)
    member_rows = tuple(
        int(handle.get_slice(member.tensor_name).get_shape()[1 if member.transpose else 0])
        for member in group.members
    )
    rank, extra_scale_bits = _planned_group_rank(handle, group, protocol, profiles)
    cost = factor_bit_cost(
        out_features,
        in_features,
        rank,
        scale_bits=protocol.scale_bits,
        rank_alignment=protocol.rank_alignment,
    ) + BitCost(scale_bits=extra_scale_bits)
    evaluation = _evaluate_prediction(
        target,
        prediction,
        output_importance,
        fit_covariance,
        held_out_covariance,
        diagonal_input,
    )
    result = {
        "block": group.members[0].block,
        "group": group.label,
        "members": [member.label for member in group.members],
        "shape": [out_features, in_features],
        "rank": rank,
        "source_elements": source_elements,
        "target_bits": math.floor(source_elements * protocol.target_bpw),
        "bit_cost": asdict(cost),
        "actual_bpw": cost.total / source_elements,
        "transformed": False,
        "factorization": factorization,
        "evaluation": evaluation,
    }
    return result, _member_reconstructions(group, prediction, member_rows)


def _group_pair(
    handle: Any,
    group: GroupSpec,
    protocol: ProbeProtocol,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]],
    fit_covariance: torch.Tensor | None,
    held_out_covariance: torch.Tensor | None,
    *,
    damp_fraction: float,
    covariance_diagonal_blend: float,
    covariance_scale_passes: int,
    left_flip_steps: int,
    right_flip_batches: int,
    right_flip_batch_size: int,
) -> tuple[
    tuple[dict[str, Any], tuple[tuple[MemberSpec, torch.Tensor, float], ...]],
    tuple[dict[str, Any], tuple[tuple[MemberSpec, torch.Tensor, float], ...]],
]:
    target = _materialize_group_weight(handle, group).to(
        device=protocol.device,
        dtype=torch.bfloat16,
    )
    raw_input, output_importance = _group_importance(group, profiles)
    raw_input = raw_input.to(protocol.device).float()
    output_importance = output_importance.to(protocol.device).float()
    regularized = (
        None
        if fit_covariance is None
        else regularize_covariance(
            fit_covariance.to(protocol.device),
            damp_fraction=damp_fraction,
            diagonal_blend=covariance_diagonal_blend,
        )
    )
    diagonal = raw_input if regularized is None else regularized.diagonal().clone()
    rank, _extra_scale_bits = _planned_group_rank(handle, group, protocol, profiles)
    generator = torch.Generator(device=protocol.device).manual_seed(
        _logical_seed(protocol.seed, f"{group.members[0].block}:{group.label}")
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(protocol.device)
        torch.cuda.synchronize(protocol.device)
    baseline_started = time.perf_counter()
    factorized = factorize_admm_with_parameters(
        target,
        diagonal,
        output_importance,
        rank,
        generator,
        _factorization_parameters(protocol),
    )
    fitted = fit_scales(
        target,
        factorized.left_binary,
        factorized.right_binary,
        factorized.scale_pre,
        factorized.scale_mid,
        factorized.scale_post,
        diagonal,
        output_importance,
        alternating_passes=protocol.scale_fit_passes,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.synchronize(protocol.device)
    baseline_wall = time.perf_counter() - baseline_started
    baseline_metadata = {
        "iterations_completed": factorized.iterations_completed,
        "scale_fit_accepted": fitted.accepted,
        "scale_fit_rollback_reason": fitted.rollback_reason,
        "factor_objective_error": fitted.after_error,
        "factor_objective_target": float(
            ((target.float().square()) * diagonal.reshape(1, -1) * output_importance.reshape(-1, 1)).sum()
        ),
        "factor_objective_normalized_rmse": math.sqrt(
            fitted.after_error
            / max(
                float(
                    (
                        target.float().square()
                        * diagonal.reshape(1, -1)
                        * output_importance.reshape(-1, 1)
                    ).sum()
                ),
                1e-30,
            )
        ),
        "wall_seconds": baseline_wall,
        "peak_device_bytes": (
            int(torch.cuda.max_memory_allocated(protocol.device))
            if protocol.device.startswith("cuda")
            else 0
        ),
    }
    baseline = _base_result_payload(
        handle,
        group,
        protocol,
        profiles,
        target,
        fitted.reconstruction,
        output_importance,
        diagonal,
        None if fit_covariance is None else fit_covariance.to(protocol.device),
        None if held_out_covariance is None else held_out_covariance.to(protocol.device),
        baseline_metadata,
    )
    if regularized is None:
        del target, raw_input, output_importance, diagonal, factorized, fitted
        return baseline, baseline

    if protocol.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(protocol.device)
        torch.cuda.synchronize(protocol.device)
    candidate_started = time.perf_counter()
    first_scales = fit_covariance_scales(
        target,
        factorized.left_binary,
        factorized.right_binary,
        fitted.scale_pre,
        fitted.scale_mid,
        fitted.scale_post,
        regularized,
        output_importance,
        alternating_passes=covariance_scale_passes,
    )
    signs = refine_covariance_signs(
        target,
        factorized.left_binary,
        factorized.right_binary,
        first_scales.scale_pre,
        first_scales.scale_mid,
        first_scales.scale_post,
        regularized,
        output_importance,
        left_steps=left_flip_steps,
        right_batches=right_flip_batches,
        right_batch_size=right_flip_batch_size,
    )
    final_scales = fit_covariance_scales(
        target,
        signs.left_binary,
        signs.right_binary,
        first_scales.scale_pre,
        first_scales.scale_mid,
        first_scales.scale_post,
        regularized,
        output_importance,
        alternating_passes=covariance_scale_passes,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.synchronize(protocol.device)
    candidate_wall = time.perf_counter() - candidate_started
    candidate_metadata = {
        "initial_covariance_error": first_scales.before_error,
        "after_first_scale_error": first_scales.after_error,
        "first_scale_fit_accepted": first_scales.accepted,
        "first_scale_fit_rollback_reason": first_scales.rollback_reason,
        "sign_before_error": signs.before_error,
        "sign_after_error": signs.after_error,
        "left_flips": signs.left_flips,
        "right_flips": signs.right_flips,
        "final_scale_before_error": final_scales.before_error,
        "final_scale_after_error": final_scales.after_error,
        "final_scale_fit_accepted": final_scales.accepted,
        "final_scale_fit_rollback_reason": final_scales.rollback_reason,
        "wall_seconds": candidate_wall,
        "peak_device_bytes": (
            int(torch.cuda.max_memory_allocated(protocol.device))
            if protocol.device.startswith("cuda")
            else 0
        ),
    }
    candidate = _base_result_payload(
        handle,
        group,
        protocol,
        profiles,
        target,
        final_scales.reconstruction,
        output_importance,
        diagonal,
        fit_covariance.to(protocol.device),
        held_out_covariance.to(protocol.device) if held_out_covariance is not None else None,
        candidate_metadata,
    )
    del (
        target,
        raw_input,
        output_importance,
        diagonal,
        regularized,
        factorized,
        fitted,
        first_scales,
        signs,
        final_scales,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return baseline, candidate


def _reconstruction_set(
    group_results: dict[str, dict[str, Any]],
    member_results: dict[str, tuple[tuple[MemberSpec, torch.Tensor, float], ...]],
) -> SpliceReconstructionSet:
    reconstructions = []
    unit_members = []
    unit_errors = []
    for key in sorted(group_results):
        layers = []
        group_members = member_results[key]
        group_error = float(group_results[key]["evaluation"]["original_error"])
        group_target = float(group_results[key]["evaluation"]["original_target"])
        for member, weight, _energy in group_members:
            layer = LayerId(BlockId(member.block), PROJECTION_PATHS[member.projection])
            layers.append(layer)
            reconstructions.append(
                SpliceReconstruction(
                    layer,
                    weight,
                    None,
                    group_error / max(group_target, 1e-30),
                )
            )
        unit_members.append((key, tuple(layers)))
        unit_errors.append((key, group_error / max(group_target, 1e-30)))
    blocks = {item.layer.block.index for item in reconstructions}
    expected_layers = len(blocks) * len(PROJECTION_PATHS)
    if (
        len(reconstructions) != expected_layers
        or len({item.layer for item in reconstructions}) != expected_layers
    ):
        raise ValueError("covariance reconstruction inventory must contain complete blocks")
    for block in blocks:
        paths = {
            item.layer.path
            for item in reconstructions
            if item.layer.block.index == block
        }
        if paths != set(PROJECTION_PATHS.values()):
            raise ValueError(f"covariance reconstruction block {block} is incomplete")
    return SpliceReconstructionSet(
        tuple(reconstructions),
        tuple(unit_members),
        tuple(unit_errors),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 24))
    parser.add_argument("--block-output-blocks", type=_parse_ints)
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--fit-tokens", type=int, default=2048)
    parser.add_argument("--held-out-tokens", type=int, default=2048)
    parser.add_argument("--wikitext-samples", type=int, default=12)
    parser.add_argument("--block-output-samples", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--target-bpw", type=float, default=1.0)
    parser.add_argument("--rank-alignment", type=int, default=1)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=400)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument("--covariance-scale-passes", type=int, default=2)
    parser.add_argument("--left-flip-steps", type=int, default=8)
    parser.add_argument("--right-flip-batches", type=int, default=8)
    parser.add_argument("--right-flip-batch-size", type=int, default=128)
    parser.add_argument("--damp-fraction", type=float, default=0.01)
    parser.add_argument("--covariance-diagonal-blend", type=float, default=0.0)
    parser.add_argument("--covariance-promotion-threshold", type=float, default=0.10)
    parser.add_argument("--minimum-relative-kl-gain", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if not args.blocks or len(set(args.blocks)) != len(args.blocks) or any(block < 0 for block in args.blocks):
        raise ValueError("covariance-binary blocks must be unique non-negative indices")
    if (
        args.fit_tokens <= 0
        or args.held_out_tokens <= 0
        or args.wikitext_samples <= 0
        or args.block_output_samples <= 0
        or args.sequence_length < 2
    ):
        raise ValueError("covariance-binary probe dataset dimensions must be positive")
    if (
        args.covariance_scale_passes < 0
        or args.left_flip_steps < 0
        or args.right_flip_batches < 0
        or args.right_flip_batch_size <= 0
    ):
        raise ValueError("covariance-binary refinement settings are invalid")
    if (
        args.damp_fraction < 0
        or not 0 <= args.covariance_diagonal_blend <= 1
        or not 0 <= args.covariance_promotion_threshold <= 1
    ):
        raise ValueError("covariance-binary thresholds are invalid")
    if not 0 <= args.minimum_relative_kl_gain <= 1:
        raise ValueError("minimum relative KL gain must be in [0, 1]")
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    adapter = adapter_for_config(config)
    expected_blocks = adapter.decoder_block_count_from_config(config)
    if any(block >= expected_blocks for block in args.blocks):
        raise ValueError("requested block is outside the model")
    block_output_blocks = args.block_output_blocks or args.blocks
    if any(block not in args.blocks for block in block_output_blocks):
        raise ValueError("block-output blocks must be a subset of factorized blocks")
    covariance_samples = math.ceil(
        (args.fit_tokens + args.held_out_tokens) / args.sequence_length
    )
    all_tokens, dataset_fingerprint, _bos = _wikitext_tokens(
        args.snapshot,
        samples=covariance_samples + args.wikitext_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    covariance_tokens = all_tokens[:covariance_samples]
    functional_tokens = all_tokens[covariance_samples:]
    protocol = ProbeProtocol(
        1,
        args.model_revision,
        args.target_bpw,
        args.rank_alignment,
        args.scale_bits,
        args.outer_iterations,
        args.inner_iterations,
        args.regularization,
        args.penalty_schedule,
        args.convergence_check_interval,
        True,
        args.scale_fit_passes,
        args.seed,
        args.device,
        str(args.calibration_state.resolve()),
        0.0,
    )
    profiles = load_calibration_profiles(args.calibration_state, 0.0)
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        decoder_blocks = tuple(adapter.get_decoder_layers(teacher))
        covariances = _capture_covariances(
            teacher,
            _input_capture_specs(decoder_blocks, args.blocks),
            covariance_tokens,
            fit_tokens=args.fit_tokens,
            held_out_tokens=args.held_out_tokens,
            device=args.device,
        )
        teacher.to("cpu")
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        baseline_groups: dict[str, dict[str, Any]] = {}
        baseline_members: dict[str, tuple[tuple[MemberSpec, torch.Tensor, float], ...]] = {}
        candidate_groups: dict[str, dict[str, Any]] = {}
        candidate_members: dict[str, tuple[tuple[MemberSpec, torch.Tensor, float], ...]] = {}
        with safe_open(str(args.model), framework="pt", device="cpu") as handle:
            for block in args.blocks:
                for group in block_groups(block):
                    key = f"{block}:{group.label}"
                    pair = (
                        covariances[_covariance_key(block, group.label)]
                        if group.label in TRANSFORMED_GROUPS
                        else None
                    )
                    print(f"factorizing and refining group={key}", flush=True)
                    baseline, candidate = _group_pair(
                        handle,
                        group,
                        protocol,
                        profiles,
                        None if pair is None else pair[0],
                        None if pair is None else pair[1],
                        damp_fraction=args.damp_fraction,
                        covariance_diagonal_blend=args.covariance_diagonal_blend,
                        covariance_scale_passes=args.covariance_scale_passes,
                        left_flip_steps=args.left_flip_steps,
                        right_flip_batches=args.right_flip_batches,
                        right_flip_batch_size=args.right_flip_batch_size,
                    )
                    baseline_groups[key], baseline_members[key] = baseline
                    candidate_groups[key], candidate_members[key] = candidate
        reconstruction_sets = {
            BASELINE_KEY: _reconstruction_set(baseline_groups, baseline_members),
            CANDIDATE_KEY: _reconstruction_set(candidate_groups, candidate_members),
        }
        reconstruction_metrics = {
            BASELINE_KEY: {
                "aggregate": _aggregate_groups(baseline_groups),
                "groups": baseline_groups,
            },
            CANDIDATE_KEY: {
                "aggregate": _aggregate_groups(candidate_groups),
                "groups": candidate_groups,
            },
        }
        baseline_bits = int(reconstruction_metrics[BASELINE_KEY]["aggregate"]["actual_bits"])
        candidate_bits = int(reconstruction_metrics[CANDIDATE_KEY]["aggregate"]["actual_bits"])
        if candidate_bits != baseline_bits:
            raise ValueError("covariance refinement changed the physical factor bit budget")
        baseline_ranks = {key: int(value["rank"]) for key, value in baseline_groups.items()}
        candidate_ranks = {key: int(value["rank"]) for key, value in candidate_groups.items()}
        if candidate_ranks != baseline_ranks:
            raise ValueError("covariance refinement changed the factor rank inventory")

        teacher.to(args.device)
        teacher.eval()
        decoder_blocks = tuple(adapter.get_decoder_layers(teacher))
        output_tokens = functional_tokens[: args.block_output_samples]
        output_reference = _capture_outputs(
            teacher,
            {block: decoder_blocks[block] for block in block_output_blocks},
            output_tokens,
            device=args.device,
        )
        arms = ("full",) if args.full_only else ("full", *(f"block:{block}" for block in args.blocks))
        kl_results = {}
        block_outputs = {}
        teacher_batches: tuple[torch.Tensor, ...] | None = None
        baseline_nll = math.nan
        for key in (BASELINE_KEY, CANDIDATE_KEY):
            evaluator = DenseKlSpliceEvaluator(
                teacher,
                reconstruction_sets[key],
                functional_tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            if teacher_batches is None:
                baseline_nll, teacher_batches = evaluator.teacher_cache_state()
            else:
                evaluator.install_teacher_cache(baseline_nll, teacher_batches)
            kl_results[key] = {arm: evaluator(arm) for arm in arms}
            block_outputs[key] = _isolated_block_outputs(
                evaluator,
                teacher,
                decoder_blocks,
                output_reference,
                output_tokens,
                device=args.device,
            )
            del evaluator
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del teacher

    comparisons = {
        arm: _paired_summary(
            kl_results[BASELINE_KEY][arm],
            kl_results[CANDIDATE_KEY][arm],
        )
        for arm in arms
    }
    held_baseline = float(
        reconstruction_metrics[BASELINE_KEY]["aggregate"]["held_out_covariance_error"]
    )
    held_candidate = float(
        reconstruction_metrics[CANDIDATE_KEY]["aggregate"]["held_out_covariance_error"]
    )
    held_reduction = (held_baseline - held_candidate) / max(held_baseline, 1e-30)
    full_comparison = comparisons["full"]
    functional_promotion = (
        float(full_comparison["relative_kl_delta"]) <= -args.minimum_relative_kl_gain
        and float(full_comparison["upper_delta"]) < 0
    )
    promotes = (
        held_reduction >= args.covariance_promotion_threshold
        and functional_promotion
    )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "role": "analysis-only covariance-aware binary refinement; not a compression artifact",
        "model_source": MODEL_SOURCE,
        "model_revision": args.model_revision,
        "blocks": list(args.blocks),
        "block_output_blocks": list(block_output_blocks),
        "functional_arms": list(arms),
        "refined_groups": sorted(TRANSFORMED_GROUPS),
        "held_identical_group": "down",
        "representation": "diag(post) @ B_left @ diag(mid) @ B_right @ diag(pre)",
        "protocol": {
            **to_dict(protocol),
            "fit_tokens": args.fit_tokens,
            "held_out_tokens": args.held_out_tokens,
            "wikitext_samples": args.wikitext_samples,
            "block_output_samples": args.block_output_samples,
            "sequence_length": args.sequence_length,
            "dataset_fingerprint": dataset_fingerprint,
            "covariance_slice_hash": _token_hash(covariance_tokens),
            "functional_slice_hash": _token_hash(functional_tokens),
            "damp_fraction": args.damp_fraction,
            "covariance_diagonal_blend": args.covariance_diagonal_blend,
            "covariance_scale_passes": args.covariance_scale_passes,
            "left_flip_steps": args.left_flip_steps,
            "right_flip_batches": args.right_flip_batches,
            "right_flip_batch_size": args.right_flip_batch_size,
            "covariance_promotion_threshold": args.covariance_promotion_threshold,
            "minimum_relative_kl_gain": args.minimum_relative_kl_gain,
        },
        "teacher_baseline_nll": baseline_nll,
        "reconstruction": reconstruction_metrics,
        "kl": {
            key: {arm: to_dict(result) for arm, result in results.items()}
            for key, results in kl_results.items()
        },
        "isolated_block_output_normalized_rmse": block_outputs,
        "paired_comparisons_vs_baseline": comparisons,
        "promotion": {
            "held_out_covariance_relative_error_reduction": held_reduction,
            "functional_promotion": functional_promotion,
            "promotes_covariance_binary_refinement": promotes,
        },
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "promotion": payload["promotion"],
                "paired_comparisons": comparisons,
                "isolated_block_output_normalized_rmse": block_outputs,
            },
            indent=2,
        )
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
