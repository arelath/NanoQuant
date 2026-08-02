"""Compare production full-rank ADMM with tiny exhaustive and intensive searches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from safetensors import safe_open

from nanoquant.config.codec import semantic_hash
from nanoquant.domain.calibration_math import shrink_importance
from nanoquant.domain.covariance_refinement import refine_binary_factors_under_covariance
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.scale_fit import fit_scales, reconstruct
from nanoquant.infrastructure.io_utils import atomic_write_json

PINNED_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"


@dataclass(frozen=True, slots=True)
class TinyOptimalityProtocol:
    schema_version: int
    model_revision: str
    exact_size: int
    heuristic_size: int
    include_exact: bool
    include_heuristic: bool
    synthetic_cases: int
    real_crops: int
    tensor_key: str | None
    profile_key: str | None
    calibration_shrinkage: float
    production_seeds: int
    outer_iterations: int
    inner_iterations: int
    scale_passes: int
    exact_scale_starts: int
    exact_scale_passes: int
    exact_batch_size: int
    population: int
    elite: int
    offspring_per_elite: int
    generations: int
    heuristic_scale_starts: int
    heuristic_scale_passes: int
    local_sweeps: int
    block_coordinate_sweeps: int
    seed: int
    device: str


@dataclass(frozen=True, slots=True)
class PopulationFit:
    left: torch.Tensor
    right: torch.Tensor
    pre: torch.Tensor
    mid: torch.Tensor
    post: torch.Tensor
    errors: torch.Tensor


def _logical_seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "little") % (2**63 - 1)


def _sign_from_bits(values: torch.Tensor, bit_count: int) -> torch.Tensor:
    shifts = torch.arange(bit_count, dtype=torch.int64, device=values.device)
    return (((values[:, None] >> shifts) & 1).float() * 2.0) - 1.0


def gauge_reduced_sign_pair_count(
    rows: int,
    columns: int | None = None,
    rank: int | None = None,
) -> int:
    columns = rows if columns is None else columns
    rank = min(rows, columns) if rank is None else rank
    exponent = (rank - 1) * (rows + columns - 1)
    if rows <= 0 or columns <= 0 or rank <= 0 or rank > min(rows, columns) or exponent > 62:
        raise ValueError("gauge-reduced sign geometry exceeds the int64 enumerator")
    return 1 << exponent


def gauge_reduced_sign_pair_range(
    rows: int,
    start: int,
    end: int,
    *,
    columns: int | None = None,
    rank: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a contiguous range of gauge-distinct sign pairs.

    `left` has an all-positive first row and first column. `right` has an
    all-positive first row. Signed pre/mid/post scales retain every represented
    matrix despite those gauge choices.
    """

    columns = rows if columns is None else columns
    rank = min(rows, columns) if rank is None else rank
    total = gauge_reduced_sign_pair_count(rows, columns, rank)
    if not 0 <= start <= end <= total:
        raise ValueError("gauge-reduced sign range is invalid")
    left_bits = (rows - 1) * (rank - 1)
    right_bits = columns * (rank - 1)
    right_count = 1 << right_bits
    combinations = end - start
    indices = torch.arange(start, end, dtype=torch.int64, device=device)
    left_indices = torch.div(indices, right_count, rounding_mode="floor")
    right_indices = indices.remainder(right_count)
    left = torch.ones((combinations, rows, rank), dtype=torch.float32, device=device)
    right = torch.ones((combinations, rank, columns), dtype=torch.float32, device=device)
    if left_bits:
        left[:, 1:, 1:] = _sign_from_bits(left_indices, left_bits).reshape(
            combinations, rows - 1, rank - 1
        )
    if right_bits:
        right[:, 1:, :] = _sign_from_bits(right_indices, right_bits).reshape(
            combinations, rank - 1, columns
        )
    return left, right


def gauge_reduced_sign_pairs(
    rows: int,
    *,
    columns: int | None = None,
    rank: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate every sign pair after fixing exact scale/sign gauges."""

    return gauge_reduced_sign_pair_range(
        rows,
        0,
        gauge_reduced_sign_pair_count(rows, columns, rank),
        columns=columns,
        rank=rank,
        device=device,
    )


def _weighted_errors(
    target: torch.Tensor,
    prediction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> torch.Tensor:
    return ((prediction - target).square() * output_importance[None, :, None] * input_importance[None, None, :]).sum(
        dim=(1, 2)
    )


def _population_reconstruct(
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(
        left * post[:, :, None],
        right * mid[:, :, None] * pre[:, None, :],
    )


def fit_scale_population(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    starts: int,
    passes: int,
    seed: int,
    epsilon: float = 1e-10,
) -> PopulationFit:
    """Run batched multistart ALS for fixed binary sign candidates."""

    if left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]:
        raise ValueError("population factors must have equal batch dimensions")
    if (
        starts <= 0
        or passes < 0
        or left.shape[1] != target.shape[0]
        or right.shape[2] != target.shape[1]
        or left.shape[2] != right.shape[1]
    ):
        raise ValueError("population scale-fit geometry or settings are invalid")
    device = left.device
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    target = target.to(device=device, dtype=dtype)
    input_weight = input_importance.to(device=device, dtype=dtype).reshape(-1).clamp_min(epsilon)
    output_weight = output_importance.to(device=device, dtype=dtype).reshape(-1).clamp_min(epsilon)
    left = torch.sign(left.to(dtype=dtype)).repeat_interleave(starts, dim=0)
    right = torch.sign(right.to(dtype=dtype)).repeat_interleave(starts, dim=0)
    count, rows, rank = left.shape
    columns = right.shape[2]
    generator = torch.Generator(device=device).manual_seed(seed)
    pre = torch.ones((count, columns), dtype=dtype, device=device)
    post = torch.ones((count, rows), dtype=dtype, device=device)
    if starts > 1:
        start_index = torch.arange(count, device=device).remainder(starts)
        randomized = start_index != 0
        random_count = int(randomized.sum())
        pre[randomized] = (
            torch.where(
                torch.rand((random_count, columns), generator=generator, device=device) >= 0.5,
                1.0,
                -1.0,
            )
            * torch.exp(0.5 * torch.randn((random_count, columns), generator=generator, device=device))
        ).to(dtype)
        post[randomized] = (
            torch.where(
                torch.rand((random_count, rows), generator=generator, device=device) >= 0.5,
                1.0,
                -1.0,
            )
            * torch.exp(0.5 * torch.randn((random_count, rows), generator=generator, device=device))
        ).to(dtype)
    mid = torch.ones((count, rank), dtype=dtype, device=device)
    best_pre = pre.clone()
    best_mid = mid.clone()
    best_post = post.clone()
    best_prediction = _population_reconstruct(left, right, pre, mid, post)
    best_errors = _weighted_errors(target, best_prediction, input_weight, output_weight)

    for _ in range(passes + 1):
        scaled_left = left * post[:, :, None]
        scaled_right = right * pre[:, None, :]
        left_gram = torch.einsum("bir,bis,i->brs", scaled_left, scaled_left, output_weight)
        right_gram = torch.einsum("brj,bsj,j->brs", scaled_right, scaled_right, input_weight)
        system = left_gram * right_gram
        system = 0.5 * (system + system.mT)
        diagonal = system.diagonal(dim1=-2, dim2=-1)
        ridge = (diagonal.abs().mean(dim=1) * 1e-6).clamp_min(max(epsilon, 1e-6))
        diagonal.add_(ridge[:, None])
        rhs = torch.einsum(
            "bir,ij,brj,i,j->br",
            scaled_left,
            target,
            scaled_right,
            output_weight,
            input_weight,
        )
        solution, info = torch.linalg.solve_ex(system, rhs[:, :, None])
        failed = info != 0
        if bool(failed.any()):
            fallback_system = system[failed].clone()
            fallback_system.diagonal(dim1=-2, dim2=-1).add_((ridge[failed] * 1_000).clamp_min(1e-3)[:, None])
            fallback, fallback_info = torch.linalg.solve_ex(
                fallback_system,
                rhs[failed, :, None],
            )
            fallback[fallback_info != 0] = 0
            solution[failed] = fallback
        mid = solution.squeeze(2)
        mid = torch.nan_to_num(mid)

        prediction = _population_reconstruct(left, right, pre, mid, post)
        errors = _weighted_errors(target, prediction, input_weight, output_weight)
        improved = torch.isfinite(errors) & (errors < best_errors)
        best_errors = torch.where(improved, errors, best_errors)
        best_pre[improved] = pre[improved]
        best_mid[improved] = mid[improved]
        best_post[improved] = post[improved]
        if _ == passes:
            break

        base = torch.bmm(left, right * mid[:, :, None] * pre[:, None, :])
        post = torch.nan_to_num(
            (base * target[None] * input_weight[None, None, :]).sum(dim=2)
            / (base.square() * input_weight[None, None, :]).sum(dim=2).clamp_min(epsilon)
        )
        base = torch.bmm(left * post[:, :, None] * mid[:, None, :], right)
        pre = torch.nan_to_num(
            (base * target[None] * output_weight[None, :, None]).sum(dim=1)
            / (base.square() * output_weight[None, :, None]).sum(dim=1).clamp_min(epsilon)
        )

    return PopulationFit(left, right, best_pre, best_mid, best_post, best_errors)


def _best_population(fit: PopulationFit, count: int) -> PopulationFit:
    count = min(count, fit.errors.numel())
    indices = torch.topk(fit.errors, count, largest=False).indices
    return PopulationFit(
        fit.left.index_select(0, indices),
        fit.right.index_select(0, indices),
        fit.pre.index_select(0, indices),
        fit.mid.index_select(0, indices),
        fit.post.index_select(0, indices),
        fit.errors.index_select(0, indices),
    )


def _concatenate_populations(*populations: PopulationFit) -> PopulationFit:
    return PopulationFit(
        torch.cat(tuple(item.left for item in populations), dim=0),
        torch.cat(tuple(item.right for item in populations), dim=0),
        torch.cat(tuple(item.pre for item in populations), dim=0),
        torch.cat(tuple(item.mid for item in populations), dim=0),
        torch.cat(tuple(item.post for item in populations), dim=0),
        torch.cat(tuple(item.errors for item in populations), dim=0),
    )


def exhaustive_sign_oracle(
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    starts: int,
    passes: int,
    seed: int,
    device: str,
    batch_size: int,
    rank: int | None = None,
) -> tuple[PopulationFit, int]:
    if batch_size <= 0:
        raise ValueError("exhaustive sign batch size must be positive")
    rank = min(target.shape) if rank is None else rank
    sign_configurations = gauge_reduced_sign_pair_count(target.shape[0], target.shape[1], rank)
    best: PopulationFit | None = None
    for start in range(0, sign_configurations, batch_size):
        end = min(start + batch_size, sign_configurations)
        left, right = gauge_reduced_sign_pair_range(
            target.shape[0],
            start,
            end,
            columns=target.shape[1],
            rank=rank,
            device=device,
        )
        fitted = fit_scale_population(
            target,
            left,
            right,
            input_importance,
            output_importance,
            starts=starts,
            passes=passes,
            seed=_logical_seed(seed, f"batch|{start}|{end}"),
        )
        candidate = _best_population(fitted, 1)
        if best is None or float(candidate.errors[0]) < float(best.errors[0]):
            best = candidate
    if best is None:
        raise RuntimeError("exhaustive sign enumeration produced no candidates")
    return best, sign_configurations


def _production_candidates(
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    protocol: TinyOptimalityProtocol,
    key: str,
) -> tuple[list[dict[str, float | int]], PopulationFit]:
    records: list[dict[str, float | int]] = []
    factors: list[tuple[torch.Tensor, ...]] = []
    target_device = target.to(protocol.device)
    input_device = input_importance.to(protocol.device)
    output_device = output_importance.to(protocol.device)
    for seed_index in range(protocol.production_seeds):
        started = time.perf_counter()
        generator = torch.Generator(device=protocol.device).manual_seed(
            _logical_seed(protocol.seed, f"{key}|admm|{seed_index}")
        )
        result = factorize_admm_with_parameters(
            target_device,
            input_device,
            output_device,
            target.shape[0],
            generator,
            AdmmParameters(
                outer_iterations=protocol.outer_iterations,
                inner_iterations=protocol.inner_iterations,
                regularization=3e-2,
                penalty_schedule="cubic",
                convergence_check_interval=100,
            ),
        )
        fitted = fit_scales(
            target_device,
            result.left_binary,
            result.right_binary,
            result.scale_pre,
            result.scale_mid,
            result.scale_post,
            input_device,
            output_device,
            alternating_passes=protocol.scale_passes,
        )
        records.append(
            {
                "seed": seed_index,
                "weighted_error": fitted.after_error,
                "wall_seconds": time.perf_counter() - started,
            }
        )
        factors.append(
            (
                result.left_binary.detach(),
                result.right_binary.detach(),
                fitted.scale_pre.detach(),
                fitted.scale_mid.detach(),
                fitted.scale_post.detach(),
            )
        )
    return records, PopulationFit(
        torch.stack(tuple(item[0] for item in factors)),
        torch.stack(tuple(item[1] for item in factors)),
        torch.stack(tuple(item[2] for item in factors)),
        torch.stack(tuple(item[3] for item in factors)),
        torch.stack(tuple(item[4] for item in factors)),
        torch.tensor(
            [record["weighted_error"] for record in records],
            device=target_device.device,
        ),
    )


def _random_signs(count: int, size: int, generator: torch.Generator, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.randint(0, 2, (count, size, size), generator=generator, device=device).float().mul_(2).sub_(1)
    right = torch.randint(0, 2, (count, size, size), generator=generator, device=device).float().mul_(2).sub_(1)
    return left, right


def _mutate_elites(
    elite: PopulationFit,
    offspring_per_elite: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = elite.left.repeat_interleave(offspring_per_elite, dim=0).clone()
    right = elite.right.repeat_interleave(offspring_per_elite, dim=0).clone()
    count, size, _ = left.shape
    flat = torch.cat((left.reshape(count, -1), right.reshape(count, -1)), dim=1)
    maximum = flat.shape[1]
    mutation_counts = torch.randint(1, max(2, maximum // 10 + 1), (count,), generator=generator, device=flat.device)
    for index in range(count):
        selected = torch.randperm(maximum, generator=generator, device=flat.device)[: int(mutation_counts[index])]
        flat[index, selected] *= -1
    return flat[:, : size * size].reshape(count, size, size), flat[:, size * size :].reshape(count, size, size)


def _local_refine(
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    population: PopulationFit,
    sweeps: int,
    scale_passes: int,
) -> PopulationFit:
    covariance = torch.diag(input_importance.float()).to(target.device)
    output = output_importance.float().to(target.device)
    candidates: list[tuple[torch.Tensor, ...]] = []
    errors: list[float] = []
    for index in range(population.errors.numel()):
        left = population.left[index]
        right = population.right[index]
        pre = population.pre[index]
        mid = population.mid[index]
        post = population.post[index]
        best_error = float(population.errors[index])
        for _ in range(sweeps):
            refined = refine_binary_factors_under_covariance(
                target,
                left,
                right,
                pre,
                mid,
                post,
                covariance,
                output,
                scale_passes=scale_passes,
                left_steps=target.shape[0] * target.shape[0],
                right_batches=target.shape[0] * target.shape[0],
                right_batch_size=1,
            )
            if refined.after_error >= best_error * (1.0 - 1e-10):
                break
            left, right = refined.left_binary, refined.right_binary
            pre, mid, post = refined.scale_pre, refined.scale_mid, refined.scale_post
            best_error = refined.after_error
        candidates.append((left, right, pre, mid, post))
        errors.append(best_error)
    best = min(range(len(errors)), key=errors.__getitem__)
    left, right, pre, mid, post = candidates[best]
    return PopulationFit(
        left[None],
        right[None],
        pre[None],
        mid[None],
        post[None],
        torch.tensor([errors[best]], device=target.device),
    )


def exhaustive_row_column_descent(
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    population: PopulationFit,
    *,
    sweeps: int,
    scale_passes: int,
    epsilon: float = 1e-10,
) -> PopulationFit:
    """Enumerate every rank-bit pattern for each left row and right column."""

    if sweeps < 0 or target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("exhaustive row/column descent settings or target are invalid")
    device = target.device
    size = target.shape[0]
    if size > 16:
        raise ValueError("row/column enumeration is bounded to rank 16")
    pattern_ids = torch.arange(1 << size, dtype=torch.int64, device=device)
    patterns = _sign_from_bits(pattern_ids, size).to(target.dtype)
    input_weight = input_importance.to(device=device, dtype=target.dtype).clamp_min(epsilon)
    output_weight = output_importance.to(device=device, dtype=target.dtype).clamp_min(epsilon)
    candidates: list[tuple[torch.Tensor, ...]] = []
    errors: list[float] = []
    for candidate in range(population.errors.numel()):
        left = population.left[candidate].to(device).float().clone()
        right = population.right[candidate].to(device).float().clone()
        pre = population.pre[candidate].to(device).float().clone()
        mid = population.mid[candidate].to(device).float().clone()
        post = population.post[candidate].to(device).float().clone()
        fitted = fit_scales(
            target,
            left,
            right,
            pre,
            mid,
            post,
            input_weight,
            output_weight,
            alternating_passes=scale_passes,
        )
        pre, mid, post = fitted.scale_pre, fitted.scale_mid, fitted.scale_post
        best_error = fitted.after_error
        for _ in range(sweeps):
            scaled_right = right * mid[:, None] * pre[None, :]
            row_bases = patterns @ scaled_right
            row_denominator = (row_bases.square() * input_weight[None, :]).sum(dim=1).clamp_min(epsilon)
            row_numerator = (target * input_weight[None, :]) @ row_bases.mT
            row_scores = row_numerator.square() / row_denominator[None, :]
            row_choices = row_scores.argmax(dim=1)
            left = patterns.index_select(0, row_choices)
            post = row_numerator.gather(1, row_choices[:, None]).squeeze(1) / row_denominator.index_select(
                0, row_choices
            )

            scaled_left = left * post[:, None] * mid[None, :]
            column_bases = scaled_left @ patterns.mT
            column_denominator = (column_bases.square() * output_weight[:, None]).sum(dim=0).clamp_min(epsilon)
            column_numerator = (target * output_weight[:, None]).mT @ column_bases
            column_scores = column_numerator.square() / column_denominator[None, :]
            column_choices = column_scores.argmax(dim=1)
            right = patterns.index_select(0, column_choices).mT.contiguous()
            pre = column_numerator.gather(1, column_choices[:, None]).squeeze(1) / column_denominator.index_select(
                0, column_choices
            )

            fitted = fit_scales(
                target,
                left,
                right,
                pre,
                mid,
                post,
                input_weight,
                output_weight,
                alternating_passes=scale_passes,
            )
            if fitted.after_error >= best_error * (1.0 - 1e-10):
                break
            pre, mid, post = fitted.scale_pre, fitted.scale_mid, fitted.scale_post
            best_error = fitted.after_error
        candidates.append((left, right, pre, mid, post))
        errors.append(best_error)
    best = min(range(len(errors)), key=errors.__getitem__)
    left, right, pre, mid, post = candidates[best]
    return PopulationFit(
        left[None],
        right[None],
        pre[None],
        mid[None],
        post[None],
        torch.tensor([errors[best]], device=device),
    )


def heuristic_oracle(
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    production: PopulationFit,
    protocol: TinyOptimalityProtocol,
    key: str,
) -> tuple[PopulationFit, dict[str, Any]]:
    generator = torch.Generator(device=protocol.device).manual_seed(_logical_seed(protocol.seed, f"{key}|heuristic"))
    left, right = _random_signs(protocol.population, target.shape[0], generator, protocol.device)
    left = torch.cat((left, production.left), dim=0)
    right = torch.cat((right, production.right), dim=0)
    population = fit_scale_population(
        target,
        left,
        right,
        input_importance,
        output_importance,
        starts=protocol.heuristic_scale_starts,
        passes=protocol.heuristic_scale_passes,
        seed=_logical_seed(protocol.seed, f"{key}|initial-scales"),
    )
    elite = _best_population(population, protocol.elite)
    history = [float(elite.errors.min())]
    for generation in range(protocol.generations):
        mutated_left, mutated_right = _mutate_elites(elite, protocol.offspring_per_elite, generator)
        fitted = fit_scale_population(
            target,
            torch.cat((elite.left, mutated_left), dim=0),
            torch.cat((elite.right, mutated_right), dim=0),
            input_importance,
            output_importance,
            starts=1,
            passes=protocol.heuristic_scale_passes,
            seed=_logical_seed(protocol.seed, f"{key}|generation|{generation}"),
        )
        elite = _best_population(fitted, protocol.elite)
        history.append(float(elite.errors.min()))
    refined = _local_refine(
        target.to(protocol.device),
        input_importance.to(protocol.device),
        output_importance.to(protocol.device),
        _best_population(elite, min(8, protocol.elite)),
        protocol.local_sweeps,
        protocol.scale_passes,
    )
    block_starts = _concatenate_populations(
        _best_population(production, min(16, production.errors.numel())),
        _best_population(elite, min(8, protocol.elite)),
    )
    block_coordinate = exhaustive_row_column_descent(
        target.to(protocol.device),
        input_importance.to(protocol.device),
        output_importance.to(protocol.device),
        block_starts,
        sweeps=protocol.block_coordinate_sweeps,
        scale_passes=protocol.scale_passes,
    )
    best = refined if float(refined.errors[0]) <= float(block_coordinate.errors[0]) else block_coordinate
    return best, {
        "random_candidates": protocol.population,
        "elite": protocol.elite,
        "offspring_per_elite": protocol.offspring_per_elite,
        "generations": protocol.generations,
        "local_candidates": min(8, protocol.elite),
        "block_coordinate_candidates": block_starts.errors.numel(),
        "block_coordinate_sweeps": protocol.block_coordinate_sweeps,
        "one_block_update_patterns": 1 << target.shape[0],
        "post_evolution_one_bit_error": float(refined.errors[0]),
        "post_evolution_block_coordinate_error": float(block_coordinate.errors[0]),
        "best_error_history": history,
    }


def _score_case(
    name: str,
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    protocol: TinyOptimalityProtocol,
    *,
    known_optimum: float | None,
) -> dict[str, Any]:
    target = target.float()
    input_importance = input_importance.float()
    output_importance = output_importance.float()
    denominator = float((target.square() * output_importance[:, None] * input_importance[None, :]).sum())
    production_records, production = _production_candidates(target, input_importance, output_importance, protocol, name)
    production_seed0 = float(production_records[0]["weighted_error"])
    production_best = float(production.errors.min())
    started = time.perf_counter()
    if target.shape[0] == protocol.exact_size:
        oracle, sign_configurations = exhaustive_sign_oracle(
            target,
            input_importance,
            output_importance,
            starts=protocol.exact_scale_starts,
            passes=protocol.exact_scale_passes,
            seed=_logical_seed(protocol.seed, f"{name}|exhaustive"),
            device=protocol.device,
            batch_size=protocol.exact_batch_size,
        )
        search = {
            "kind": "gauge-reduced-exhaustive-signs-multistart-als",
            "sign_configurations": sign_configurations,
            "continuous_starts_per_configuration": protocol.exact_scale_starts,
            "scale_passes": protocol.exact_scale_passes,
            "batch_size": protocol.exact_batch_size,
        }
    else:
        oracle, search = heuristic_oracle(
            target.to(protocol.device),
            input_importance.to(protocol.device),
            output_importance.to(protocol.device),
            production,
            protocol,
            name,
        )
        search["kind"] = "population-evolution-plus-exact-one-bit-descent"
    oracle_error = float(oracle.errors[0])
    floor = oracle_error if known_optimum is None else min(known_optimum, oracle_error)
    return {
        "name": name,
        "size": target.shape[0],
        "known_optimum_error": known_optimum,
        "target_weighted_energy": denominator,
        "production_seed0_error": production_seed0,
        "production_best_of_seeds_error": production_best,
        "oracle_error": oracle_error,
        "comparison_floor_error": floor,
        "production_seed0_nrmse": math.sqrt(production_seed0 / max(denominator, 1e-30)),
        "production_best_nrmse": math.sqrt(production_best / max(denominator, 1e-30)),
        "oracle_nrmse": math.sqrt(oracle_error / max(denominator, 1e-30)),
        "oracle_improvement_vs_seed0_fraction": 1.0 - oracle_error / max(production_seed0, 1e-30),
        "oracle_improvement_vs_best_production_fraction": 1.0 - oracle_error / max(production_best, 1e-30),
        "production_excess_over_oracle_fraction": production_best / max(oracle_error, 1e-30) - 1.0,
        "search_wall_seconds": time.perf_counter() - started,
        "production_seeds": production_records,
        "search": search,
    }


def _synthetic_cases(
    size: int, count: int, seed: int
) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, float | None]]:
    result: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, float | None]] = []
    for index in range(count):
        generator = torch.Generator().manual_seed(_logical_seed(seed, f"synthetic|{size}|{index}"))
        gaussian = torch.randn((size, size), generator=generator)
        result.append((f"gaussian-{size}x{size}-{index}", gaussian, torch.ones(size), torch.ones(size), None))
        left = torch.randint(0, 2, (size, size), generator=generator).float().mul_(2).sub_(1)
        right = torch.randint(0, 2, (size, size), generator=generator).float().mul_(2).sub_(1)
        pre = torch.exp(0.35 * torch.randn(size, generator=generator))
        mid = torch.exp(0.35 * torch.randn(size, generator=generator))
        post = torch.exp(0.35 * torch.randn(size, generator=generator))
        represented = reconstruct(left, right, pre, mid, post)
        result.append((f"represented-{size}x{size}-{index}", represented, torch.ones(size), torch.ones(size), 0.0))
    return result


def _load_profile(state: Path, key: str, shrinkage: float) -> tuple[torch.Tensor, torch.Tensor]:
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    sample_count = int(manifest["sample_count"])
    matches = [index for index, layer in enumerate(manifest["layers"]) if layer["path"] == key]
    if len(matches) != 1:
        raise ValueError(f"calibration profile must resolve exactly once: {key}")
    index = matches[0]
    with safe_open(str(state / "state.safetensors"), framework="pt", device="cpu") as handle:
        return (
            shrink_importance(handle.get_tensor(f"layer_{index}.inputs.total").float() / sample_count, shrinkage),
            shrink_importance(handle.get_tensor(f"layer_{index}.outputs.total").float() / sample_count, shrinkage),
        )


def _real_cases(
    args: argparse.Namespace, size: int
) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, None]]:
    if args.real_crops == 0:
        return []
    if args.model is None or args.calibration_state is None or args.tensor_key is None or args.profile_key is None:
        raise ValueError("real crops require model, calibration state, tensor key, and profile key")
    input_importance, output_importance = _load_profile(
        args.calibration_state, args.profile_key, args.calibration_shrinkage
    )
    with safe_open(str(args.model), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(args.tensor_key).float()
    if min(weight.shape) < size:
        raise ValueError("real tensor is smaller than the requested crop")
    result = []
    for index in range(args.real_crops):
        generator = torch.Generator().manual_seed(_logical_seed(args.seed, f"real|{size}|{index}"))
        row = int(torch.randint(0, weight.shape[0] - size + 1, (), generator=generator))
        column = int(torch.randint(0, weight.shape[1] - size + 1, (), generator=generator))
        result.append(
            (
                f"real-{size}x{size}-{index}-r{row}-c{column}",
                weight[row : row + size, column : column + size],
                input_importance[column : column + size],
                output_importance[row : row + size],
                None,
            )
        )
    return result


def run(args: argparse.Namespace) -> int:
    protocol = TinyOptimalityProtocol(
        5,
        args.model_revision,
        args.exact_size,
        args.heuristic_size,
        not args.skip_exact,
        not args.skip_heuristic,
        args.synthetic_cases,
        args.real_crops,
        args.tensor_key,
        args.profile_key,
        args.calibration_shrinkage,
        args.production_seeds,
        args.outer_iterations,
        args.inner_iterations,
        args.scale_passes,
        args.exact_scale_starts,
        args.exact_scale_passes,
        args.exact_batch_size,
        args.population,
        args.elite,
        args.offspring_per_elite,
        args.generations,
        args.heuristic_scale_starts,
        args.heuristic_scale_passes,
        args.local_sweeps,
        args.block_coordinate_sweeps,
        args.seed,
        args.device,
    )
    protocol_hash = semantic_hash(asdict(protocol))
    payload: dict[str, Any]
    if args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("protocol_hash") != protocol_hash:
            raise ValueError("existing tiny-optimality evidence uses another protocol")
    else:
        payload = {
            "schema_version": 1,
            "status": "running",
            "protocol_hash": protocol_hash,
            "protocol": asdict(protocol),
            "results": {},
        }
    cases = []
    sizes = []
    if not args.skip_exact:
        sizes.append(args.exact_size)
    if not args.skip_heuristic:
        sizes.append(args.heuristic_size)
    for size in sizes:
        cases.extend(_synthetic_cases(size, args.synthetic_cases, args.seed))
        cases.extend(_real_cases(args, size))
    for name, target, input_importance, output_importance, optimum in cases:
        if name in payload["results"]:
            continue
        print(f"running {name}", flush=True)
        payload["results"][name] = _score_case(
            name,
            target,
            input_importance,
            output_importance,
            protocol,
            known_optimum=optimum,
        )
        atomic_write_json(args.output, payload)
        result = payload["results"][name]
        print(
            f"production={result['production_best_nrmse']:.8f} "
            f"oracle={result['oracle_nrmse']:.8f} "
            f"oracle_gain={100 * result['oracle_improvement_vs_best_production_fraction']:.3f}%",
            flush=True,
        )
    payload["status"] = "completed"
    atomic_write_json(args.output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--calibration-state", type=Path)
    parser.add_argument("--tensor-key")
    parser.add_argument("--profile-key")
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--exact-size", type=int, default=3)
    parser.add_argument("--heuristic-size", type=int, default=10)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--skip-heuristic", action="store_true")
    parser.add_argument("--synthetic-cases", type=int, default=3)
    parser.add_argument("--real-crops", type=int, default=3)
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--production-seeds", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--scale-passes", type=int, default=16)
    parser.add_argument("--exact-scale-starts", type=int, default=16)
    parser.add_argument("--exact-scale-passes", type=int, default=64)
    parser.add_argument("--exact-batch-size", type=int, default=65536)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--elite", type=int, default=32)
    parser.add_argument("--offspring-per-elite", type=int, default=16)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--heuristic-scale-starts", type=int, default=2)
    parser.add_argument("--heuristic-scale-passes", type=int, default=24)
    parser.add_argument("--local-sweeps", type=int, default=8)
    parser.add_argument("--block-coordinate-sweeps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
