"""Pure bit accounting, allocation, and retry policies."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import AttemptSummary, BitCost, RetryDecision


@dataclass(frozen=True, slots=True)
class RankResponseSegment:
    maximum_rank_fraction: float
    beta_per_rank: float


@dataclass(frozen=True, slots=True)
class ReconstructionAllocationUnit:
    unit_id: str
    out_features: int
    in_features: int
    baseline_rank: int
    baseline_squared_error: float
    sensitivity: float
    protected: bool
    calibrated_rank_floor_fraction: float
    calibrated_rank_ceiling_fraction: float
    segments: tuple[RankResponseSegment, ...]
    fixed_bits: int = 0


@dataclass(frozen=True, slots=True)
class ReconstructionAllocationDecision:
    unit_id: str
    baseline_rank: int
    planned_rank: int
    baseline_squared_error: float
    predicted_squared_error: float


@dataclass(frozen=True, slots=True)
class ReconstructionAllocationResult:
    decisions: tuple[ReconstructionAllocationDecision, ...]
    spent_bits: int
    remaining_bits: int
    protected_baseline_objective: float
    protected_planned_objective: float


def integrated_rank_response(
    baseline_rank: int,
    rank: int,
    calibrated_floor_fraction: float,
    segments: tuple[RankResponseSegment, ...],
) -> float:
    """Integrate a piecewise per-rank beta curve from baseline to ``rank``."""

    if baseline_rank <= 0 or not segments:
        raise ValueError("rank response requires a positive baseline and segments")
    lower = calibrated_floor_fraction * baseline_rank
    upper = segments[-1].maximum_rank_fraction * baseline_rank
    if rank < math.ceil(lower - 1e-9) or rank > math.floor(upper + 1e-9):
        raise ValueError("rank lies outside the calibrated response range")
    start = float(baseline_rank)
    end = float(rank)
    sign = 1.0
    if end < start:
        start, end = end, start
        sign = -1.0
    integral = 0.0
    segment_floor = lower
    for segment in segments:
        segment_ceiling = segment.maximum_rank_fraction * baseline_rank
        overlap = max(0.0, min(end, segment_ceiling) - max(start, segment_floor))
        integral += overlap * segment.beta_per_rank
        segment_floor = segment_ceiling
    return sign * integral


def predicted_squared_error(unit: ReconstructionAllocationUnit, rank: int) -> float:
    integral = integrated_rank_response(
        unit.baseline_rank,
        rank,
        unit.calibrated_rank_floor_fraction,
        unit.segments,
    )
    return unit.baseline_squared_error * math.exp(-2.0 * integral)


def allocate_reconstruction_rank_budget(
    units: tuple[ReconstructionAllocationUnit, ...],
    target_bits: int,
    *,
    multiple: int,
    floor_fraction: float,
    ceiling_fraction: float,
    sensitivity_strength: float,
    protected_rank_floor_fraction: float,
    target_protected_error_reduction_fraction: float,
) -> ReconstructionAllocationResult:
    """Allocate exact aligned factor ranks by diminishing predicted error reduction per bit."""

    if not units or target_bits <= 0 or multiple <= 0:
        raise ValueError("reconstruction allocation inputs are invalid")
    if not 0 <= sensitivity_strength <= 1:
        raise ValueError("sensitivity strength must be in [0, 1]")
    baseline_weights = [
        factor_bit_cost(unit.out_features, unit.in_features, unit.baseline_rank).total for unit in units
    ]
    if any(unit.baseline_squared_error <= 0 or unit.sensitivity <= 0 for unit in units):
        raise ValueError("reconstruction evidence must contain positive errors and sensitivities")
    log_mean = sum(bits * math.log(unit.sensitivity) for unit, bits in zip(units, baseline_weights, strict=True)) / sum(
        baseline_weights
    )
    sensitivity_weights = {
        unit.unit_id: math.exp(sensitivity_strength * (math.log(unit.sensitivity) - log_mean)) for unit in units
    }
    floors: dict[str, int] = {}
    caps: dict[str, int] = {}
    ranks: dict[str, int] = {}
    for unit in units:
        effective_floor = max(floor_fraction, unit.calibrated_rank_floor_fraction)
        if unit.protected:
            effective_floor = max(effective_floor, protected_rank_floor_fraction)
        effective_ceiling = min(ceiling_fraction, unit.calibrated_rank_ceiling_fraction)
        floor_rank = math.ceil(unit.baseline_rank * effective_floor / multiple) * multiple
        cap_rank = math.floor(unit.baseline_rank * effective_ceiling / multiple) * multiple
        physical_cap = math.floor(min(unit.in_features, unit.out_features) / multiple) * multiple
        cap_rank = min(cap_rank, physical_cap)
        if floor_rank > cap_rank:
            raise ValueError(f"reconstruction rank floor exceeds cap for {unit.unit_id}")
        floors[unit.unit_id] = floor_rank
        caps[unit.unit_id] = cap_rank
        ranks[unit.unit_id] = floor_rank

    def cost(unit: ReconstructionAllocationUnit, rank: int) -> int:
        return factor_bit_cost(unit.out_features, unit.in_features, rank).total + unit.fixed_bits

    spent = sum(cost(unit, ranks[unit.unit_id]) for unit in units)
    if spent > target_bits:
        raise ValueError("protected reconstruction rank floors exceed target bit budget")
    by_id = {unit.unit_id: unit for unit in units}
    while True:
        candidates: list[tuple[float, int, str, int]] = []
        for index, unit in enumerate(units):
            current = ranks[unit.unit_id]
            proposed = current + multiple
            if proposed > caps[unit.unit_id]:
                continue
            marginal = cost(unit, proposed) - cost(unit, current)
            if spent + marginal > target_bits:
                continue
            gain = sensitivity_weights[unit.unit_id] * (
                predicted_squared_error(unit, current) - predicted_squared_error(unit, proposed)
            )
            candidates.append((gain / marginal, -index, unit.unit_id, marginal))
        if not candidates:
            break
        _priority, _tie, selected, marginal = max(candidates)
        ranks[selected] += multiple
        spent += marginal

    decisions = tuple(
        ReconstructionAllocationDecision(
            unit.unit_id,
            unit.baseline_rank,
            ranks[unit.unit_id],
            unit.baseline_squared_error,
            predicted_squared_error(unit, ranks[unit.unit_id]),
        )
        for unit in units
    )
    protected = [unit for unit in units if unit.protected]
    protected_baseline = sum(sensitivity_weights[unit.unit_id] * unit.baseline_squared_error for unit in protected)
    protected_planned = sum(
        sensitivity_weights[unit.unit_id] * predicted_squared_error(unit, ranks[unit.unit_id]) for unit in protected
    )
    if protected and protected_planned > protected_baseline * (1 - target_protected_error_reduction_fraction):
        raise ValueError("requested protected reconstruction improvement is infeasible")
    if set(ranks) != set(by_id):
        raise AssertionError("reconstruction allocator lost a planning unit")
    return ReconstructionAllocationResult(
        decisions,
        spent,
        target_bits - spent,
        protected_baseline,
        protected_planned,
    )


def factor_bit_cost(
    out_features: int, in_features: int, rank: int, *, scale_bits: int = 16, rank_alignment: int = 1
) -> BitCost:
    if min(out_features, in_features, rank) < 0 or rank_alignment <= 0:
        raise ValueError("dimensions/rank must be non-negative and alignment positive")
    padded_rank = math.ceil(rank / rank_alignment) * rank_alignment
    logical_binary = rank * (out_features + in_features)
    stored_binary = padded_rank * (out_features + in_features)
    return BitCost(
        binary_factor_bits=logical_binary,
        scale_bits=scale_bits * (out_features + in_features + rank),
        padding_bits=stored_binary - logical_binary,
    )


def outlier_bit_cost(out_features: int, count: int, *, value_bits: int, index_bits: int | None = None) -> BitCost:
    if min(out_features, count, value_bits) < 0:
        raise ValueError("cost inputs must not be negative")
    index_width = index_bits if index_bits is not None else max(1, math.ceil(math.log2(max(2, out_features))))
    return BitCost(outlier_value_bits=out_features * count * value_bits, outlier_index_bits=count * index_width)


def uniform_rank(
    total_weight_elements: int,
    layer_dimensions: tuple[tuple[int, int], ...],
    target_bpw: float,
    *,
    multiple: int = 1,
    fixed_bits: int = 0,
) -> int:
    denominator = sum(out_features + in_features for out_features, in_features in layer_dimensions)
    available = max(0.0, total_weight_elements * target_bpw - fixed_bits)
    raw = available / denominator if denominator else 0
    return max(0, int(raw // multiple) * multiple)


def bounded_rank(
    uniform: int,
    sensitivity: float,
    *,
    alpha: float,
    floor_fraction: float,
    ceiling_fraction: float,
    multiple: int,
    edge_boost: float = 0.0,
    edge: bool = False,
) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    candidate = uniform * (1 + alpha * sensitivity + (edge_boost if edge else 0))
    candidate = min(uniform * ceiling_fraction, max(uniform * floor_fraction, candidate))
    return max(multiple, int(round(candidate / multiple)) * multiple)


def allocate_rank_budget(
    layer_dimensions: tuple[tuple[int, int], ...],
    utilities: tuple[float, ...],
    target_bits: int,
    *,
    fixed_bits: int = 0,
    multiple: int = 1,
    floor_ranks: tuple[int, ...] | None = None,
    ceiling_ranks: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Allocate affordable rank units by marginal utility per bit."""
    if len(layer_dimensions) != len(utilities) or multiple <= 0:
        raise ValueError("allocation inputs have inconsistent lengths")
    count = len(layer_dimensions)
    floors = floor_ranks or (0,) * count
    ceilings = ceiling_ranks or tuple(min(shape) for shape in layer_dimensions)
    if len(floors) != count or len(ceilings) != count:
        raise ValueError("rank bounds have inconsistent lengths")
    ranks = [math.ceil(max(0, floor) / multiple) * multiple for floor in floors]
    costs = [(out_features + in_features) * multiple for out_features, in_features in layer_dimensions]
    spent = fixed_bits + sum(
        rank * (out_features + in_features)
        for rank, (out_features, in_features) in zip(ranks, layer_dimensions, strict=True)
    )
    if spent > target_bits:
        raise ValueError("rank floors exceed target bit budget")
    while True:
        candidates = [
            index
            for index in range(count)
            if ranks[index] + multiple <= ceilings[index] and spent + costs[index] <= target_bits
        ]
        if not candidates:
            break
        selected = max(candidates, key=lambda index: (utilities[index] / costs[index], -index))
        ranks[selected] += multiple
        spent += costs[selected]
    return tuple(ranks)


def retry_score(
    weighted_error: float, raw_error: float, weighted_threshold: float | None, raw_threshold: float | None
) -> float:
    ratios = []
    if weighted_threshold is not None:
        ratios.append(weighted_error / max(weighted_threshold, 1e-12))
    if raw_threshold is not None:
        ratios.append(raw_error / max(raw_threshold, 1e-12))
    return max(ratios, default=0.0)


def decide_retry(
    attempts: tuple[AttemptSummary, ...],
    *,
    maximum_attempts: int,
    rank_increase_fraction: float,
    rank_multiple: int,
    hard_rank_cap: int,
    available_extra_bits: int,
    cost_per_rank: int,
    weighted_threshold: float | None,
    raw_threshold: float | None,
) -> RetryDecision:
    if not attempts:
        raise ValueError("at least one attempt is required")
    best_index = min(
        range(len(attempts)), key=lambda index: (attempts[index].retry_score, attempts[index].weighted_error)
    )
    current = attempts[-1]
    score = retry_score(current.weighted_error, current.raw_error, weighted_threshold, raw_threshold)
    if score <= 1:
        return RetryDecision("accept_best", best_index, None, 0, "thresholds satisfied")
    if len(attempts) >= maximum_attempts:
        return RetryDecision("accept_best", best_index, None, 0, "attempt limit reached")
    proposed = math.ceil(current.rank * (1 + rank_increase_fraction) / rank_multiple) * rank_multiple
    proposed = min(proposed, hard_rank_cap)
    if proposed <= current.rank:
        return RetryDecision("accept_best", best_index, None, 0, "rank cap reached")
    extra = (proposed - current.rank) * cost_per_rank
    if extra > available_extra_bits:
        return RetryDecision("accept_best", best_index, None, 0, "extra-bit budget exhausted")
    return RetryDecision("retry", None, proposed, extra, "reconstruction threshold exceeded")


def effective_bpw(cost: BitCost, original_weight_elements: int) -> float:
    if original_weight_elements <= 0:
        raise ValueError("original weight elements must be positive")
    return cost.total / original_weight_elements
