"""Pure bit accounting, allocation, and retry policies."""

from __future__ import annotations

import math

from .models import AttemptSummary, BitCost, RetryDecision


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
