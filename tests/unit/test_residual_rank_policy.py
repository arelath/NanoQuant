"""Exact rank and scale charges for independent residual factors."""

import pytest

from nanoquant.domain.models import ArtifactRef, AttemptSummary, BitCost
from nanoquant.domain.planning import decide_retry, factor_bit_cost
from nanoquant.domain.residual_factorization import residual_stage_ranks


def test_residual_components_are_not_limited_to_the_matrix_dimension():
    assert residual_stage_ranks(1152, 6912, 1856, multiple=32) == (1152, 704)
    assert residual_stage_ranks(64, 96, 224, multiple=32) == (64, 64, 64, 32)
    assert residual_stage_ranks(64, 96, 32, multiple=32) == (32,)
    with pytest.raises(ValueError, match="aligned"):
        residual_stage_ranks(2, 3, 32, multiple=32)


def test_stage_activation_charges_independent_outer_scales_and_padding():
    actual = factor_bit_cost(64, 96, 160, residual_stages=True, rank_alignment=32)
    expected = sum((factor_bit_cost(64, 96, rank, rank_alignment=32) for rank in (64, 64, 32)), BitCost())
    assert actual == expected
    concatenated = factor_bit_cost(64, 96, 160, rank_alignment=32)
    assert actual.total - concatenated.total == 2 * 16 * (64 + 96)
    padded = factor_bit_cost(64, 96, 65, residual_stages=True, rank_alignment=32)
    assert padded == factor_bit_cost(64, 96, 64) + factor_bit_cost(64, 96, 1, rank_alignment=32)


def test_single_stage_policy_preserves_existing_cost_exactly():
    assert factor_bit_cost(64, 96, 64, residual_stages=True) == factor_bit_cost(64, 96, 64)


def test_retry_charges_a_new_stages_outer_scales_before_spending_the_reserve():
    def cost(rank):
        return factor_bit_cost(64, 96, rank, residual_stages=True, rank_alignment=32).total

    attempt = AttemptSummary(0, 64, ArtifactRef("fixture", "f", 1), 1.0, 1.0, BitCost(), 2.0, False, "test")
    kwargs = dict(maximum_attempts=2, rank_increase_fraction=0.5, rank_multiple=32, hard_rank_cap=128,
                  cost_per_rank=176, weighted_threshold=0.5, raw_threshold=None, rank_cost=cost)
    declined = decide_retry((attempt,), available_extra_bits=32 * 176, **kwargs)
    assert declined.action == "accept_best" and declined.reason == "extra-bit budget exhausted"
    exact = cost(96) - cost(64)
    accepted = decide_retry((attempt,), available_extra_bits=exact, **kwargs)
    assert accepted.action == "retry" and accepted.projected_extra_bits == exact
