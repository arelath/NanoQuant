from __future__ import annotations

import torch

from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.progressive_sign_fixing import (
    apply_progressive_constraint,
    choose_next_fixed_bit,
    empty_progressive_constraint,
    factorize_progressive_sign_fixing_admm,
    maximum_progressive_rank_for_budget,
    progressive_sign_fixing_bit_cost,
)


def test_progressive_rank_stays_within_free_word_budget() -> None:
    baseline = factor_bit_cost(1152, 6912, 970, scale_bits=16).total
    rank8 = maximum_progressive_rank_for_budget(
        1152,
        6912,
        baseline,
        variable_bits_per_word=8,
    )
    rank12 = maximum_progressive_rank_for_budget(
        1152,
        6912,
        baseline,
        variable_bits_per_word=12,
    )

    assert rank8 > rank12 > 2 * 970
    assert progressive_sign_fixing_bit_cost(
        1152,
        6912,
        rank12,
        variable_bits_per_word=12,
    ).total <= baseline


def test_next_fixed_bit_selects_strongest_majority() -> None:
    values = torch.ones((4, 32))
    values[:2, 0] = -1
    values[0, 1] = -1
    constraint = choose_next_fixed_bit(
        values,
        empty_progressive_constraint("cpu"),
        iteration=7,
    )

    # Many positions are unanimous +1; the first such position wins the
    # deterministic tie after positions 0 and 1 show weaker majorities.
    assert constraint.decisions[0].position == 2
    assert constraint.decisions[0].value == 1
    assert constraint.decisions[0].majority_fraction == 1.0


def test_apply_progressive_constraint_preserves_variable_positions() -> None:
    signs = torch.tensor([[1.0, -1.0] * 16])
    constraint = empty_progressive_constraint("cpu")
    constraint = choose_next_fixed_bit(torch.ones((3, 32)), constraint, iteration=1)

    constrained = apply_progressive_constraint(signs, constraint)

    position = constraint.decisions[0].position
    assert constrained[0, position] == 1
    variable = ~constraint.fixed_mask
    assert torch.equal(constrained[0, variable], signs[0, variable])


def test_progressive_factorization_exports_fixed_positions() -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn((5, 32), generator=generator)
    result = factorize_progressive_sign_fixing_admm(
        weight,
        torch.ones(32),
        torch.ones(5),
        32,
        torch.Generator().manual_seed(23),
        variable_bits_per_word=28,
        outer_iterations=8,
        inner_iterations=2,
        convergence_check_interval=2,
        fixing_warmup_fraction=0.0,
        fixing_fraction=0.5,
    )

    assert result.left_constraint.fixed_count == 4
    assert result.right_constraint.fixed_count == 4
    assert torch.equal(
        apply_progressive_constraint(
            result.factors.left_binary,
            result.left_constraint,
        ),
        result.factors.left_binary,
    )
    assert torch.equal(
        apply_progressive_constraint(
            result.factors.right_binary,
            result.right_constraint,
        ),
        result.factors.right_binary,
    )
    assert torch.isfinite(result.factors.reconstruction).all()
