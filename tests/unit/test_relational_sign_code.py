from __future__ import annotations

import torch

from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.relational_sign_code import (
    apply_relational_constraint,
    choose_next_relation,
    empty_relational_constraint,
    factorize_relational_sign_admm,
    maximum_relational_rank_for_budget,
    relational_sign_bit_cost,
)


def test_relational_rank_stays_within_free_word_budget() -> None:
    baseline = factor_bit_cost(1152, 6912, 970, scale_bits=16).total
    rank8 = maximum_relational_rank_for_budget(
        1152,
        6912,
        baseline,
        variable_bits_per_word=8,
    )
    rank12 = maximum_relational_rank_for_budget(
        1152,
        6912,
        baseline,
        variable_bits_per_word=12,
    )

    assert rank8 > rank12 > 2 * 970
    assert relational_sign_bit_cost(
        1152,
        6912,
        rank12,
        variable_bits_per_word=12,
    ).total <= baseline


def test_next_relation_finds_a_perfect_inversion() -> None:
    generator = torch.Generator().manual_seed(19)
    values = torch.sign(torch.randn((512, 32), generator=generator))
    values[:, 1] = -values[:, 0]

    constraint = choose_next_relation(
        values,
        empty_relational_constraint("cpu"),
        iteration=7,
    )

    decision = constraint.decisions[0]
    assert decision.source_root == 0
    assert decision.dependent_root == 1
    assert decision.relation == -1
    assert decision.agreement_fraction == 1.0


def test_relational_projection_preserves_every_learned_relation() -> None:
    generator = torch.Generator().manual_seed(23)
    values = torch.randn((9, 64), generator=generator)
    constraint = empty_relational_constraint("cpu")
    constraint = choose_next_relation(values, constraint, iteration=1)
    constraint = choose_next_relation(values, constraint, iteration=2)

    projected = apply_relational_constraint(values, constraint)

    for decision in constraint.decisions:
        assert torch.equal(
            projected[:, decision.dependent_root],
            decision.relation * projected[:, decision.source_root],
        )
        assert torch.equal(
            projected[:, 32 + decision.dependent_root],
            decision.relation * projected[:, 32 + decision.source_root],
        )


def test_relational_factorization_exports_decodable_signs() -> None:
    generator = torch.Generator().manual_seed(29)
    weight = torch.randn((5, 32), generator=generator)
    result = factorize_relational_sign_admm(
        weight,
        torch.ones(32),
        torch.ones(5),
        32,
        torch.Generator().manual_seed(31),
        variable_bits_per_word=28,
        outer_iterations=8,
        inner_iterations=2,
        convergence_check_interval=2,
        relation_warmup_fraction=0.0,
        relation_freeze_fraction=0.5,
    )

    assert result.left_constraint.root_count == 28
    assert result.right_constraint.root_count == 28
    assert torch.equal(
        apply_relational_constraint(
            result.factors.left_binary,
            result.left_constraint,
        ),
        result.factors.left_binary,
    )
    assert torch.equal(
        apply_relational_constraint(
            result.factors.right_binary,
            result.right_constraint,
        ),
        result.factors.right_binary,
    )
    assert torch.isfinite(result.factors.reconstruction).all()
