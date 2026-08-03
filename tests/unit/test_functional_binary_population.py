from __future__ import annotations

import torch

from nanoquant.domain.functional_binary_population import (
    build_functional_binary_population,
    canonical_binary_hash,
)


def test_canonical_hash_removes_row_component_and_column_gauges() -> None:
    left = torch.tensor([[1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])
    right = torch.tensor([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]])
    gauged_left = left.clone()
    gauged_right = right.clone()
    gauged_left[1] *= -1
    gauged_left[:, 1] *= -1
    gauged_right[1] *= -1
    gauged_right[:, 2] *= -1

    assert canonical_binary_hash(left, right) == canonical_binary_hash(
        gauged_left, gauged_right
    )


def test_population_uses_coupled_component_gradient_proposals() -> None:
    left = torch.ones((4, 3))
    right = torch.ones((3, 5))
    left_gradient = torch.zeros_like(left)
    right_gradient = torch.zeros_like(right)
    left_gradient[[1, 3], 2] = torch.tensor([9.0, 8.0])
    right_gradient[2, [0, 4]] = torch.tensor([7.0, 6.0])

    population = build_functional_binary_population(
        left,
        right,
        left_gradient,
        right_gradient,
        population_size=4,
        flips_per_factor=2,
        seed=4,
    )

    assert len(population) == 4
    gradient = population[1]
    assert gradient.components == (2,)
    assert gradient.left_flips == 2
    assert gradient.right_flips == 2
    assert gradient.proposal_score == 30.0
    assert len({candidate.canonical_hash for candidate in population}) == len(population)


def test_population_is_deterministic_and_can_span_multiple_components() -> None:
    generator = torch.Generator().manual_seed(9)
    left = torch.randn((5, 4), generator=generator)
    right = torch.randn((4, 6), generator=generator)
    left_gradient = torch.randn((5, 4), generator=generator)
    right_gradient = torch.randn((4, 6), generator=generator)

    first = build_functional_binary_population(
        left,
        right,
        left_gradient,
        right_gradient,
        population_size=7,
        flips_per_factor=2,
        components_per_candidate=2,
        seed=17,
    )
    second = build_functional_binary_population(
        left,
        right,
        left_gradient,
        right_gradient,
        population_size=7,
        flips_per_factor=2,
        components_per_candidate=2,
        seed=17,
    )

    assert [candidate.canonical_hash for candidate in first] == [
        candidate.canonical_hash for candidate in second
    ]
    assert all(len(candidate.components) == 2 for candidate in first[1:])
