from __future__ import annotations

import pytest
import torch

from nanoquant.domain.binary_factor_search import refine_binary_factors_separable
from nanoquant.domain.scale_fit import reconstruct


def _random_problem(size: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    target = torch.randn((size, size), generator=generator)
    left = torch.randint(0, 2, (size, size), generator=generator).float().mul_(2).sub_(1)
    right = torch.randint(0, 2, (size, size), generator=generator).float().mul_(2).sub_(1)
    return target, left, right, torch.ones(size), torch.ones(size), torch.ones(size)


def test_direct_binary_search_never_regresses_weighted_objective() -> None:
    target, left, right, pre, mid, post = _random_problem(4, 12)
    input_importance = torch.tensor([0.5, 1.2, 0.8, 2.0])
    output_importance = torch.tensor([1.1, 0.7, 1.8, 0.6])

    result = refine_binary_factors_separable(
        target,
        left,
        right,
        pre,
        mid,
        post,
        input_importance,
        output_importance,
        outer_passes=4,
        scale_passes=8,
        block_bits=4,
        hard_fraction=1.0,
    )

    expected = reconstruct(
        result.left_binary,
        result.right_binary,
        result.scale_pre,
        result.scale_mid,
        result.scale_post,
    )
    torch.testing.assert_close(result.reconstruction, expected)
    assert result.after_error <= result.before_error
    assert result.block_patterns_evaluated <= 4 * 2 * 4 * (1 << 4)


def test_full_rank_block_search_is_at_least_as_strong_as_one_bit_search() -> None:
    target, left, right, pre, mid, post = _random_problem(4, 21)
    common = dict(
        target=target,
        left_binary=left,
        right_binary=right,
        scale_pre=pre,
        scale_mid=mid,
        scale_post=post,
        input_importance=torch.ones(4),
        output_importance=torch.ones(4),
        outer_passes=4,
        scale_passes=8,
        continuous_candidates=False,
        one_bit_passes=16,
        hard_fraction=1.0,
        component_passes=0,
    )
    one_bit = refine_binary_factors_separable(**common, pair_passes=0, block_bits=0)
    block = refine_binary_factors_separable(**common, pair_passes=2, block_bits=4, block_passes=2)

    assert block.after_error <= one_bit.after_error + 1e-6


def test_direct_binary_search_substantially_improves_a_represented_target() -> None:
    generator = torch.Generator().manual_seed(8)
    true_left = torch.randint(0, 2, (3, 3), generator=generator).float().mul_(2).sub_(1)
    true_right = torch.randint(0, 2, (3, 3), generator=generator).float().mul_(2).sub_(1)
    target = reconstruct(
        true_left,
        true_right,
        torch.tensor([0.7, 1.2, 0.9]),
        torch.tensor([1.1, 0.6, 1.4]),
        torch.tensor([0.8, 1.3, 1.0]),
    )
    initial_left = torch.randint(0, 2, (3, 3), generator=generator).float().mul_(2).sub_(1)
    initial_right = torch.randint(0, 2, (3, 3), generator=generator).float().mul_(2).sub_(1)

    result = refine_binary_factors_separable(
        target,
        initial_left,
        initial_right,
        torch.ones(3),
        torch.ones(3),
        torch.ones(3),
        torch.ones(3),
        torch.ones(3),
        outer_passes=12,
        scale_passes=32,
        one_bit_passes=16,
        pair_passes=4,
        pair_pool_size=3,
        block_bits=3,
        block_passes=4,
        hard_fraction=1.0,
    )

    assert result.after_error < result.before_error * 0.1


def test_component_replacement_is_monotonic_and_bounded() -> None:
    target, left, right, pre, mid, post = _random_problem(4, 44)

    result = refine_binary_factors_separable(
        target,
        left,
        right,
        pre,
        mid,
        post,
        torch.ones(4),
        torch.ones(4),
        outer_passes=4,
        scale_passes=8,
        continuous_candidates=False,
        one_bit_passes=0,
        pair_passes=0,
        block_bits=0,
        component_passes=2,
        component_limit=2,
        component_alternating_steps=8,
    )

    assert result.after_error <= result.before_error
    assert result.component_updates <= 4 * 2 * 2


def test_joint_window_exhausts_the_gauge_reduced_three_by_three_signs() -> None:
    target, left, right, pre, mid, post = _random_problem(3, 81)

    result = refine_binary_factors_separable(
        target,
        left,
        right,
        pre,
        mid,
        post,
        torch.ones(3),
        torch.ones(3),
        outer_passes=1,
        scale_passes=16,
        continuous_candidates=False,
        one_bit_passes=0,
        pair_passes=0,
        block_bits=0,
        component_passes=0,
        joint_passes=1,
        joint_bits=10,
        joint_candidate_refits=8,
    )

    assert result.after_error <= result.before_error
    assert result.joint_patterns_evaluated == 1024


def test_direct_binary_search_rejects_an_unbounded_block() -> None:
    target, left, right, pre, mid, post = _random_problem(3, 1)

    with pytest.raises(ValueError, match="settings"):
        refine_binary_factors_separable(
            target,
            left,
            right,
            pre,
            mid,
            post,
            torch.ones(3),
            torch.ones(3),
            block_bits=17,
        )

    with pytest.raises(ValueError, match="settings"):
        refine_binary_factors_separable(
            target,
            left,
            right,
            pre,
            mid,
            post,
            torch.ones(3),
            torch.ones(3),
            joint_bits=21,
        )
