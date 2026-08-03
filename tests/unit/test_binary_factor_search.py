from __future__ import annotations

import pytest
import torch

import nanoquant.domain.binary_factor_search as binary_factor_search
from nanoquant.domain.binary_factor_search import (
    _component_candidate_order,
    _hard_vector_indices,
    _one_bit_pass,
    _scores,
    _variable_depth_pass,
    refine_binary_factors_separable,
)
from nanoquant.domain.scale_fit import fit_scales, reconstruct


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


def test_shared_codebook_transfers_a_better_pattern_between_rows() -> None:
    right = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    true_left = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    initial_left = torch.tensor([[1.0, 1.0], [1.0, -1.0], [1.0, -1.0]])
    target = true_left @ right

    result = refine_binary_factors_separable(
        target,
        initial_left,
        right,
        torch.ones(2),
        torch.ones(2),
        torch.ones(3),
        torch.ones(2),
        torch.ones(3),
        outer_passes=2,
        scale_passes=4,
        continuous_candidates=False,
        one_bit_passes=0,
        codebook_passes=2,
        codebook_size=8,
        pair_passes=0,
        block_bits=0,
        component_passes=0,
    )

    assert result.codebook_updates > 0
    assert result.after_error < result.before_error * 0.01


def test_variable_depth_chain_crosses_a_two_bit_barrier() -> None:
    generator = torch.Generator().manual_seed(0)
    design = torch.randn((6, 6), generator=generator)
    gram = design @ design.mT
    cross = torch.randn((1, 6), generator=generator) * 2
    vectors = torch.randint(0, 2, (1, 6), generator=generator).float().mul_(2).sub_(1)
    scores, alpha, beta, _ = _scores(vectors, cross, gram, 1e-8)
    scales = alpha / beta
    for _ in range(20):
        vectors, scales, scores, updates = _one_bit_pass(
            vectors, cross, gram, scales, scores, 1e-8, 1e-10
        )
        if updates == 0:
            break
    local_score = float(scores[0])
    one_bit_local = vectors.clone()

    refined, _scales, refined_scores, updates = _variable_depth_pass(
        vectors,
        cross,
        gram,
        scales,
        scores,
        6,
        1e-8,
        1e-10,
    )

    assert updates == 1
    assert float(refined_scores[0]) > local_score
    assert int((refined != one_bit_local).sum()) == 2


def test_one_bit_pass_honors_the_highest_gain_update_cap() -> None:
    vectors = torch.ones((2, 2))
    cross = torch.tensor([[2.0, -1.0], [1.0, -0.2]])
    gram = torch.eye(2)
    scores, alpha, beta, _ = _scores(vectors, cross, gram, 1e-8)
    scales = alpha / beta

    refined, _scales, refined_scores, updates = _one_bit_pass(
        vectors,
        cross,
        gram,
        scales,
        scores,
        1e-8,
        1e-10,
        maximum_updates=1,
    )

    assert updates == 1
    assert int((refined[0] != vectors[0]).sum()) == 1
    assert torch.equal(refined[1], vectors[1])
    assert float(refined_scores.sum()) > float(scores.sum())


def test_hard_vectors_use_actual_weighted_residual_not_relative_explained_energy() -> None:
    scores = torch.tensor([1.0, 9.0, 5.0])
    vector_weights = torch.tensor([1.0, 2.0, 0.5])
    target_energy = torch.tensor([1.1, 20.0, 30.0])

    selected = _hard_vector_indices(scores, vector_weights, target_energy, 2)

    assert selected.tolist() == [1, 2]


def test_component_candidates_interleave_weak_strong_and_residual_aligned_pools() -> None:
    mid = torch.tensor([0.1, 0.5, 3.0, 1.0])
    removal_cost = torch.tensor([-2.0, 4.0, 6.0, 3.0])
    residual_alignment = torch.tensor([0.2, 9.0, 1.0, 0.5])

    selected = _component_candidate_order(mid, removal_cost, residual_alignment, 3)

    assert selected.tolist() == [0, 2, 1]


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


def test_joint_window_keeps_scale_profiled_screening_when_batches_are_forced_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, left, right, pre, mid, post = _random_problem(3, 91)
    observed_batch_sizes: list[int] = []
    original = binary_factor_search._joint_scale_screen_batch

    def observe(*args: object, **kwargs: object) -> torch.Tensor:
        candidate_left = args[1]
        assert isinstance(candidate_left, torch.Tensor)
        observed_batch_sizes.append(candidate_left.shape[0])
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(binary_factor_search, "_JOINT_SCALE_SCREEN_BATCH_ELEMENTS", 18)
    monkeypatch.setattr(binary_factor_search, "_joint_scale_screen_batch", observe)

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
        scale_passes=8,
        continuous_candidates=False,
        one_bit_passes=0,
        pair_passes=0,
        block_bits=0,
        component_passes=0,
        joint_passes=1,
        joint_bits=10,
        joint_candidate_refits=4,
        joint_batch_size=64,
        joint_screen_scale_passes=2,
    )

    assert result.joint_patterns_evaluated == 1024
    assert sum(observed_batch_sizes) == 1024
    assert max(observed_batch_sizes) == 2


def test_joint_scale_screen_matches_the_full_scale_refit_ranking() -> None:
    target, left, right, pre, mid, post = _random_problem(3, 101)
    input_importance = torch.tensor([0.7, 1.1, 1.8])
    output_importance = torch.tensor([1.4, 0.6, 1.2])
    candidate_left = left.expand(4, -1, -1).clone()
    candidate_right = right.expand(4, -1, -1).clone()
    candidate_left[1, 1, 0] *= -1
    candidate_right[2, 1, 1] *= -1
    candidate_left[3, 2, 1] *= -1
    candidate_right[3, 2, 2] *= -1

    screened = binary_factor_search._joint_scale_screen_batch(
        target,
        candidate_left,
        candidate_right,
        pre,
        mid,
        post,
        input_importance,
        output_importance,
        8,
        1e-8,
    )
    refitted = torch.tensor(
        [
            fit_scales(
                target,
                candidate_left[index],
                candidate_right[index],
                pre,
                mid,
                post,
                input_importance,
                output_importance,
                alternating_passes=8,
            ).after_error
            for index in range(candidate_left.shape[0])
        ]
    )

    assert torch.allclose(screened, refitted, rtol=1e-5, atol=1e-6)
    assert int(screened.argmin()) == int(refitted.argmin())


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
