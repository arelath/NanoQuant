from __future__ import annotations

import pytest
import torch

import tools.probe_tiny_factorization_optimality as tiny_optimality
from tools.probe_tiny_factorization_optimality import (
    PopulationFit,
    _canonical_population_bits,
    _diverse_elite,
    exhaustive_row_column_descent,
    exhaustive_sign_oracle,
    fit_scale_population,
    gauge_reduced_sign_pair_count,
    gauge_reduced_sign_pair_range,
    gauge_reduced_sign_pairs,
)


def test_gauge_reduced_three_by_three_enumerates_expected_sign_classes() -> None:
    left, right = gauge_reduced_sign_pairs(3)

    assert left.shape == right.shape == (1024, 3, 3)
    assert torch.equal(left[:, 0, :], torch.ones(1024, 3))
    assert torch.equal(left[:, :, 0], torch.ones(1024, 3))
    assert torch.equal(right[:, 0, :], torch.ones(1024, 3))
    assert torch.unique(torch.cat((left.reshape(-1), right.reshape(-1)))).tolist() == [-1.0, 1.0]


def test_gauge_reduced_ranges_partition_full_enumeration() -> None:
    full_left, full_right = gauge_reduced_sign_pairs(3)
    first_left, first_right = gauge_reduced_sign_pair_range(3, 0, 413)
    second_left, second_right = gauge_reduced_sign_pair_range(3, 413, 1024)

    assert gauge_reduced_sign_pair_count(4) == 2_097_152
    assert gauge_reduced_sign_pair_count(5) == 68_719_476_736
    torch.testing.assert_close(torch.cat((first_left, second_left)), full_left)
    torch.testing.assert_close(torch.cat((first_right, second_right)), full_right)


def test_low_rank_gauge_reduction_scales_with_rank_instead_of_matrix_size() -> None:
    left, right = gauge_reduced_sign_pair_range(5, 0, 13, columns=5, rank=3)

    assert gauge_reduced_sign_pair_count(5, 5, 3) == 262_144
    assert gauge_reduced_sign_pair_count(6, 6, 3) == 4_194_304
    assert gauge_reduced_sign_pair_count(10, 10, 2) == 524_288
    assert left.shape == (13, 5, 3)
    assert right.shape == (13, 3, 5)
    assert torch.equal(left[:, 0, :], torch.ones(13, 3))
    assert torch.equal(left[:, :, 0], torch.ones(13, 5))
    assert torch.equal(right[:, 0, :], torch.ones(13, 5))


def test_exhaustive_oracle_batching_preserves_complete_one_start_result() -> None:
    target = torch.tensor([[0.2, -0.7, 1.1], [0.9, 0.3, -0.4], [-0.5, 0.8, 0.6]])
    arguments = (target, torch.ones(3), torch.ones(3))
    unbatched, unbatched_count = exhaustive_sign_oracle(
        *arguments,
        starts=1,
        passes=4,
        seed=5,
        device="cpu",
        batch_size=1024,
    )
    batched, batched_count = exhaustive_sign_oracle(
        *arguments,
        starts=1,
        passes=4,
        seed=5,
        device="cpu",
        batch_size=137,
    )

    assert unbatched_count == batched_count == 1024
    torch.testing.assert_close(unbatched.errors, batched.errors)


def test_population_scale_fit_recovers_representable_target_when_given_true_signs() -> None:
    left = torch.tensor([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])
    right = torch.tensor([[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]])
    pre = torch.tensor([0.7, 1.2, 0.9])
    mid = torch.tensor([1.1, 0.6, 1.4])
    post = torch.tensor([0.8, 1.3, 1.0])
    target = (left * post[:, None]) @ (right * mid[:, None] * pre[None, :])

    fitted = fit_scale_population(
        target,
        left[None],
        right[None],
        torch.ones(3),
        torch.ones(3),
        starts=8,
        passes=64,
        seed=4,
    )

    assert float(fitted.errors.min()) < 1e-10


def test_population_scale_fit_supports_rectangular_low_rank_factors() -> None:
    left = torch.tensor([[1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    right = torch.tensor([[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]])
    target = (left * torch.tensor([0.8, 1.3, 0.9, 1.1])[:, None]) @ (
        right * torch.tensor([1.2, 0.7])[:, None] * torch.tensor([0.7, 1.2, 0.9])[None, :]
    )

    fitted = fit_scale_population(
        target,
        left[None],
        right[None],
        torch.ones(3),
        torch.ones(4),
        starts=8,
        passes=64,
        seed=4,
    )

    assert float(fitted.errors.min()) < 1e-10


def test_exhaustive_row_column_descent_never_regresses_candidate() -> None:
    generator = torch.Generator().manual_seed(12)
    target = torch.randn((4, 4), generator=generator)
    left = torch.randint(0, 2, (1, 4, 4), generator=generator).float().mul_(2).sub_(1)
    right = torch.randint(0, 2, (1, 4, 4), generator=generator).float().mul_(2).sub_(1)
    initial = fit_scale_population(
        target,
        left,
        right,
        torch.ones(4),
        torch.ones(4),
        starts=1,
        passes=8,
        seed=3,
    )
    candidate = PopulationFit(
        initial.left,
        initial.right,
        initial.pre,
        initial.mid,
        initial.post,
        initial.errors,
    )

    refined = exhaustive_row_column_descent(
        target,
        torch.ones(4),
        torch.ones(4),
        candidate,
        sweeps=8,
        scale_passes=8,
    )

    assert float(refined.errors[0]) <= float(initial.errors[0])


def test_population_scale_fit_chunking_preserves_one_start_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(22)
    target = torch.randn((4, 4), generator=generator)
    left = torch.randint(0, 2, (7, 4, 4), generator=generator).float().mul_(2).sub_(1)
    right = torch.randint(0, 2, (7, 4, 4), generator=generator).float().mul_(2).sub_(1)
    arguments = (target, left, right, torch.ones(4), torch.ones(4))
    unbatched = fit_scale_population(*arguments, starts=1, passes=4, seed=7)
    monkeypatch.setattr(tiny_optimality, "_POPULATION_SCALE_SYSTEM_ELEMENTS", 32)

    batched = fit_scale_population(*arguments, starts=1, passes=4, seed=7)

    torch.testing.assert_close(batched.errors, unbatched.errors)
    torch.testing.assert_close(batched.mid, unbatched.mid)


def test_diverse_elite_keeps_quality_and_canonical_novelty() -> None:
    generator = torch.Generator().manual_seed(31)
    left = torch.randint(0, 2, (8, 4, 4), generator=generator).float().mul_(2).sub_(1)
    right = torch.randint(0, 2, (8, 4, 4), generator=generator).float().mul_(2).sub_(1)
    population = PopulationFit(
        left,
        right,
        torch.ones((8, 4)),
        torch.ones((8, 4)),
        torch.ones((8, 4)),
        torch.arange(8, dtype=torch.float32),
    )

    elite = _diverse_elite(population, 4, quality_fraction=0.5, pool_multiplier=2)

    assert elite.errors[:2].tolist() == [0.0, 1.0]
    bits = _canonical_population_bits(elite.left, elite.right)
    assert torch.unique(bits, dim=0).shape[0] == 4
