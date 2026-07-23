from __future__ import annotations

import torch

from nanoquant.application.low_rank_patch import fit_low_rank_patch_family


def test_patch_family_reuses_one_decomposition_and_improves_held_out_error() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(6, 3, generator=generator)
    right = torch.randn(3, 8, generator=generator)
    target = left @ right
    reconstruction = torch.zeros_like(target)
    covariance = torch.eye(8)
    mean = torch.zeros(8)

    candidates = fit_low_rank_patch_family(
        target,
        reconstruction,
        covariance,
        covariance,
        mean,
        mean,
        ranks=(1, 2, 3),
        ridge_fraction=1e-4,
        storage_dtype=torch.float16,
    )

    assert tuple(candidate.rank for candidate in candidates) == (1, 2, 3)
    assert all(candidate.accepted for candidate in candidates)
    assert all(
        candidate.held_out_error_after < candidate.held_out_error_before
        for candidate in candidates
    )
    assert candidates[2].held_out_error_after < candidates[1].held_out_error_after
    assert candidates[1].held_out_error_after < candidates[0].held_out_error_after


def test_patch_family_rejects_invalid_rank_inventory() -> None:
    value = torch.eye(2)
    mean = torch.zeros(2)

    try:
        fit_low_rank_patch_family(
            value,
            torch.zeros_like(value),
            value,
            value,
            mean,
            mean,
            ranks=(2, 1),
            ridge_fraction=1e-2,
            storage_dtype=torch.float16,
        )
    except ValueError as error:
        assert "unique, positive, and increasing" in str(error)
    else:
        raise AssertionError("unsorted patch ranks were accepted")
