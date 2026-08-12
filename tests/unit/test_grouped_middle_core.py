from __future__ import annotations

import torch

from nanoquant.domain.grouped_middle_core import (
    fit_equal_rate_grouped_middle_core,
    grouped_dense_reconstruction,
    grouped_rank_at_or_below_diagonal_rate,
)


def test_grouped_rank_is_complete_and_no_larger_than_diagonal_rate() -> None:
    rank, diagonal_bits, grouped_bits = grouped_rank_at_or_below_diagonal_rate(
        960,
        6912,
        1152,
    )

    assert rank == 958
    assert rank % 2 == 0
    assert grouped_bits <= diagonal_bits
    assert (rank + 2) * (6912 + 1152 + 32) > diagonal_bits


def test_grouped_core_recovers_cross_component_interaction_at_equal_rate() -> None:
    generator = torch.Generator().manual_seed(17)
    left = torch.sign(torch.randn((12, 6), generator=generator))
    right = torch.sign(torch.randn((6, 10), generator=generator))
    pre = torch.linspace(0.7, 1.3, 10)
    post = torch.linspace(0.8, 1.2, 12)
    mid = torch.tensor([2.0, 1.8, 1.5, 1.2, 0.01, 0.005])
    # Equal-rate grouped rank is four. Inject an off-diagonal interaction that
    # no diagonal middle scale can express with these fixed factors.
    core = torch.diag(mid[:4])
    core[0, 1] = 0.75
    core[1, 0] = -0.45
    target = grouped_dense_reconstruction(left[:, :4], right[:4], pre, core, post)

    result = fit_equal_rate_grouped_middle_core(
        target,
        left,
        right,
        pre,
        mid,
        post,
        torch.ones(10),
        torch.ones(12),
        alternating_passes=2,
        storage_dtype=torch.float32,
    )

    assert result.grouped_bits <= result.diagonal_bits
    assert result.accepted
    assert result.after_error < result.before_error * 1e-5
    torch.testing.assert_close(result.reconstruction, target, atol=2e-3, rtol=2e-3)


def test_residual_pairing_finds_cross_component_pair() -> None:
    generator = torch.Generator().manual_seed(31)
    left = torch.sign(torch.randn((14, 6), generator=generator))
    right = torch.sign(torch.randn((6, 11), generator=generator))
    pre = torch.ones(11)
    post = torch.ones(14)
    mid = torch.tensor([3.0, 2.5, 2.0, 1.5, 0.01, 0.005])
    core = torch.diag(mid[:4])
    core[0, 3] = 1.25
    core[3, 0] = -0.8
    target = grouped_dense_reconstruction(left[:, :4], right[:4], pre, core, post)

    result = fit_equal_rate_grouped_middle_core(
        target,
        left,
        right,
        pre,
        mid,
        post,
        torch.ones(11),
        torch.ones(14),
        pairing="residual",
        storage_dtype=torch.float32,
    )

    paired = result.component_indices.reshape(-1, 2).tolist()
    assert [0, 3] in paired or [3, 0] in paired
    assert result.after_error < result.before_error * 1e-5


def test_grouped_core_preserves_protected_columns() -> None:
    generator = torch.Generator().manual_seed(23)
    left = torch.sign(torch.randn((8, 6), generator=generator))
    right = torch.sign(torch.randn((6, 7), generator=generator))
    target = torch.randn((8, 7), generator=generator)

    result = fit_equal_rate_grouped_middle_core(
        target,
        left,
        right,
        torch.ones(7),
        torch.linspace(1.0, 0.1, 6),
        torch.ones(8),
        torch.ones(7),
        torch.ones(8),
        protected_columns=torch.tensor([2, 5]),
        storage_dtype=torch.float32,
    )

    assert torch.count_nonzero(result.reconstruction[:, [2, 5]]) == 0
