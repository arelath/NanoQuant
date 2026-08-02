import torch

from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.scale_fit import reconstruct
from tools.probe_cyclic_scale_rank import (
    cyclic_reconstruct,
    cyclic_scale_bit_cost,
    fit_cyclic_scales,
    maximum_equal_bit_rank,
)


def test_scale_rank_one_exactly_matches_current_reconstruction() -> None:
    generator = torch.Generator().manual_seed(4)
    left = torch.where(torch.rand(5, 4, generator=generator) > 0.5, 1.0, -1.0)
    right = torch.where(torch.rand(4, 6, generator=generator) > 0.5, 1.0, -1.0)
    pre = torch.rand(6, generator=generator)
    mid = torch.rand(4, generator=generator)
    post = torch.rand(5, generator=generator)

    actual = cyclic_reconstruct(left, right, pre[None, :], mid, post[:, None])

    torch.testing.assert_close(actual, reconstruct(left, right, pre, mid, post))


def test_cyclic_scale_cost_charges_every_additional_bank() -> None:
    base = factor_bit_cost(256, 1152, 128, scale_bits=16, rank_alignment=32).total

    assert cyclic_scale_bit_cost(256, 1152, 128, 1) == base
    assert cyclic_scale_bit_cost(256, 1152, 128, 3) == base + 2 * (256 + 1152) * 16
    assert maximum_equal_bit_rank(256, 1152, base, 1) == 128
    assert maximum_equal_bit_rank(256, 1152, base, 3) == 96


def test_cyclic_scale_fit_can_recover_group_specific_magnitudes() -> None:
    generator = torch.Generator().manual_seed(9)
    left = torch.where(torch.rand(8, 6, generator=generator) > 0.5, 1.0, -1.0)
    right = torch.where(torch.rand(6, 9, generator=generator) > 0.5, 1.0, -1.0)
    mid = torch.rand(6, generator=generator) + 0.25
    true_pre = torch.rand(2, 9, generator=generator) + 0.5
    true_post = torch.rand(8, 2, generator=generator) + 0.5
    target = cyclic_reconstruct(left, right, true_pre, mid, true_post)
    initial_pre = true_pre.mean(dim=0)
    initial_post = true_post.mean(dim=1)

    result = fit_cyclic_scales(
        target,
        left,
        right,
        initial_pre,
        mid,
        initial_post,
        torch.ones(9),
        torch.ones(8),
        2,
        alternating_passes=6,
    )

    assert result.accepted
    assert result.after_error < result.before_error * 0.1
