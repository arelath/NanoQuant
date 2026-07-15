from __future__ import annotations

import pytest

from nanoquant.runtime import profile_ratio_samples, sum_aligned_profile_samples


def test_profile_accounting_sums_aligned_non_overlapping_regions() -> None:
    regions = {
        "block-0": (1.0, 2.0, 3.0),
        "block-1": (4.0, 5.0, 6.0),
        "head": (0.5, 0.25, 0.125),
    }

    accounted = sum_aligned_profile_samples(regions, ("block-0", "block-1", "head"))

    assert accounted == pytest.approx((5.5, 7.25, 9.125))
    assert profile_ratio_samples(accounted, (10.0, 10.0, 10.0)) == pytest.approx(
        (0.55, 0.725, 0.9125)
    )


@pytest.mark.parametrize(
    ("regions", "names", "message"),
    [
        ({}, (), "non-empty"),
        ({"a": (1.0,)}, ("a", "a"), "unique"),
        ({"a": (1.0,)}, ("missing",), "missing"),
        ({"a": (1.0,), "b": (1.0, 2.0)}, ("a", "b"), "counts differ"),
        ({"a": (-1.0,)}, ("a",), "invalid"),
    ],
)
def test_profile_accounting_rejects_unaligned_or_invalid_regions(
    regions: dict[str, tuple[float, ...]], names: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sum_aligned_profile_samples(regions, names)


def test_profile_ratios_require_aligned_positive_denominators() -> None:
    with pytest.raises(ValueError, match="counts differ"):
        profile_ratio_samples((1.0,), (1.0, 2.0))
    with pytest.raises(ValueError, match="positive"):
        profile_ratio_samples((1.0,), (0.0,))
