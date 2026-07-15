"""Sample alignment and accounting helpers for nested runtime profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _samples(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"runtime profile region {name!r} has no samples")
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError(f"runtime profile region {name!r} has invalid samples")
    return result


def sum_aligned_profile_samples(
    regions: Mapping[str, Sequence[float]],
    names: Sequence[str],
) -> tuple[float, ...]:
    """Sum non-overlapping regions sample-by-sample after strict alignment checks."""

    selected = tuple(names)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("runtime profile region names must be non-empty and unique")
    missing = [name for name in selected if name not in regions]
    if missing:
        raise ValueError(f"runtime profile regions are missing: {missing}")
    aligned = tuple(_samples(regions[name], name) for name in selected)
    lengths = {len(values) for values in aligned}
    if len(lengths) != 1:
        raise ValueError("runtime profile region sample counts differ")
    return tuple(sum(values) for values in zip(*aligned, strict=True))


def profile_ratio_samples(
    numerator: Sequence[float],
    denominator: Sequence[float],
) -> tuple[float, ...]:
    """Return aligned accounting ratios, rejecting zero or inconsistent denominators."""

    upper = _samples(numerator, "numerator")
    lower = _samples(denominator, "denominator")
    if len(upper) != len(lower):
        raise ValueError("runtime profile ratio sample counts differ")
    if any(value <= 0 for value in lower):
        raise ValueError("runtime profile ratio denominators must be positive")
    return tuple(left / right for left, right in zip(upper, lower, strict=True))
