"""Stable statistics and timing contracts for deployment runtime benchmarks."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("runtime benchmark samples must be non-empty")
    if not 0 <= quantile <= 1:
        raise ValueError("runtime benchmark quantile must be in [0, 1]")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


@dataclass(frozen=True, slots=True)
class BenchmarkDistribution:
    """Raw timing samples and auditable latency/throughput summaries."""

    samples_seconds: tuple[float, ...]
    unit_name: str
    units_per_sample: int
    latency_seconds: dict[str, float]
    throughput_per_second: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples_seconds": list(self.samples_seconds),
            "unit_name": self.unit_name,
            "units_per_sample": self.units_per_sample,
            "latency_seconds": dict(self.latency_seconds),
            "throughput_per_second": dict(self.throughput_per_second),
        }


def summarize_benchmark(
    samples_seconds: Sequence[float],
    *,
    unit_name: str,
    units_per_sample: int,
) -> BenchmarkDistribution:
    """Summarize positive samples without discarding their original order."""

    samples = tuple(float(value) for value in samples_seconds)
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("runtime benchmark samples must be finite and positive")
    if not unit_name:
        raise ValueError("runtime benchmark unit name must be non-empty")
    if units_per_sample <= 0:
        raise ValueError("runtime benchmark units per sample must be positive")
    ordered = tuple(sorted(samples))
    rates = tuple(sorted(units_per_sample / value for value in samples))

    def distribution(values: Sequence[float]) -> dict[str, float]:
        return {
            "min": float(values[0]),
            "p10": _percentile(values, 0.10),
            "p50": _percentile(values, 0.50),
            "p90": _percentile(values, 0.90),
            "p99": _percentile(values, 0.99),
            "max": float(values[-1]),
            "mean": float(sum(values) / len(values)),
        }

    return BenchmarkDistribution(
        samples,
        unit_name,
        units_per_sample,
        distribution(ordered),
        distribution(rates),
    )


def benchmark_wall(
    operation: Callable[[], object],
    *,
    warmups: int,
    repetitions: int,
    synchronize: Callable[[], object] | None = None,
    unit_name: str,
    units_per_sample: int,
) -> BenchmarkDistribution:
    """Measure an end-to-end callable with synchronization outside each timed region."""

    if warmups < 0 or repetitions <= 0:
        raise ValueError("runtime benchmark warmups must be non-negative and repetitions positive")
    sync = synchronize or (lambda: None)
    for _ in range(warmups):
        operation()
    sync()
    samples = []
    for _ in range(repetitions):
        sync()
        started = time.perf_counter()
        operation()
        sync()
        samples.append(time.perf_counter() - started)
    return summarize_benchmark(
        samples,
        unit_name=unit_name,
        units_per_sample=units_per_sample,
    )
