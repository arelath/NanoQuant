from __future__ import annotations

from unittest.mock import patch

import pytest

from nanoquant.runtime import benchmark_wall, summarize_benchmark


def test_runtime_benchmark_summary_retains_raw_samples_and_tail_statistics() -> None:
    summary = summarize_benchmark(
        (4.0, 1.0, 3.0, 2.0),
        unit_name="tokens",
        units_per_sample=8,
    )

    assert summary.samples_seconds == (4.0, 1.0, 3.0, 2.0)
    assert summary.latency_seconds == pytest.approx(
        {
            "min": 1.0,
            "p10": 1.3,
            "p50": 2.5,
            "p90": 3.7,
            "p99": 3.97,
            "max": 4.0,
            "mean": 2.5,
        }
    )
    assert summary.throughput_per_second["min"] == 2.0
    assert summary.throughput_per_second["p50"] == pytest.approx(10 / 3)
    assert summary.throughput_per_second["max"] == 8.0
    assert summary.as_dict()["unit_name"] == "tokens"


@pytest.mark.parametrize(
    ("samples", "unit_name", "units"),
    [
        ((), "tokens", 1),
        ((0.0,), "tokens", 1),
        ((float("nan"),), "tokens", 1),
        ((1.0,), "", 1),
        ((1.0,), "tokens", 0),
    ],
)
def test_runtime_benchmark_summary_rejects_ambiguous_inputs(
    samples: tuple[float, ...], unit_name: str, units: int
) -> None:
    with pytest.raises(ValueError):
        summarize_benchmark(samples, unit_name=unit_name, units_per_sample=units)


def test_wall_benchmark_excludes_warmups_and_records_requested_repetitions() -> None:
    calls: list[str] = []

    def operation() -> None:
        calls.append("operation")

    def synchronize() -> None:
        calls.append("synchronize")

    with patch(
        "nanoquant.runtime.benchmark.time.perf_counter",
        side_effect=(1.0, 1.1, 2.0, 2.2),
    ):
        result = benchmark_wall(
            operation,
            warmups=2,
            repetitions=2,
            synchronize=synchronize,
            unit_name="tokens",
            units_per_sample=4,
        )

    assert result.samples_seconds == pytest.approx((0.1, 0.2))
    assert calls.count("operation") == 4
    assert calls.count("synchronize") == 5


def test_wall_benchmark_rejects_invalid_iteration_counts() -> None:
    with pytest.raises(ValueError):
        benchmark_wall(lambda: None, warmups=-1, repetitions=1, unit_name="x", units_per_sample=1)
    with pytest.raises(ValueError):
        benchmark_wall(lambda: None, warmups=0, repetitions=0, unit_name="x", units_per_sample=1)
