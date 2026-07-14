"""Measure V3 event routing against the pre-refactor JSONL write shape."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from nanoquant.infrastructure.device_memory import sample_device_memory
from nanoquant.infrastructure.events import (
    ConsoleEventDestination,
    EventRouter,
    JsonlEventDestination,
)
from nanoquant.ports.event_sink import Event, Severity


class LegacyLikeSink:
    """The former sink's envelope, lock, serialization, write, and flush path."""

    def __init__(self, path: Path) -> None:
        self.handle = path.open("a", encoding="utf-8", newline="\n")
        self.lock = threading.Lock()
        self.sequence = 0

    def emit(self, index: int) -> None:
        with self.lock:
            self.sequence += 1
            event = Event(
                1,
                datetime.now(timezone.utc).isoformat(),
                "benchmark",
                self.sequence,
                "factorize-attempt",
                "info",
                "factorization.retry_decision",
                _fields(index),
            )
            self.handle.write(json.dumps(asdict(event), sort_keys=True, separators=(",", ":"), default=str) + "\n")
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _fields(index: int) -> dict[str, object]:
    return {
        "layer": "block.7.mlp.down_proj",
        "attempt": index % 3,
        "rank": 512,
        "weighted_error": 0.3125,
        "raw_error": 0.2875,
        "action": "accept",
        "reason": "threshold satisfied",
    }


def _legacy(path: Path, count: int) -> float:
    started = time.perf_counter()
    sink = LegacyLikeSink(path)
    for index in range(count):
        sink.emit(index)
    sink.close()
    return time.perf_counter() - started


def _router(path: Path, count: int, *, console: bool) -> float:
    destinations = [JsonlEventDestination(path, Severity.INFO)]
    if console:
        destinations.append(ConsoleEventDestination(Severity.INFO, io.StringIO()))
    router = EventRouter(
        "benchmark",
        event_level=Severity.INFO,
        destinations=tuple(destinations),
    )
    started = time.perf_counter()
    for index in range(count):
        router.emit("factorize-attempt", "info", "factorization.retry_decision", **_fields(index))
    router.close()
    return time.perf_counter() - started


def _meters(count: int) -> float:
    started = time.perf_counter()
    for _index in range(count):
        sample_device_memory()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.events <= 0 or args.repeats <= 0:
        raise ValueError("events and repeats must be positive")

    measurements: dict[str, list[float]] = {
        "legacy_like": [],
        "router_jsonl": [],
        "router_console": [],
        "resource_meter": [],
    }
    with tempfile.TemporaryDirectory(prefix="nanoquant-event-benchmark-") as directory:
        root = Path(directory)
        for repeat in range(args.repeats):
            measurements["legacy_like"].append(_legacy(root / f"legacy-{repeat}.jsonl", args.events))
            measurements["router_jsonl"].append(_router(root / f"router-{repeat}.jsonl", args.events, console=False))
            measurements["router_console"].append(
                _router(root / f"console-{repeat}.jsonl", args.events, console=True)
            )
            measurements["resource_meter"].append(_meters(args.events))

    medians = {name: statistics.median(values) for name, values in measurements.items()}
    baseline = medians["legacy_like"]
    meter_microseconds = medians["resource_meter"] * 1_000_000 / args.events
    print(
        json.dumps(
            {
                "events": args.events,
                "repeats": args.repeats,
                "median_seconds": medians,
                "microseconds_per_event": {
                    name: seconds * 1_000_000 / args.events for name, seconds in medians.items()
                },
                "ratio_to_legacy_like": {name: seconds / baseline for name, seconds in medians.items()},
                "resource_sampler": {
                    "microseconds_per_sample": meter_microseconds,
                    "estimated_default_cpu_fraction": meter_microseconds / 5_000_000,
                    "cuda_stream_synchronizations": 0,
                },
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
