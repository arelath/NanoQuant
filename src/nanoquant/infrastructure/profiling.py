"""Hierarchical in-process profiling and versioned profile artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import cast

import torch

from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
from nanoquant.ports.event_sink import EventSink

_PROFILE_SCHEMA_VERSION = 1
_MAX_MARKS = 256
_ATTRIBUTE_TYPES = (str, int, float, bool, type(None))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * fraction + 0.5)
    return ordered[index]


def _group_key(attributes: Mapping[str, object]) -> str:
    return "|".join(f"{name}={attributes[name]}" for name in sorted(attributes))


def _validate_attributes(attributes: Mapping[str, object]) -> None:
    invalid = [name for name, value in attributes.items() if not isinstance(value, _ATTRIBUTE_TYPES)]
    if invalid:
        raise TypeError(f"profiling attributes must be scalar: {', '.join(sorted(invalid))}")


@dataclass(slots=True)
class _GroupAggregate:
    count: int = 0
    wall_seconds: float = 0.0
    self_seconds: float = 0.0

    def record(self, elapsed: float, self_seconds: float) -> None:
        self.count += 1
        self.wall_seconds += elapsed
        self.self_seconds += self_seconds

    def payload(self) -> dict[str, object]:
        return {
            "count": self.count,
            "wall_seconds": self.wall_seconds,
            "self_seconds": self.self_seconds,
        }


@dataclass(slots=True)
class _PhaseAggregate:
    sample_limit: int
    count: int = 0
    failed_count: int = 0
    wall_seconds: float = 0.0
    self_seconds: float = 0.0
    minimum: float = float("inf")
    maximum: float = 0.0
    samples: list[float] = field(default_factory=list)
    self_samples: list[float] = field(default_factory=list)
    groups: dict[str, _GroupAggregate] = field(default_factory=dict)

    def record(self, elapsed: float, self_seconds: float, attributes: Mapping[str, object], failed: bool) -> None:
        self.count += 1
        self.failed_count += int(failed)
        self.wall_seconds += elapsed
        self.self_seconds += self_seconds
        self.minimum = min(self.minimum, elapsed)
        self.maximum = max(self.maximum, elapsed)
        if len(self.samples) < self.sample_limit:
            self.samples.append(elapsed)
            self.self_samples.append(self_seconds)
        else:
            # A deterministic ring preserves a bounded recent distribution without
            # consuming the application's random-number state.
            sample_index = (self.count - 1) % self.sample_limit
            self.samples[sample_index] = elapsed
            self.self_samples[sample_index] = self_seconds
        if attributes:
            self.groups.setdefault(_group_key(attributes), _GroupAggregate()).record(elapsed, self_seconds)

    def payload(self, path: str) -> dict[str, object]:
        return {
            "path": path,
            "count": self.count,
            "failed_count": self.failed_count,
            "wall_seconds": self.wall_seconds,
            "self_seconds": self.self_seconds,
            "unattributed_seconds": self.self_seconds,
            "min": 0.0 if self.count == 0 else self.minimum,
            "p50": _percentile(self.samples, 0.50),
            "p90": _percentile(self.samples, 0.90),
            "self_p50": _percentile(self.self_samples, 0.50),
            "self_p90": _percentile(self.self_samples, 0.90),
            "max": self.maximum,
            "groups": {name: group.payload() for name, group in sorted(self.groups.items())},
        }


@dataclass(slots=True)
class _CounterAggregate:
    total: float = 0.0
    by_phase: dict[str, float] = field(default_factory=dict)
    groups: dict[str, float] = field(default_factory=dict)

    def add(self, value: float, phase: str, attributes: Mapping[str, object]) -> None:
        self.total += value
        self.by_phase[phase] = self.by_phase.get(phase, 0.0) + value
        if attributes:
            key = _group_key(attributes)
            self.groups[key] = self.groups.get(key, 0.0) + value

    def payload(self, name: str) -> dict[str, object]:
        return {
            "name": name,
            "total": self.total,
            "by_phase": dict(sorted(self.by_phase.items())),
            "groups": dict(sorted(self.groups.items())),
        }


@dataclass(slots=True)
class _Frame:
    path: str
    started: float
    attributes: dict[str, object]
    span_id: str | None
    parent_span_id: str | None
    child_seconds: float = 0.0


class _MeasuredPhase(AbstractContextManager[None]):
    __slots__ = ("_attributes", "_frame", "_name", "_profiler")

    def __init__(self, profiler: Profiler, name: str, attributes: dict[str, object]) -> None:
        self._profiler = profiler
        self._name = name
        self._attributes = attributes
        self._frame: _Frame | None = None

    def __enter__(self) -> None:
        self._frame = self._profiler._enter(self._name, self._attributes)
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._frame is None:
            raise RuntimeError("profiling phase exited without entering")
        self._profiler._exit(self._frame, exc_value)
        return None


class Profiler:
    """Thread-confined O(1) aggregate profiler implementing ``PhaseRecorder``."""

    def __init__(
        self,
        config: ProfilingConfig,
        *,
        run_id: str,
        events: EventSink | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if config.level is ProfilingLevel.OFF:
            raise ValueError("disabled profiling must use NULL_RECORDER")
        if config.level is not ProfilingLevel.MACRO:
            raise NotImplementedError(f"profiling level {config.level.value!r} is not implemented yet")
        if config.cuda_timing:
            raise NotImplementedError("CUDA phase timing is not implemented yet")
        if config.raw_samples_per_phase <= 0:
            raise ValueError("raw_samples_per_phase must be positive")
        self.config = config
        self.run_id = run_id
        self._events = events
        self._clock = clock
        self._thread_id = threading.get_ident()
        self._process_started = _utc_now()
        self._started = self._now()
        self._finished: float | None = None
        self._stack: list[_Frame] = []
        self._phases: dict[str, _PhaseAggregate] = {}
        self._counters: dict[str, _CounterAggregate] = {}
        self._marks: list[dict[str, object]] = []
        self._top_level_seconds = 0.0
        self._recorder_seconds = 0.0
        self._span_sequence = 0

    def _now(self) -> float:
        return float(self._clock())

    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread_id:
            raise RuntimeError("Profiler is thread-confined")

    @staticmethod
    def _validate_name(name: str, kind: str) -> None:
        if not name or "." in name or name.lower() != name or not all(
            character.isalnum() or character == "_" for character in name
        ):
            raise ValueError(f"invalid profiling {kind} name: {name!r}")

    def phase(self, name: str, /, **attributes: object) -> AbstractContextManager[None]:
        self._validate_name(name, "phase")
        _validate_attributes(attributes)
        return _MeasuredPhase(self, name, dict(attributes))

    def _enter(self, name: str, attributes: dict[str, object]) -> _Frame:
        overhead_started = self._now()
        self._check_thread()
        parent = self._stack[-1] if self._stack else None
        path = f"{parent.path}/{name}" if parent is not None else name
        span_id = None
        parent_span_id = None if parent is None else parent.span_id
        if self.config.emit_span_events and self._events is not None:
            self._span_sequence += 1
            span_id = f"profile-{os.getpid()}-{self._span_sequence}"
            self._events.emit(
                "profile",
                "info",
                "phase.started",
                span_id=span_id,
                parent_span_id=parent_span_id,
                path=path,
                **attributes,
            )
        frame = _Frame(path, self._now(), attributes, span_id, parent_span_id)
        self._stack.append(frame)
        self._recorder_seconds += self._now() - overhead_started
        return frame

    def _exit(self, frame: _Frame, error: BaseException | None) -> None:
        ended = self._now()
        overhead_started = ended
        self._check_thread()
        if not self._stack or self._stack[-1] is not frame:
            raise RuntimeError("profiling phases must exit in stack order")
        self._stack.pop()
        elapsed = max(0.0, ended - frame.started)
        self_seconds = max(0.0, elapsed - frame.child_seconds)
        aggregate = self._phases.setdefault(frame.path, _PhaseAggregate(self.config.raw_samples_per_phase))
        aggregate.record(elapsed, self_seconds, frame.attributes, error is not None)
        if self._stack:
            self._stack[-1].child_seconds += elapsed
        else:
            self._top_level_seconds += elapsed
        if self.config.emit_span_events and self._events is not None:
            severity = "error" if error is not None else "info"
            fields: dict[str, object] = {
                "path": frame.path,
                "wall_seconds": elapsed,
                **frame.attributes,
            }
            if error is not None:
                fields.update(error_type=type(error).__name__, error=str(error))
            self._events.emit(
                "profile",
                severity,
                "phase.failed" if error is not None else "phase.completed",
                span_id=frame.span_id,
                parent_span_id=frame.parent_span_id,
                **fields,
            )
        self._recorder_seconds += self._now() - overhead_started

    def add(self, counter: str, value: float, /, **attributes: object) -> None:
        overhead_started = self._now()
        self._check_thread()
        if any(not segment for segment in counter.split(".")):
            raise ValueError(f"invalid profiling counter name: {counter!r}")
        for segment in counter.split("."):
            self._validate_name(segment, "counter")
        _validate_attributes(attributes)
        phase = self._stack[-1].path if self._stack else "unscoped"
        self._counters.setdefault(counter, _CounterAggregate()).add(float(value), phase, attributes)
        self._recorder_seconds += self._now() - overhead_started

    def mark(self, name: str, /, **attributes: object) -> None:
        overhead_started = self._now()
        self._check_thread()
        self._validate_name(name, "mark")
        _validate_attributes(attributes)
        if len(self._marks) < _MAX_MARKS:
            self._marks.append(
                {
                    "name": name,
                    "elapsed_seconds": self._now() - self._started,
                    "phase": self._stack[-1].path if self._stack else None,
                    "attributes": dict(sorted(attributes.items())),
                }
            )
        self._recorder_seconds += self._now() - overhead_started

    def finish(self) -> None:
        self._check_thread()
        if self._stack:
            raise RuntimeError("cannot finish a profile with open phases")
        if self._finished is None:
            self._finished = self._now()

    def snapshot(self) -> dict[str, object]:
        self.finish()
        if self._finished is None:
            raise AssertionError("finished profile has no end time")
        wall_total = max(0.0, self._finished - self._started)
        leaf_paths = [
            path
            for path in self._phases
            if path != "run" and not any(other.startswith(f"{path}/") for other in self._phases)
        ]
        measured = sum(self._phases[path].wall_seconds for path in leaf_paths)
        if not leaf_paths:
            measured = self._top_level_seconds
        attributed = min(wall_total, measured)
        coverage = attributed / wall_total if wall_total else 1.0
        recorder_fraction = self._recorder_seconds / wall_total if wall_total else 0.0
        warnings: list[dict[str, object]] = []
        if coverage < 0.90:
            warnings.append(
                {
                    "code": "PERF001",
                    "message": "profile coverage is below 90%",
                    "fraction": coverage,
                }
            )
        if recorder_fraction > 0.005:
            warnings.append(
                {
                    "code": "PERF002",
                    "message": "macro profiling recorder time exceeds 0.5%",
                    "fraction": recorder_fraction,
                }
            )
        environment = _environment_payload()
        return {
            "schema_version": _PROFILE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "process_id": os.getpid(),
            "process_started": self._process_started,
            "level": self.config.level.value,
            "environment": environment,
            "coverage": {
                "wall_total_seconds": wall_total,
                "attributed_seconds": attributed,
                "fraction": coverage,
            },
            "recorder_seconds": self._recorder_seconds,
            "recorder_fraction": recorder_fraction,
            "warnings": warnings,
            "phases": [aggregate.payload(path) for path, aggregate in sorted(self._phases.items())],
            "counters": [counter.payload(name) for name, counter in sorted(self._counters.items())],
            "marks": self._marks,
        }


def _environment_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    if bool(getattr(torch.cuda, "_initialized", False)):
        payload["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    payload["runtime_fingerprint"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return payload


class ProfileWriter:
    """Write a profiler snapshot and its human-readable summary atomically."""

    def __init__(self, output: str | Path) -> None:
        self.output = Path(output)

    def write(self, profiler: Profiler) -> tuple[Path, Path]:
        self.output.mkdir(parents=True, exist_ok=True)
        json_path, markdown_path = self._next_paths()
        payload = profiler.snapshot()
        json_temp = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
        markdown_temp = markdown_path.with_name(f".{markdown_path.name}.{os.getpid()}.tmp")
        json_temp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        markdown_temp.write_text(self.render_markdown(payload), encoding="utf-8")
        json_temp.replace(json_path)
        markdown_temp.replace(markdown_path)
        return json_path, markdown_path

    def _next_paths(self) -> tuple[Path, Path]:
        suffix = 1
        while True:
            marker = "" if suffix == 1 else f".{suffix}"
            json_path = self.output / f"profile{marker}.json"
            markdown_path = self.output / f"profile{marker}.md"
            if not json_path.exists() and not markdown_path.exists():
                return json_path, markdown_path
            suffix += 1

    @staticmethod
    def render_markdown(payload: Mapping[str, object]) -> str:
        coverage = payload["coverage"]
        if not isinstance(coverage, Mapping):
            raise TypeError("profile coverage must be an object")
        phases_value = payload["phases"]
        if not isinstance(phases_value, list):
            raise TypeError("profile phases must be an array")
        phases = [item for item in phases_value if isinstance(item, Mapping)]
        fraction = float(coverage["fraction"])
        wall_total = float(coverage["wall_total_seconds"])
        attributed = float(coverage["attributed_seconds"])
        lines = [
            "# Performance profile",
            "",
            f"- Run: `{payload['run_id']}`",
            f"- Level: `{payload['level']}`",
            f"- Coverage: {fraction:.2%} ({attributed:.3f}s of {wall_total:.3f}s)",
            f"- Recorder overhead: {cast(float, payload['recorder_seconds']):.6f}s",
            "",
            "## Top phases by self time",
            "",
            "| Phase | Count | Self seconds | Inclusive seconds | Failed |",
            "|---|---:|---:|---:|---:|",
        ]
        warnings_value = payload.get("warnings", [])
        if isinstance(warnings_value, list) and warnings_value:
            lines[6:6] = [
                "- Diagnostics: "
                + ", ".join(str(item.get("code")) for item in warnings_value if isinstance(item, Mapping))
            ]
        for phase in sorted(phases, key=lambda item: float(item["self_seconds"]), reverse=True)[:20]:
            lines.append(
                f"| `{phase['path']}` | {phase['count']} | {float(phase['self_seconds']):.6f} | "
                f"{float(phase['wall_seconds']):.6f} | {phase['failed_count']} |"
            )
        lines.extend(
            [
                "",
                "## Top phases by inclusive time",
                "",
                "| Phase | Count | Inclusive seconds | P50 | P90 | Maximum |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for phase in sorted(phases, key=lambda item: float(item["wall_seconds"]), reverse=True)[:20]:
            lines.append(
                f"| `{phase['path']}` | {phase['count']} | {float(phase['wall_seconds']):.6f} | "
                f"{float(phase['p50']):.6f} | {float(phase['p90']):.6f} | {float(phase['max']):.6f} |"
            )
        lines.append("")
        return "\n".join(lines)


@contextmanager
def profiled_run(
    config: ProfilingConfig,
    output: str | Path,
    events: EventSink | None,
    *,
    run_id: str,
) -> Iterator[PhaseRecorder]:
    """Build a configured recorder and write its per-process artifacts on exit."""
    override = os.environ.get("NANOQUANT_PROFILE")
    if override:
        try:
            config = replace(config, level=ProfilingLevel(override.lower()))
        except ValueError as exc:
            raise ValueError(f"invalid NANOQUANT_PROFILE level: {override!r}") from exc
    if config.level is ProfilingLevel.OFF:
        yield NULL_RECORDER
        return
    profiler = Profiler(config, run_id=run_id, events=events)
    try:
        yield profiler
    finally:
        ProfileWriter(output).write(profiler)
