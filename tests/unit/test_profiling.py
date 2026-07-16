from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import nanoquant.infrastructure.device_memory as device_memory_module
import nanoquant.infrastructure.profiling as profiling_module
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.profiling import NULL_RECORDER
from nanoquant.infrastructure.profiling import Profiler, ProfileWriter, profiled_run
from nanoquant.infrastructure.resource_usage import ProcessMemorySnapshot
from nanoquant.ports.event_sink import Event


@dataclass
class Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass
class FakeCudaClock:
    value_ms: float = 0.0

    def event(self) -> FakeCudaEvent:
        return FakeCudaEvent(self)

    def advance(self, milliseconds: float) -> None:
        self.value_ms += milliseconds


class FakeCudaEvent:
    def __init__(self, clock: FakeCudaClock) -> None:
        self.clock = clock
        self.recorded_ms: float | None = None

    def record(self) -> None:
        self.recorded_ms = self.clock.value_ms

    def elapsed_time(self, end_event: FakeCudaEvent) -> float:
        assert self.recorded_ms is not None
        assert end_event.recorded_ms is not None
        return end_event.recorded_ms - self.recorded_ms


class MemoryEvents:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(
        self,
        stage: str,
        severity: str,
        name: str,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        **fields: object,
    ) -> Event:
        event = Event(1, "now", "test", len(self.events) + 1, stage, severity, name, fields, span_id, parent_span_id)
        self.events.append(event)
        return event


def _phases(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    values = payload["phases"]
    assert isinstance(values, list)
    return {str(item["path"]): item for item in values if isinstance(item, dict)}


def test_profiler_aggregates_nested_self_time_counters_groups_and_coverage() -> None:
    clock = Clock()
    profiler = Profiler(
        ProfilingConfig(raw_samples_per_phase=2, emit_span_events=False),
        run_id="test",
        clock=clock,
    )
    with profiler.phase("run"):
        clock.advance(1.0)
        with profiler.phase("block", block=3):
            clock.advance(2.0)
            profiler.add("transfer.h2d_bytes", 12, direction="h2d")
        clock.advance(3.0)
    payload = profiler.snapshot()
    phases = _phases(payload)

    assert phases["run"]["wall_seconds"] == 6.0
    assert phases["run"]["self_seconds"] == 4.0
    assert phases["run/block"]["wall_seconds"] == 2.0
    assert phases["run/block"]["groups"] == {"block=3": {"count": 1, "self_seconds": 2.0, "wall_seconds": 2.0}}
    assert payload["coverage"] == {
        "wall_total_seconds": 6.0,
        "attributed_seconds": 2.0,
        "fraction": 1.0 / 3.0,
    }
    assert payload["warnings"] == [
        {"code": "PERF001", "message": "profile coverage is below 90%", "fraction": 1.0 / 3.0}
    ]
    counters = payload["counters"]
    assert isinstance(counters, list)
    assert counters == [
        {
            "name": "transfer.h2d_bytes",
            "total": 12.0,
            "by_phase": {"run/block": 12.0},
            "groups": {"direction=h2d": 12.0},
        }
    ]


def test_profiler_records_repeated_samples_and_failure_without_swallowing_exception() -> None:
    clock = Clock()
    profiler = Profiler(
        ProfilingConfig(raw_samples_per_phase=3, emit_span_events=False),
        run_id="test",
        clock=clock,
    )
    for duration in (1.0, 3.0, 2.0):
        with profiler.phase("work"):
            clock.advance(duration)
    with pytest.raises(RuntimeError, match="boom"):
        with profiler.phase("failed"):
            clock.advance(0.5)
            raise RuntimeError("boom")
    phases = _phases(profiler.snapshot())

    assert phases["work"]["count"] == 3
    assert phases["work"]["min"] == 1.0
    assert phases["work"]["p50"] == 2.0
    assert phases["work"]["p90"] == 3.0
    assert phases["work"]["self_p50"] == 2.0
    assert phases["work"]["self_p90"] == 3.0
    assert phases["work"]["max"] == 3.0
    assert phases["failed"]["failed_count"] == 1


def test_profiler_mirrors_parented_span_events() -> None:
    clock = Clock()
    events = MemoryEvents()
    profiler = Profiler(ProfilingConfig(emit_span_events=True), run_id="test", events=events, clock=clock)
    with profiler.phase("run"):
        with profiler.phase("child", block=1):
            clock.advance(1.0)
    profiler.finish()

    assert [event.name for event in events.events] == [
        "phase.started",
        "phase.started",
        "phase.completed",
        "phase.completed",
    ]
    assert events.events[1].parent_span_id == events.events[0].span_id
    assert events.events[2].fields["path"] == "run/child"


def test_profiler_span_attributes_do_not_collide_with_event_arguments() -> None:
    events = MemoryEvents()
    profiler = Profiler(ProfilingConfig(emit_span_events=True), run_id="test", events=events)

    with profiler.phase("stage", stage="outlier-selection", name="fixture"):
        pass

    assert events.events[0].fields["profile_stage"] == "outlier-selection"
    assert events.events[0].fields["profile_name"] == "fixture"
    assert events.events[1].fields["profile_stage"] == "outlier-selection"
    phase = _phases(profiler.snapshot())["stage"]
    assert "name=fixture|stage=outlier-selection" in phase["groups"]


def test_null_recorder_reuses_one_context_and_accepts_namespaced_counters() -> None:
    first = NULL_RECORDER.phase("anything")
    second = NULL_RECORDER.phase("anything_else", block=1)
    assert first is second
    with first:
        NULL_RECORDER.add("transfer.h2d_bytes", 1.0)
        NULL_RECORDER.mark("checkpoint")


def test_writer_versions_per_process_artifacts_and_renders_summary(tmp_path: Path) -> None:
    clock = Clock()
    profiler = Profiler(ProfilingConfig(emit_span_events=False), run_id="write-test", clock=clock)
    with profiler.phase("run"):
        with profiler.phase("work"):
            clock.advance(2.0)
    first_json, first_markdown = ProfileWriter(tmp_path).write(profiler)
    second_json, second_markdown = ProfileWriter(tmp_path).write(profiler)

    assert first_json.name == "profile.json"
    assert first_markdown.name == "profile.md"
    assert second_json.name == "profile.2.json"
    assert second_markdown.name == "profile.2.md"
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["run_id"] == "write-test"
    assert payload["environment"]["runtime_fingerprint"].startswith("sha256:")
    summary = first_markdown.read_text(encoding="utf-8")
    assert "Coverage: 100.00%" in summary
    assert "`run`" in summary


def test_profiled_run_writes_on_exception_and_off_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stop"):
        with profiled_run(ProfilingConfig(emit_span_events=False), tmp_path, None, run_id="failed") as recorder:
            with recorder.phase("run"):
                raise ValueError("stop")
    payload = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert _phases(payload)["run"]["failed_count"] == 1

    disabled = tmp_path / "disabled"
    with profiled_run(ProfilingConfig(level=ProfilingLevel.OFF), disabled, None, run_id="off") as recorder:
        assert recorder is NULL_RECORDER
    assert not disabled.exists()


def test_profiler_rejects_invalid_names_attributes_and_open_snapshot() -> None:
    profiler = Profiler(ProfilingConfig(emit_span_events=False), run_id="invalid")
    with pytest.raises(ValueError, match="phase"):
        profiler.phase("bad.name")
    with pytest.raises(TypeError, match="scalar"):
        profiler.phase("good", tensor=object())
    with pytest.raises(ValueError, match="counter"):
        profiler.add("bad..counter", 1.0)
    phase = profiler.phase("open")
    phase.__enter__()
    with pytest.raises(RuntimeError, match="open phases"):
        profiler.snapshot()
    phase.__exit__(None, None, None)


def test_micro_level_is_supported_and_trace_fails_explicitly() -> None:
    events = MemoryEvents()
    profiler = Profiler(ProfilingConfig(level=ProfilingLevel.MICRO), run_id="micro", events=events)
    with profiler.phase("iteration"):
        pass
    assert profiler.config.level is ProfilingLevel.MICRO
    assert not events.events
    with pytest.raises(NotImplementedError, match="trace"):
        Profiler(ProfilingConfig(level=ProfilingLevel.TRACE), run_id="trace")


def test_memory_counters_record_phase_boundaries_without_initializing_cuda() -> None:
    samples = iter(
        (
            {"host.private_bytes": 100, "host.peak_private_bytes": 100},
            {"host.private_bytes": 120, "host.peak_private_bytes": 120},
            {"host.private_bytes": 110, "host.peak_private_bytes": 120},
            {"host.private_bytes": 160, "host.peak_private_bytes": 160},
        )
    )
    profiler = Profiler(
        ProfilingConfig(memory_counters=True),
        run_id="memory",
        memory_sampler=lambda: next(samples),
    )
    with profiler.phase("run"):
        with profiler.phase("work"):
            pass

    payload = profiler.snapshot()
    phases = _phases(payload)
    assert payload["memory_counters"] == {"enabled": True, "available": True}
    assert phases["run"]["memory"]["host.private_bytes"] == {
        "count": 1,
        "first": 100,
        "last": 160,
        "minimum": 100,
        "maximum": 160,
        "net_change": 60,
        "positive_delta": 60,
    }
    assert phases["run/work"]["memory"]["host.private_bytes"] == {
        "count": 1,
        "first": 120,
        "last": 110,
        "minimum": 110,
        "maximum": 120,
        "net_change": -10,
        "positive_delta": 0,
    }
    summary = ProfileWriter.render_markdown(payload)
    assert "## Run memory counters" in summary
    assert "`host.private_bytes` | 100 | 160" in summary


def test_runtime_memory_sample_separates_live_reserved_and_device_visible_cuda_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device_memory_module, "process_memory_snapshot", lambda: ProcessMemorySnapshot(10, 20, 30, 40))
    monkeypatch.setattr(device_memory_module.torch.cuda, "_initialized", True)
    monkeypatch.setattr(
        device_memory_module.torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.current": 100,
            "reserved_bytes.all.current": 180,
            "allocated_bytes.all.peak": 200,
            "reserved_bytes.all.peak": 240,
            "allocation.all.allocated": 7,
            "allocation.all.freed": 3,
        },
    )
    monkeypatch.setattr(device_memory_module.torch.cuda, "mem_get_info", lambda: (250, 1_000))
    monkeypatch.setattr(device_memory_module, "gpu_process_memory_snapshot", lambda: None)

    assert profiling_module._runtime_memory_sample() == {
        "host.working_set_bytes": 10,
        "host.peak_working_set_bytes": 20,
        "host.private_bytes": 30,
        "host.peak_private_bytes": 40,
        "cuda.allocated_bytes": 100,
        "cuda.reserved_bytes": 180,
        "cuda.peak_allocated_bytes": 200,
        "cuda.peak_reserved_bytes": 240,
        "cuda.device_free_bytes": 250,
        "cuda.device_used_bytes": 750,
        "cuda.device_total_bytes": 1_000,
        "cuda.allocation_count": 7,
        "cuda.free_count": 3,
    }


def test_memory_counter_failure_is_diagnostic_only() -> None:
    def fail() -> dict[str, int]:
        raise OSError("unavailable")

    profiler = Profiler(ProfilingConfig(memory_counters=True), run_id="memory-failure", memory_sampler=fail)
    with profiler.phase("run"):
        pass
    payload = profiler.snapshot()

    assert payload["memory_counters"] == {"enabled": True, "available": False}
    assert any(warning["code"] == "PERF004" for warning in payload["warnings"])
    assert _phases(payload)["run"]["memory"] == {}


def test_cuda_timing_is_sampled_deferred_resolved_once_and_estimated() -> None:
    wall_clock = Clock()
    cuda_clock = FakeCudaClock()
    synchronizations: list[None] = []
    profiler = Profiler(
        ProfilingConfig(
            level=ProfilingLevel.MICRO,
            cuda_timing=True,
            cuda_sample_every=2,
            raw_samples_per_phase=2,
            emit_span_events=False,
        ),
        run_id="cuda",
        clock=wall_clock,
        cuda_event_factory=cuda_clock.event,
        cuda_synchronize=lambda: synchronizations.append(None),
    )
    for duration_ms in (10.0, 20.0, 30.0, 40.0, 50.0):
        with profiler.phase("work"):
            wall_clock.advance(1.0)
            cuda_clock.advance(duration_ms)

    payload = profiler.snapshot()
    phase = _phases(payload)["work"]
    assert synchronizations == [None]
    assert phase["cuda_sample_count"] == 2
    assert phase["cuda_seconds"] == pytest.approx(0.1)
    assert phase["cuda_p50"] == pytest.approx(0.03)
    assert phase["cuda_p90"] == pytest.approx(0.03)
    assert payload["cuda_timing"] == {"enabled": True, "sample_every": 2, "resolved": True}
    profiler.snapshot()
    assert synchronizations == [None]


def test_cuda_resolution_failure_warns_without_masking_profile() -> None:
    cuda_clock = FakeCudaClock()

    def fail_synchronize() -> None:
        raise RuntimeError("device lost")

    profiler = Profiler(
        ProfilingConfig(cuda_timing=True),
        run_id="cuda-failure",
        cuda_event_factory=cuda_clock.event,
        cuda_synchronize=fail_synchronize,
    )
    with profiler.phase("work"):
        cuda_clock.advance(1.0)
    payload = profiler.snapshot()

    warnings = payload["warnings"]
    assert isinstance(warnings, list)
    assert any(item["code"] == "PERF003" for item in warnings)
    assert payload["cuda_timing"] == {"enabled": True, "sample_every": 16, "resolved": False}
