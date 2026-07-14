from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

import nanoquant.infrastructure.device_memory as device_memory_module
from nanoquant.infrastructure.device_memory import (
    CudaMemoryHistory,
    PeakReading,
    PeakWindow,
    ResourceEventSink,
    ResourceSampler,
    capture_oom_forensics,
    sample_device_memory,
)
from nanoquant.infrastructure.resource_usage import ProcessMemorySnapshot
from nanoquant.ports.event_sink import Event, Severity


@dataclass
class RecordingSink:
    events: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)

    def emit(
        self,
        stage: str,
        severity: str,
        name: str,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        **fields: object,
    ) -> Event | None:
        del stage, span_id, parent_span_id
        self.events.append((severity, name, fields))
        return None


def test_memory_sample_does_not_initialize_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        device_memory_module,
        "process_memory_snapshot",
        lambda: ProcessMemorySnapshot(10, 20, 30, 40),
    )
    monkeypatch.setattr(torch.cuda, "_initialized", False)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: pytest.fail("CUDA must not be sampled"))

    assert sample_device_memory() == {
        "host.working_set_bytes": 10,
        "host.peak_working_set_bytes": 20,
        "host.private_bytes": 30,
        "host.peak_private_bytes": 40,
    }


def test_nested_peak_windows_fold_inner_resets_into_parent() -> None:
    peak = PeakReading(0, 0)
    reset_count = 0

    def reset() -> None:
        nonlocal peak, reset_count
        reset_count += 1
        peak = PeakReading(0, 0)

    def read() -> PeakReading:
        return peak

    with PeakWindow("cuda:0", resetter=reset, reader=read, enabled=True) as outer:
        peak = PeakReading(100, 150)
        with PeakWindow("cuda:0", resetter=reset, reader=read, enabled=True) as inner:
            peak = PeakReading(200, 250)
        assert inner.reading == PeakReading(200, 250)
        peak = PeakReading(120, 180)

    assert reset_count == 2
    assert outer.reading == PeakReading(200, 250)


def test_boundary_enrichment_respects_filter_and_quarantines_sampler_failure() -> None:
    sink = RecordingSink()
    calls = 0

    def fail() -> dict[str, int]:
        nonlocal calls
        calls += 1
        raise RuntimeError("injected")

    events = ResourceEventSink(sink, Severity.INFO, fail)
    events.emit("resident", "debug", "probe.completed")
    assert calls == 0

    events.emit("run", "info", "run.started", config_hash="hash")
    events.emit("run", "info", "run.completed")

    assert calls == 1
    assert [name for _severity, name, _fields in sink.events] == [
        "probe.completed",
        "observability.boundary_memory_disabled",
        "run.started",
        "run.completed",
    ]


def test_periodic_sampler_disables_itself_after_first_failure() -> None:
    sink = RecordingSink()
    sampler = ResourceSampler(sink, 0.001, lambda: (_ for _ in ()).throw(RuntimeError("injected")))
    sampler.start()
    deadline = time.monotonic() + 1
    while not sink.events and time.monotonic() < deadline:
        time.sleep(0.005)
    sampler.stop()

    assert [name for _severity, name, _fields in sink.events] == ["observability.sampler_disabled"]


def test_oom_forensics_records_standard_meters_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = RecordingSink()
    monkeypatch.setattr(torch.cuda, "memory_summary", lambda: "allocator summary")
    error = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.50 MiB")

    capture_oom_forensics(
        tmp_path,
        sink,
        error,
        sampler=lambda: {"cuda.allocated_bytes": 123},
        stage="factorize",
        block=2,
        layer="mlp.down_proj",
    )

    assert len(sink.events) == 1
    severity, name, fields = sink.events[0]
    assert (severity, name) == ("error", "resource.oom_snapshot")
    assert fields["requested_bytes"] == int(2.5 * 2**20)
    assert fields["cuda.allocated_bytes"] == 123
    report = tmp_path / str(fields["oom_report_path"])
    assert report.read_text(encoding="utf-8") == "allocator summary"


def test_allocator_history_adapter_is_bounded_and_writes_atomic_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def record(**kwargs: object) -> None:
        calls.append(kwargs)

    def dump(path: str) -> None:
        Path(path).write_bytes(b"snapshot")

    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", record)
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", dump)
    history = CudaMemoryHistory(tmp_path, True, max_entries=123)

    assert history.start()
    snapshot = history.dump("terminal")
    history.stop()

    assert calls == [{"max_entries": 123}, {"enabled": None}]
    assert snapshot is not None and snapshot.read_bytes() == b"snapshot"
