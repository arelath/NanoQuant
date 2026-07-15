from __future__ import annotations

import gc
import os
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
    release_cached_host_memory,
    sample_device_memory,
)
from nanoquant.infrastructure.resource_usage import GpuProcessMemorySnapshot, ProcessMemorySnapshot
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


def test_memory_sample_includes_wddm_dedicated_and_shared_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        device_memory_module,
        "process_memory_snapshot",
        lambda: ProcessMemorySnapshot(10, 20, 30, 40),
    )
    monkeypatch.setattr(
        device_memory_module,
        "gpu_process_memory_snapshot",
        lambda: GpuProcessMemorySnapshot(50, 60, 70, 80),
    )
    monkeypatch.setattr(torch.cuda, "_initialized", True)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: {})
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (90, 100))

    sample = sample_device_memory()

    assert sample["wddm.dedicated_bytes"] == 50
    assert sample["wddm.shared_bytes"] == 60
    assert sample["wddm.peak_dedicated_bytes"] == 70
    assert sample["wddm.peak_shared_bytes"] == 80


def test_release_cached_host_memory_uses_accelerator_cache_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def release() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(torch._C, "_accelerator_emptyHostCache", release)

    assert release_cached_host_memory()
    assert calls == 1


@pytest.mark.skipif(os.name != "nt" or not torch.cuda.is_available(), reason="WDDM CUDA counters require Windows")
def test_wddm_shared_usage_exposes_and_releases_pinned_host_cache() -> None:
    torch.zeros(1, device="cuda")
    release_cached_host_memory()
    value = torch.empty(128 * 2**20, dtype=torch.uint8, pin_memory=True)
    value.fill_(1)
    live = sample_device_memory()
    del value
    gc.collect()
    cached = sample_device_memory()
    release_cached_host_memory()
    released = sample_device_memory()

    assert {"wddm.dedicated_bytes", "wddm.shared_bytes"} <= live.keys()
    assert cached["wddm.shared_bytes"] >= live["wddm.shared_bytes"] - 2**20
    assert released["wddm.shared_bytes"] <= cached["wddm.shared_bytes"] - 64 * 2**20


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
