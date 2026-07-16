"""Shared non-synchronizing device-memory meters and diagnostic instruments."""

from __future__ import annotations

import os
import re
import threading
import weakref
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import torch

from nanoquant.infrastructure.resource_usage import gpu_process_memory_snapshot, process_memory_snapshot
from nanoquant.ports.event_sink import Event, EventSink, Severity

MemorySampler = Callable[[], Mapping[str, int]]


class SharedDeviceMemoryLimitExceeded(RuntimeError):
    """Raised when a WDDM process crosses its configured shared-memory ceiling."""

    code = "VRAM001"


class SharedDeviceMemoryGuard:
    """Thread-safe fail-closed monitor for process-scoped WDDM shared memory."""

    def __init__(self, maximum_bytes: int) -> None:
        if maximum_bytes < 0:
            raise ValueError("maximum WDDM shared memory must be non-negative")
        self.maximum_bytes = maximum_bytes
        self._peak_bytes = 0
        self._violation_bytes: int | None = None
        self._metric_seen = False
        self._lock = threading.Lock()

    def observe(self, sample: Mapping[str, int]) -> None:
        value = sample.get("wddm.shared_bytes")
        if value is None:
            return
        measured = int(value)
        with self._lock:
            self._metric_seen = True
            self._peak_bytes = max(self._peak_bytes, measured)
            if measured > self.maximum_bytes:
                self._violation_bytes = max(self._violation_bytes or 0, measured)

    @property
    def peak_bytes(self) -> int:
        with self._lock:
            return self._peak_bytes

    @property
    def metric_seen(self) -> bool:
        with self._lock:
            return self._metric_seen

    def raise_if_violated(self, *, require_available: bool = False) -> None:
        with self._lock:
            measured = self._violation_bytes
            metric_seen = self._metric_seen
        if require_available and bool(getattr(torch.cuda, "_initialized", False)) and not metric_seen:
            raise RuntimeError("VRAM002 WDDM shared-memory meter is unavailable after CUDA initialization")
        if measured is not None:
            raise SharedDeviceMemoryLimitExceeded(
                "VRAM001 WDDM shared GPU memory limit exceeded: "
                f"observed={measured} bytes, maximum={self.maximum_bytes} bytes"
            )

    def sample_and_check(self, sampler: MemorySampler | None = None) -> Mapping[str, int]:
        resolved_sampler = sample_device_memory if sampler is None else sampler
        sample = resolved_sampler()
        self.observe(sample)
        self.raise_if_violated(require_available=True)
        return sample


class SharedDeviceMemoryMonitor(AbstractContextManager["SharedDeviceMemoryMonitor"]):
    """Poll a shared-memory guard and expose explicit safe-point checks."""

    def __init__(
        self,
        maximum_bytes: int,
        *,
        interval_seconds: float = 0.25,
        sampler: MemorySampler | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("shared-memory monitoring interval must be positive")
        self.guard = SharedDeviceMemoryGuard(maximum_bytes)
        self._interval_seconds = interval_seconds
        self._sampler = sample_device_memory if sampler is None else sampler
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sampling_error: BaseException | None = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self.guard.observe(self._sampler())
            except BaseException as exc:
                self._sampling_error = exc
                return

    def start(self) -> SharedDeviceMemoryMonitor:
        self.guard.observe(self._sampler())
        self._thread = threading.Thread(
            target=self._run,
            name="nanoquant-shared-memory-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def check(self) -> None:
        if self._sampling_error is not None:
            raise RuntimeError("VRAM002 WDDM shared-memory monitoring failed") from self._sampling_error
        self.guard.sample_and_check(self._sampler)

    def __enter__(self) -> SharedDeviceMemoryMonitor:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 2))
        if exc_value is None:
            self.check()
        return None

_BOUNDARY_EVENTS = frozenset(
    {
        "run.started",
        "run.resumed",
        "run.completed",
        "run.failed",
        "run.interrupted",
        "block.started",
        "block.completed",
        "layer.committed",
        "layer.reused",
        "tuning.epoch_completed",
        "factorized_tuning.epoch_checkpoint_committed",
        "probe.completed",
        "host_pinned_cache.released",
    }
)
_REQUESTED_BYTES = re.compile(
    r"(?:tried to allocate|allocating)\s+([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b)",
    re.IGNORECASE,
)
_UNIT_BYTES = {
    "b": 1,
    "kb": 1000,
    "kib": 2**10,
    "mb": 1000**2,
    "mib": 2**20,
    "gb": 1000**3,
    "gib": 2**30,
    "tb": 1000**4,
    "tib": 2**40,
}


def sample_device_memory() -> dict[str, int]:
    """Read host and initialized-CUDA meters without synchronizing or initializing CUDA."""

    process = process_memory_snapshot()
    sample = {
        "host.working_set_bytes": process.working_set_bytes,
        "host.peak_working_set_bytes": process.peak_working_set_bytes,
        "host.private_bytes": process.private_bytes,
        "host.peak_private_bytes": process.peak_private_bytes,
    }
    if not bool(getattr(torch.cuda, "_initialized", False)):
        return sample
    stats = torch.cuda.memory_stats()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    sample.update(
        {
            "cuda.allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
            "cuda.reserved_bytes": int(stats.get("reserved_bytes.all.current", 0)),
            "cuda.peak_allocated_bytes": int(stats.get("allocated_bytes.all.peak", 0)),
            "cuda.peak_reserved_bytes": int(stats.get("reserved_bytes.all.peak", 0)),
            "cuda.device_free_bytes": int(free_bytes),
            "cuda.device_used_bytes": int(total_bytes - free_bytes),
            "cuda.device_total_bytes": int(total_bytes),
            "cuda.allocation_count": int(stats.get("allocation.all.allocated", 0)),
            "cuda.free_count": int(stats.get("allocation.all.freed", 0)),
        }
    )
    gpu_process = gpu_process_memory_snapshot()
    if gpu_process is not None:
        sample.update(
            {
                "wddm.dedicated_bytes": gpu_process.dedicated_bytes,
                "wddm.shared_bytes": gpu_process.shared_bytes,
                "wddm.peak_dedicated_bytes": gpu_process.peak_dedicated_bytes,
                "wddm.peak_shared_bytes": gpu_process.peak_shared_bytes,
            }
        )
    return sample


def release_cached_host_memory() -> bool:
    """Release unoccupied accelerator-pinned host allocations when supported.

    The pinned-host allocator is independent of the CUDA device allocator. On
    WDDM, its cached blocks remain GPU-addressable and appear as shared GPU
    memory, so ``torch.cuda.empty_cache()`` does not release them.
    """

    release = getattr(torch._C, "_accelerator_emptyHostCache", None)
    if not callable(release):
        release = getattr(torch._C, "_host_emptyCache", None)
    if not callable(release):
        return False
    release()
    return True


@dataclass(frozen=True, slots=True)
class PeakReading:
    allocated_bytes: int
    reserved_bytes: int


def _default_peak_reader(device: str) -> PeakReading:
    return PeakReading(
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


class PeakWindow(AbstractContextManager["PeakWindow"]):
    """Own a resettable CUDA peak window and fold nested resets into its parent."""

    _local = threading.local()

    def __init__(
        self,
        device: str,
        *,
        resetter: Callable[[], None] | None = None,
        reader: Callable[[], PeakReading] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.device = device
        self._enabled = (
            device.startswith("cuda") and torch.cuda.is_available()
            if enabled is None
            else enabled
        )
        self._resetter = resetter or (lambda: torch.cuda.reset_peak_memory_stats(device))
        self._reader = reader or (lambda: _default_peak_reader(device))
        self._allocated_bytes = 0
        self._reserved_bytes = 0
        self._active = False

    @classmethod
    def _stack(cls) -> list[weakref.ReferenceType[PeakWindow]]:
        stack = cast(list[weakref.ReferenceType[PeakWindow]], getattr(cls._local, "stack", []))
        stack[:] = [reference for reference in stack if reference() is not None]
        cls._local.stack = stack
        return stack

    @classmethod
    def _active_parent(cls) -> PeakWindow | None:
        for reference in reversed(cls._stack()):
            window = reference()
            if window is not None and window._active:
                return window
        return None

    def _fold(self, reading: PeakReading) -> None:
        self._allocated_bytes = max(self._allocated_bytes, reading.allocated_bytes)
        self._reserved_bytes = max(self._reserved_bytes, reading.reserved_bytes)

    def _fold_current(self) -> None:
        if self._enabled and self._active:
            self._fold(self._reader())

    def start(self) -> PeakWindow:
        if self._active:
            raise RuntimeError("peak window is already active")
        if not self._enabled:
            self._active = True
            return self
        parent = self._active_parent()
        if parent is not None:
            parent._fold_current()
        self._resetter()
        self._active = True
        self._stack().append(weakref.ref(self))
        return self

    def finish(self) -> PeakReading:
        if not self._active:
            return self.reading
        if self._enabled:
            self._fold_current()
            stack = self._stack()
            stack[:] = [reference for reference in stack if reference() is not self]
            self._active = False
            parent = self._active_parent()
            if parent is not None:
                parent._fold(self.reading)
        else:
            self._active = False
        return self.reading

    @property
    def reading(self) -> PeakReading:
        return PeakReading(self._allocated_bytes, self._reserved_bytes)

    @property
    def peak_allocated_bytes(self) -> int:
        return self._allocated_bytes

    @property
    def peak_reserved_bytes(self) -> int:
        return self._reserved_bytes

    def __enter__(self) -> PeakWindow:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish()
        return None


class ResourceEventSink:
    """Enrich memory-relevant lifecycle events while preserving event filtering."""

    def __init__(
        self,
        sink: EventSink,
        threshold: Severity,
        sampler: MemorySampler = sample_device_memory,
        oom_callback: Callable[[BaseException, str | None, int | None, str | None], None] | None = None,
        shared_memory_guard: SharedDeviceMemoryGuard | None = None,
    ) -> None:
        self._sink = sink
        self._threshold = threshold
        self._sampler = sampler
        self._sampling_failed = False
        self._oom_callback = oom_callback
        self._captured_ooms: set[int] = set()
        self._shared_memory_guard = shared_memory_guard

    def capture_oom(
        self,
        error: BaseException,
        *,
        stage: str | None = None,
        block: int | None = None,
        layer: str | None = None,
    ) -> None:
        identity = id(error)
        if self._oom_callback is None or identity in self._captured_ooms:
            return
        self._captured_ooms.add(identity)
        self._oom_callback(error, stage, block, layer)

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
        terminal = name in {"run.failed", "run.interrupted"}
        if self._shared_memory_guard is not None and not terminal:
            self._shared_memory_guard.raise_if_violated()
            if bool(getattr(torch.cuda, "_initialized", False)) and not self._shared_memory_guard.metric_seen:
                self._shared_memory_guard.sample_and_check(self._sampler)
        parsed = Severity.parse(severity)
        enriched = fields
        if (
            name in _BOUNDARY_EVENTS
            and parsed.rank >= self._threshold.rank
            and not self._sampling_failed
        ):
            try:
                sample = self._sampler()
                if self._shared_memory_guard is not None:
                    self._shared_memory_guard.observe(sample)
                enriched = {**sample, **fields}
            except Exception as exc:
                self._sampling_failed = True
                self._sink.emit(
                    "observability",
                    "warning",
                    "observability.boundary_memory_disabled",
                    error_type=type(exc).__name__,
                )
        if self._shared_memory_guard is not None and not terminal:
            self._shared_memory_guard.raise_if_violated(require_available=True)
        return self._sink.emit(
            stage,
            severity,
            name,
            span_id=span_id,
            parent_span_id=parent_span_id,
            **enriched,
        )


class ResourceSampler:
    """A stoppable daemon that emits periodic resource samples."""

    def __init__(
        self,
        events: EventSink,
        interval_seconds: float,
        sampler: MemorySampler = sample_device_memory,
        observer: Callable[[Mapping[str, int]], None] | None = None,
    ) -> None:
        self._events = events
        self._interval_seconds = interval_seconds
        self._sampler = sampler
        self._observer = observer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._interval_seconds <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="nanoquant-resource-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                fields = self._sampler()
                if self._observer is not None:
                    self._observer(fields)
            except Exception as exc:
                try:
                    self._events.emit(
                        "observability",
                        "warning",
                        "observability.sampler_disabled",
                        error_type=type(exc).__name__,
                    )
                except Exception:
                    pass
                return
            cast(Any, self._events).emit("resource", "info", "resource.sample", **fields)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)


def _environment_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class CudaMemoryHistory:
    """Contain PyTorch allocator-history private APIs behind a best-effort adapter."""

    def __init__(self, state_directory: Path, requested: bool, *, max_entries: int = 100_000) -> None:
        self.state_directory = state_directory
        self.requested = requested or _environment_enabled("NANOQUANT_VRAM_HISTORY")
        self.max_entries = max_entries
        self.active = False
        self.unavailable_error_type: str | None = None
        self._sequence = 0

    def start(self) -> bool:
        if not self.requested:
            return False
        try:
            memory = cast(Any, torch.cuda.memory)
            record = memory._record_memory_history
            dump_snapshot = memory._dump_snapshot
            if not callable(record) or not callable(dump_snapshot):
                raise TypeError("CUDA memory history APIs are not callable")
            record(max_entries=self.max_entries)
            self.active = True
        except Exception as exc:
            self.unavailable_error_type = type(exc).__name__
            self.active = False
        return self.active

    def dump(self, reason: str) -> Path | None:
        if not self.active:
            return None
        try:
            self.state_directory.mkdir(parents=True, exist_ok=True)
            self._sequence += 1
            target = self.state_directory / f"vram-history-{self._sequence}-{reason}.pickle"
            temporary = target.with_suffix(target.suffix + ".tmp")
            memory = cast(Any, torch.cuda.memory)
            memory._dump_snapshot(str(temporary))
            temporary.replace(target)
            return target
        except Exception:
            return None

    def stop(self) -> None:
        if not self.active:
            return
        try:
            memory = cast(Any, torch.cuda.memory)
            memory._record_memory_history(enabled=None)
        except Exception:
            pass
        self.active = False


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or "cuda out of memory" in message


def requested_oom_bytes(error: BaseException) -> int | None:
    match = _REQUESTED_BYTES.search(str(error))
    if match is None:
        return None
    unit = match.group(2).lower()
    return int(float(match.group(1)) * _UNIT_BYTES[unit])


def capture_oom_forensics(
    output: Path,
    events: EventSink,
    error: BaseException,
    history: CudaMemoryHistory | None = None,
    *,
    stage: str | None = None,
    block: int | None = None,
    layer: str | None = None,
    sampler: MemorySampler = sample_device_memory,
) -> None:
    """Record diagnostics without ever replacing or mutating the original OOM."""

    fields: dict[str, object] = {
        "error_type": type(error).__name__,
        "stage_context": stage,
        "block": block,
        "layer": layer,
    }
    try:
        fields.update(sampler())
    except Exception:
        pass
    requested = requested_oom_bytes(error)
    if requested is not None:
        fields["requested_bytes"] = requested
    try:
        state = output / "state"
        state.mkdir(parents=True, exist_ok=True)
        reports = tuple(state.glob("oom-report-*.txt"))
        report = state / f"oom-report-{len(reports) + 1}.txt"
        temporary = report.with_suffix(".txt.tmp")
        temporary.write_text(torch.cuda.memory_summary(), encoding="utf-8")
        temporary.replace(report)
        fields["oom_report_path"] = str(report.relative_to(output))
    except Exception:
        pass
    if history is not None:
        snapshot = history.dump("oom")
        if snapshot is not None:
            fields["vram_history_path"] = str(snapshot.relative_to(output))
    try:
        cast(Any, events).emit("resource", "error", "resource.oom_snapshot", **fields)
    except Exception:
        pass
