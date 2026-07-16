"""Run-scoped identity, lease, event routing, and derived text rendering."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from nanoquant.config.codec import from_dict
from nanoquant.config.schema import ObservabilityConfig
from nanoquant.domain.runs import RunManifest
from nanoquant.infrastructure.device_memory import (
    CudaMemoryHistory,
    ResourceEventSink,
    ResourceSampler,
    SharedDeviceMemoryGuard,
    capture_oom_forensics,
    is_cuda_oom,
)
from nanoquant.infrastructure.events import (
    ConsoleEventDestination,
    EventDestination,
    EventRouter,
    JsonlEventDestination,
    prepare_event_stream,
    read_last_event,
    render_event_log,
)
from nanoquant.infrastructure.runs import RunDirectory
from nanoquant.ports.event_sink import EventSink, Severity


@dataclass(frozen=True, slots=True)
class RunSession:
    run_id: str
    output: Path
    events: EventSink
    manifest: RunManifest
    resumed: bool
    previous_run_id: str | None = None


def _levels(observability: ObservabilityConfig) -> tuple[Severity, Severity]:
    event_level = Severity.parse(observability.event_level)
    console_level = Severity.parse(observability.console_level)
    if event_level.rank > console_level.rank:
        raise ValueError("observability.event_level must be at least as verbose as console_level")
    if observability.record_admm_steps and event_level is not Severity.DEBUG:
        raise ValueError("record_admm_steps requires event_level=debug")
    if not math.isfinite(observability.record_resource_interval_seconds):
        raise ValueError("record_resource_interval_seconds must be finite")
    return event_level, console_level


@contextmanager
def open_run_session(
    output: str | Path,
    *,
    manifest: RunManifest,
    observability: ObservabilityConfig,
    registry_root: Path | None = None,
    console: bool = True,
    maximum_wddm_shared_bytes: int | None = None,
) -> Iterator[RunSession]:
    """Open the sole event writer for a run and render its disposable log at close."""

    root = Path(output)
    directory = RunDirectory(root.parent, root.name)
    existing_manifest = directory.manifest_path.exists()
    had_existing_state = existing_manifest or directory.events_path.exists() or (root / "state/journal.jsonl").exists()
    adopted = (
        from_dict(RunManifest, directory.read_manifest(), path="manifest")
        if existing_manifest
        else manifest
    )
    if not existing_manifest:
        directory.write_manifest(adopted)

    event_level, console_level = _levels(observability)
    lease = directory.lease()
    lease.acquire()
    router: EventRouter | None = None
    events: ResourceEventSink | None = None
    sampler: ResourceSampler | None = None
    history: CudaMemoryHistory | None = None
    shared_memory_guard = (
        None
        if maximum_wddm_shared_bytes is None
        else SharedDeviceMemoryGuard(maximum_wddm_shared_bytes)
    )
    try:
        initial_sequence, recovered_bytes = prepare_event_stream(directory.events_path)
        last_event = read_last_event(directory.events_path)
        destinations: list[EventDestination] = [JsonlEventDestination(directory.events_path, event_level)]
        if console:
            destinations.append(ConsoleEventDestination(console_level))
        router = EventRouter(
            adopted.run_id,
            event_level=event_level,
            destinations=tuple(destinations),
            initial_sequence=initial_sequence,
        )
        history = CudaMemoryHistory(
            directory.root / "state",
            observability.capture_cuda_trace,
        )
        if history.requested and not history.start():
            router.emit(
                "observability",
                "warning",
                "observability.vram_history_unavailable",
                error_type=history.unavailable_error_type,
            )
        events = ResourceEventSink(
            router,
            event_level,
            oom_callback=lambda error, stage, block, layer: capture_oom_forensics(
                directory.root,
                router,
                error,
                history,
                stage=stage,
                block=block,
                layer=layer,
            ),
            shared_memory_guard=shared_memory_guard,
        )
        sampler = ResourceSampler(
            router,
            observability.record_resource_interval_seconds,
            observer=None if shared_memory_guard is None else shared_memory_guard.observe,
        )
        sampler.start()
        previous_run_id = None if last_event is None or last_event.run_id == adopted.run_id else last_event.run_id
        if recovered_bytes:
            router.emit(
                "observability",
                "warning",
                "observability.tail_recovered",
                quarantined_bytes=recovered_bytes,
            )
        if lease.taken_over_owner is not None:
            router.emit(
                "run",
                "warning",
                "run.lease_taken_over",
                previous_owner=lease.taken_over_owner,
            )
        if registry_root is not None:
            try:
                from nanoquant.infrastructure.run_registry import register_external_run

                registry_owner = register_external_run(registry_root, directory.root, adopted)
                if registry_owner is not None:
                    router.emit(
                        "run",
                        "warning",
                        "run.registry_lock_taken_over",
                        previous_owner=registry_owner,
                    )
            except Exception as exc:
                router.emit(
                    "run",
                    "warning",
                    "run.registry_registration_failed",
                    error_type=type(exc).__name__,
                )
        try:
            yield RunSession(adopted.run_id, directory.root, events, adopted, had_existing_state, previous_run_id)
            if shared_memory_guard is not None:
                shared_memory_guard.raise_if_violated(require_available=True)
        except BaseException as exc:
            if is_cuda_oom(exc) and events is not None:
                events.capture_oom(exc)
            raise
    finally:
        try:
            if router is not None:
                if sampler is not None:
                    sampler.stop()
                if history is not None and history.active:
                    snapshot = history.dump("terminal")
                    if snapshot is not None:
                        router.emit(
                            "observability",
                            "info",
                            "observability.vram_history_dumped",
                            path=str(snapshot.relative_to(directory.root)),
                        )
                    history.stop()
                router.flush()
                try:
                    render_event_log(directory.events_path, directory.root / "run.log")
                except Exception as exc:
                    router.emit(
                        "observability",
                        "warning",
                        "observability.render_failed",
                        error_type=type(exc).__name__,
                    )
                    router.flush()
                router.close()
        finally:
            lease.release()
