import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pytest

from nanoquant.infrastructure.events import (
    ConsoleEventDestination,
    EventRouter,
    EventWriteError,
    JsonlEventSink,
    prepare_event_stream,
    project_event,
    render_event_line,
    sanitize_fields,
)
from nanoquant.ports.event_sink import Event, Severity


class MemoryDestination:
    role: Literal["required", "optional"]

    def __init__(
        self,
        role: Literal["required", "optional"],
        threshold: Severity,
        *,
        fail: bool = False,
    ) -> None:
        self.role = role
        self.threshold = threshold
        self.fail = fail
        self.events: list[Event] = []
        self.closed = False

    def accept(self, event: Event) -> None:
        if self.fail:
            raise OSError("injected destination failure")
        self.events.append(event)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_router_filters_before_sequence_and_shares_one_envelope() -> None:
    required = MemoryDestination("required", Severity.DEBUG)
    console = MemoryDestination("optional", Severity.WARNING)
    router = EventRouter(
        "run",
        event_level=Severity.INFO,
        destinations=(required, console),
    )

    assert router.emit("stage", "debug", "detail.hidden") is None
    first = router.emit("stage", "info", "work.started")
    second = router.emit("stage", "warning", "work.warning", value=2)

    assert first is required.events[0]
    assert second is required.events[1] is console.events[0]
    assert [event.sequence for event in required.events] == [1, 2]


def test_optional_failure_is_quarantined_and_reported_once() -> None:
    required = MemoryDestination("required", Severity.DEBUG)
    broken = MemoryDestination("optional", Severity.DEBUG, fail=True)
    router = EventRouter(
        "run",
        event_level=Severity.DEBUG,
        destinations=(broken, required),
    )

    router.emit("stage", "info", "work.started")
    router.emit("stage", "info", "work.completed")

    assert [event.name for event in required.events] == [
        "work.started",
        "observability.destination_disabled",
        "work.completed",
    ]
    assert required.events[1].fields == {
        "destination": "MemoryDestination",
        "error_type": "OSError",
    }


def test_required_destination_failure_is_fatal() -> None:
    broken = MemoryDestination("required", Severity.DEBUG, fail=True)
    router = EventRouter("run", event_level=Severity.DEBUG, destinations=(broken,))
    with pytest.raises(EventWriteError, match="required event destination failed"):
        router.emit("stage", "error", "work.failed")


def test_sanitizer_is_bounded_redacted_and_never_uses_object_repr() -> None:
    class Unsupported:
        pass

    fields = sanitize_fields(
        {
            "api_key": "visible",
            "message": "authorization token leaked",
            "nan": float("nan"),
            "tuple": (1, 2),
            "object": Unsupported(),
            "large": "x" * 10_000,
            "nested": {"password": "visible", "value": [1, float("inf")]},
        }
    )

    assert fields["api_key"] == "<redacted>"
    assert fields["message"] == "<redacted>"
    assert fields["nan"] == "nan"
    assert fields["tuple"] == {"unsupported_type": "builtins.tuple"}
    unsupported = fields["object"]
    assert isinstance(unsupported, dict)
    assert str(unsupported["unsupported_type"]).endswith(".<locals>.Unsupported")
    assert "truncated" in str(fields["large"])
    assert fields["nested"] == {"password": "<redacted>", "value": [1, "inf"]}
    assert len(json.dumps(fields, ensure_ascii=False).encode("utf-8")) <= 30 * 1024


def test_shared_renderer_is_one_line_and_console_uses_it() -> None:
    event = Event(
        1,
        "2026-07-14T16:12:33.482000+00:00",
        "run",
        1842,
        "quantize-blocks",
        "info",
        "block.completed",
        {"message": "first\nsecond", "block": 5},
    )
    rendered = render_event_line(event)
    assert rendered == (
        '2026-07-14T16:12:33.482000Z 0001842 INFO    quantize-blocks block.completed '
        'block=5 message="first\\nsecond"'
    )
    assert rendered.count("\n") == 0

    stream = io.StringIO()
    ConsoleEventDestination(Severity.INFO, stream).accept(event)
    assert stream.getvalue() == rendered + "\n"


def test_console_renders_progress_checkpoints_for_captured_output() -> None:
    stream = io.StringIO()
    console = ConsoleEventDestination(Severity.INFO, stream)
    base = Event(1, "2026-01-01T00:00:00+00:00", "run", 1, "resident", "info", "", {})

    console.accept(
        Event(
            1,
            base.timestamp,
            "run",
            2,
            "resident",
            "info",
            "compression.progress_initialized",
            {
                "total_blocks": 2,
                "completed_blocks": 1,
                "completed_wall_seconds": 15.0,
                "mean_block_seconds": 15.0,
            },
        )
    )
    console.accept(
        Event(
            1,
            base.timestamp,
            "run",
            3,
            "resident",
            "info",
            "block.started",
            {"block": 1, "layers": 2},
        )
    )

    output = stream.getvalue()
    assert "compression.progress_initialized" not in output
    assert "Compressing Layers:" in output
    assert "1/2 [00:00:15<00:00:15, 15.0s/block]" in output
    assert "block=2/2" in output
    assert "\r" not in output


def test_console_renders_bounded_calibration_progress_for_captured_output() -> None:
    stream = io.StringIO()
    console = ConsoleEventDestination(Severity.INFO, stream)
    timestamp = "2026-01-01T00:00:00+00:00"

    for sequence, name, fields in (
        (1, "calibration.progress_initialized", {"total_batches": 2}),
        (
            2,
            "calibration.progress_updated",
            {"completed_batches": 1, "total_batches": 2},
        ),
        (
            3,
            "calibration.progress_completed",
            {"completed_batches": 2, "total_batches": 2},
        ),
    ):
        console.accept(
            Event(
                1,
                timestamp,
                "run",
                sequence,
                "resident",
                "info",
                name,
                fields,
            )
        )

    output = stream.getvalue()
    assert "calibration.progress_" not in output
    assert output.count("Calibrating:") == 2
    assert "0/2" in output
    assert "2/2" in output


def test_console_reports_long_calibration_liveness_for_tee_output() -> None:
    stream = io.StringIO()
    console = ConsoleEventDestination(Severity.INFO, stream)
    timestamp = "2026-01-01T00:00:00+00:00"

    for sequence, name, fields in (
        (1, "calibration.progress_initialized", {"total_batches": 256}),
        (2, "calibration.progress_updated", {"completed_batches": 1, "total_batches": 256}),
        (3, "calibration.progress_updated", {"completed_batches": 2, "total_batches": 256}),
        (4, "calibration.progress_updated", {"completed_batches": 13, "total_batches": 256}),
        (5, "calibration.progress_completed", {"completed_batches": 256, "total_batches": 256}),
    ):
        console.accept(
            Event(
                1,
                timestamp,
                "run",
                sequence,
                "resident",
                "info",
                name,
                fields,
            )
        )

    output = stream.getvalue()
    assert output.count("Calibrating:") == 4
    assert "0/256" in output
    assert "1/256" in output
    assert "2/256" not in output
    assert "13/256" in output
    assert "256/256" in output


def test_console_uses_one_carriage_return_progress_line_for_tty() -> None:
    stream = TtyStringIO()
    console = ConsoleEventDestination(Severity.INFO, stream)
    timestamp = "2026-01-01T00:00:00+00:00"
    console.accept(
        Event(
            1,
            timestamp,
            "run",
            1,
            "resident",
            "info",
            "compression.progress_initialized",
            {"total_blocks": 1, "completed_blocks": 0},
        )
    )
    console.accept(
        Event(
            1,
            timestamp,
            "run",
            2,
            "resident",
            "info",
            "block.started",
            {"block": 0, "layers": 1},
        )
    )
    console.accept(
        Event(
            1,
            timestamp,
            "run",
            3,
            "resident",
            "info",
            "block.completed",
            {"block": 0, "wall_seconds": 4.0},
        )
    )
    console.close()

    output = stream.getvalue()
    assert "\rCompressing Layers:" in output
    assert "100.0%" in output
    assert output.endswith("\n")


def test_human_views_split_memory_metrics_and_keep_oom_visible() -> None:
    boundary = Event(
        1,
        "2026-07-14T16:12:33.482000+00:00",
        "run",
        12,
        "resident",
        "info",
        "block.completed",
        {
            "block": 2,
            "final_loss": 1.25,
            "rank": 64,
            "cuda.allocated_bytes": 100,
            "gpu_peak_bytes": 150,
            "host.working_set_bytes": 200,
        },
    )

    run_view = project_event(boundary, "run")
    memory_view = project_event(boundary, "memory")

    assert run_view is not None
    assert run_view.fields == {"block": 2, "final_loss": 1.25, "rank": 64}
    assert memory_view is not None
    assert memory_view.fields == {
        "block": 2,
        "cuda.allocated_bytes": 100,
        "gpu_peak_bytes": 150,
        "host.working_set_bytes": 200,
        "rank": 64,
    }

    sample = Event(1, boundary.timestamp, "run", 13, "resource", "info", "resource.sample", boundary.fields)
    assert project_event(sample, "run") is None
    assert project_event(sample, "memory") is sample

    oom = Event(
        1,
        boundary.timestamp,
        "run",
        14,
        "resource",
        "error",
        "resource.oom_snapshot",
        {
            "error_type": "OutOfMemoryError",
            "oom_report_path": "state/oom-report-1.txt",
            "requested_bytes": 4 * 2**20,
            "cuda.allocated_bytes": 100,
        },
    )
    oom_run_view = project_event(oom, "run")
    assert oom_run_view is not None
    assert oom_run_view.fields == {
        "error_type": "OutOfMemoryError",
        "oom_report_path": "state/oom-report-1.txt",
    }
    assert project_event(oom, "memory") is oom

    stream = io.StringIO()
    console = ConsoleEventDestination(Severity.INFO, stream)
    console.accept(boundary)
    console.accept(sample)
    assert "final_loss=1.25" in stream.getvalue()
    assert "allocated_bytes" not in stream.getvalue()
    assert "resource.sample" not in stream.getvalue()


def test_torn_tail_is_quarantined_and_sequence_continues(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = Event(1, "2026-01-01T00:00:00+00:00", "old", 7, "run", "info", "run.started", {})
    path.write_bytes((json.dumps(asdict(first)) + "\n" + '{"sequence":8,"bad"').encode())

    last_sequence, recovered = prepare_event_stream(path)

    assert last_sequence == 7
    assert recovered == len(b'{"sequence":8,"bad"')
    assert path.with_name("events.jsonl.orphan").read_bytes() == b'{"sequence":8,"bad"'
    assert path.read_bytes().endswith(b"\n")

    with JsonlEventSink(path, "new") as sink:
        resumed = sink.emit("run", "info", "run.resumed")
    assert resumed is not None and resumed.sequence == 8


def test_last_sequence_scan_handles_large_history_without_full_parser(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"sequence": sequence, "payload": "x" * 100}) for sequence in range(1, 2001)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert prepare_event_stream(path) == (2000, 0)
