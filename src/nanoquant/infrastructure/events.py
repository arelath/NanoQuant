"""Structured event envelope and local sinks."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from nanoquant.ports.event_sink import Event


class JsonlEventSink:
    """Thread-safe monotonic JSONL sink; each event is flushed durably."""

    def __init__(self, path: str | Path, run_id: str, observer: Callable[[Event], None] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._sequence = self._read_last_sequence()
        self._lock = threading.Lock()
        self._observer = observer
        # Held open for the sink's lifetime instead of reopened per emit; append mode
        # ("a") seeks to end-of-file on every write, so this is safe alongside other
        # readers/writers of the same path (verified for the Windows target platform).
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    def _read_last_sequence(self) -> int:
        if not self.path.exists():
            return 0
        last = 0
        with self.path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    last = max(last, int(json.loads(line)["sequence"]))
        return last

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
        with self._lock:
            self._sequence += 1
            event = Event(
                1,
                datetime.now(timezone.utc).isoformat(),
                self.run_id,
                self._sequence,
                stage,
                severity,
                name,
                fields,
                span_id,
                parent_span_id,
            )
            self._handle.write(json.dumps(asdict(event), sort_keys=True, separators=(",", ":"), default=str) + "\n")
            self._handle.flush()
        if self._observer is not None:
            self._observer(event)
        return event

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> JsonlEventSink:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @contextmanager
    def span(self, stage: str, name: str, parent_span_id: str | None = None, **fields: object) -> Iterator[str]:
        span_id = uuid4().hex
        self.emit(stage, "info", f"{name}.started", span_id=span_id, parent_span_id=parent_span_id, **fields)
        try:
            yield span_id
        except BaseException as exc:
            self.emit(
                stage,
                "error",
                f"{name}.failed",
                span_id=span_id,
                parent_span_id=parent_span_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        else:
            self.emit(stage, "info", f"{name}.completed", span_id=span_id, parent_span_id=parent_span_id)


class ConsoleRenderer:
    """Concise output generated directly from event fields."""

    def __call__(self, event: Event) -> None:
        detail = " ".join(f"{key}={value}" for key, value in sorted(event.fields.items()))
        suffix = f" {detail}" if detail else ""
        print(f"[{event.sequence:05d}] {event.severity.upper():7} {event.stage}: {event.name}{suffix}")
