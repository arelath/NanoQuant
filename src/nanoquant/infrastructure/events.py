"""Structured event routing, local destinations, recovery, and rendering."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, TextIO
from uuid import uuid4

from nanoquant.infrastructure.environment import SECRET_PATTERN
from nanoquant.ports.event_sink import Event, Severity

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_MAX_STRING_BYTES = 4 * 1024
_MAX_FIELDS_BYTES = 30 * 1024
_MAX_DEPTH = 4


class EventWriteError(RuntimeError):
    """A required event destination can no longer satisfy the audit contract."""


class EventStreamError(ValueError):
    """The canonical event stream does not end in a valid event record."""


class EventDestination(Protocol):
    role: Literal["required", "optional"]
    threshold: Severity

    def accept(self, event: Event) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _truncate_string(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= _MAX_STRING_BYTES:
        return value
    suffix = f"…[truncated {len(encoded) - _MAX_STRING_BYTES} bytes]"
    budget = max(0, _MAX_STRING_BYTES - len(suffix.encode("utf-8")))
    prefix = encoded[:budget].decode("utf-8", errors="ignore")
    omitted = len(encoded) - len(prefix.encode("utf-8"))
    return f"{prefix}…[truncated {omitted} bytes]"


def _sanitize_value(value: object, *, key: str, depth: int) -> object:
    if SECRET_PATTERN.search(key):
        return "<redacted>"
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, str):
        if SECRET_PATTERN.search(value):
            return "<redacted>"
        return _truncate_string(value)
    if depth >= _MAX_DEPTH:
        return {"truncated_depth": depth + 1}
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_value(item, key=str(item_key), depth=depth + 1)
            for item_key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_sanitize_value(item, key=key, depth=depth + 1) for item in value]
    return {"unsupported_type": _qualified_type(value)}


def sanitize_fields(fields: Mapping[str, object]) -> dict[str, object]:
    """Return bounded, deterministic, redacted JSON-compatible event fields."""

    sanitized = {
        str(key): _sanitize_value(value, key=str(key), depth=0)
        for key, value in sorted(fields.items(), key=lambda pair: str(pair[0]))
    }
    encoded = json.dumps(sanitized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) <= _MAX_FIELDS_BYTES:
        return sanitized

    sizes = sorted(
        (
            len(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
            key,
        )
        for key, value in sanitized.items()
    )
    for original_bytes, key in reversed(sizes):
        sanitized[key] = {"truncated": True, "original_bytes": original_bytes}
        encoded = json.dumps(sanitized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(encoded) <= _MAX_FIELDS_BYTES:
            break
    return sanitized


def _render_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def render_event_line(event: Event) -> str:
    """Render one event as exactly one locale-independent physical line."""

    timestamp = event.timestamp
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
    details = " ".join(f"{key}={_render_value(value)}" for key, value in sorted(event.fields.items()))
    suffix = f" {details}" if details else ""
    return (
        f"{timestamp} {event.sequence:07d} {event.severity.upper():7} "
        f"{event.stage} {event.name}{suffix}"
    )


def event_from_dict(payload: Mapping[str, object]) -> Event:
    """Decode an event envelope while keeping schema-1 streams readable."""

    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise EventStreamError("event fields must be an object")
    schema_version = payload.get("schema_version")
    sequence = payload.get("sequence")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise EventStreamError("event schema_version must be an integer")
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise EventStreamError("event sequence must be an integer")
    try:
        return Event(
            schema_version=schema_version,
            timestamp=str(payload["timestamp"]),
            run_id=str(payload["run_id"]),
            sequence=sequence,
            stage=str(payload["stage"]),
            severity=Severity.parse(str(payload["severity"])).value,
            name=str(payload["name"]),
            fields={str(key): value for key, value in fields.items()},
            span_id=None if payload.get("span_id") is None else str(payload["span_id"]),
            parent_span_id=(
                None if payload.get("parent_span_id") is None else str(payload["parent_span_id"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EventStreamError("invalid event envelope") from exc


def read_event_prefix(path: str | Path) -> Iterator[Event]:
    """Yield the valid newline-terminated event prefix without modifying it."""

    event_path = Path(path)
    if not event_path.exists():
        return
    expected = 1
    with event_path.open("rb") as source:
        for raw_line in source:
            if not raw_line.endswith(b"\n"):
                break
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            if not isinstance(payload, dict):
                break
            try:
                event = event_from_dict(payload)
            except EventStreamError:
                break
            if event.sequence != expected:
                break
            yield event
            expected += 1


def read_last_event(path: str | Path) -> Event | None:
    line = _read_last_line(Path(path))
    if not line:
        return None
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventStreamError("event stream does not end in valid JSON") from exc
    if not isinstance(payload, dict):
        raise EventStreamError("event stream does not end in an event object")
    return event_from_dict(payload)


def event_stream_integrity(path: str | Path) -> str:
    """Classify an event stream without modifying its valid prefix."""

    event_path = Path(path)
    if not event_path.exists():
        return "missing"
    expected = 1
    with event_path.open("rb") as source:
        for raw_line in source:
            if not raw_line.endswith(b"\n"):
                return "torn"
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
                if not isinstance(payload, dict):
                    return "corrupt"
                event = event_from_dict(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, EventStreamError):
                return "corrupt"
            if event.sequence != expected:
                return "corrupt"
            expected += 1
    return "ok"


def render_event_log(events_path: str | Path, output_path: str | Path) -> None:
    """Atomically render the canonical valid event prefix as a text snapshot."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            for event in read_event_prefix(events_path):
                output.write(render_event_line(event) + "\n")
            output.flush()
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_last_line(path: Path) -> bytes:
    if not path.exists() or path.stat().st_size == 0:
        return b""
    with path.open("rb") as source:
        position = source.seek(0, 2)
        buffer = b""
        while position > 0:
            amount = min(64 * 1024, position)
            position -= amount
            source.seek(position)
            buffer = source.read(amount) + buffer
            stripped = buffer.rstrip(b"\r\n")
            separator = stripped.rfind(b"\n")
            if separator >= 0 or position == 0:
                return stripped[separator + 1 :].rstrip(b"\r")
    return b""


def _sequence_from_line(line: bytes) -> int:
    try:
        payload = json.loads(line.decode("utf-8"))
        sequence = payload["sequence"]
        if type(sequence) is not int or sequence < 1:
            raise ValueError("sequence must be a positive integer")
        return sequence
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise EventStreamError("event stream does not end in a valid event record") from exc


def prepare_event_stream(path: str | Path, *, recover_torn_tail: bool = True) -> tuple[int, int]:
    """Return the last sequence and optionally quarantine a provably torn final record."""

    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    if not event_path.exists() or event_path.stat().st_size == 0:
        return 0, 0

    recovered = 0
    with event_path.open("rb+") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(size - 1)
        ends_with_newline = stream.read(1) == b"\n"
        if not ends_with_newline:
            position = size
            buffer = b""
            separator_position = -1
            while position > 0:
                amount = min(64 * 1024, position)
                position -= amount
                stream.seek(position)
                buffer = stream.read(amount) + buffer
                separator = buffer.rfind(b"\n")
                if separator >= 0:
                    separator_position = position + separator
                    break
            candidate = buffer[separator + 1 :] if separator_position >= 0 else buffer
            try:
                _sequence_from_line(candidate.rstrip(b"\r"))
            except EventStreamError:
                if not recover_torn_tail:
                    raise
                recovered = len(candidate)
                with event_path.with_name(event_path.name + ".orphan").open("ab") as orphan:
                    orphan.write(candidate)
                    orphan.flush()
                stream.seek(separator_position + 1 if separator_position >= 0 else 0)
                stream.truncate()
                stream.flush()
            else:
                stream.seek(0, 2)
                stream.write(b"\n")
                stream.flush()

    last_line = _read_last_line(event_path)
    return (_sequence_from_line(last_line) if last_line else 0), recovered


class JsonlEventDestination:
    role: Literal["required", "optional"] = "required"

    def __init__(self, path: str | Path, threshold: Severity) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    def accept(self, event: Event) -> None:
        self._handle.write(
            json.dumps(
                asdict(event),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        self._handle.flush()

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


class ConsoleEventDestination:
    role: Literal["required", "optional"] = "optional"

    def __init__(self, threshold: Severity, stream: TextIO | None = None) -> None:
        self.threshold = threshold
        self._stream = stream

    def accept(self, event: Event) -> None:
        print(render_event_line(event), file=self._stream, flush=True)

    def flush(self) -> None:
        if self._stream is not None:
            self._stream.flush()

    def close(self) -> None:
        self.flush()


class CallbackEventDestination:
    role: Literal["required", "optional"] = "optional"

    def __init__(self, callback: Callable[[Event], None], threshold: Severity = Severity.DEBUG) -> None:
        self.threshold = threshold
        self._callback = callback

    def accept(self, event: Event) -> None:
        self._callback(event)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class EventRouter:
    """Create each event once and route it to required and optional destinations."""

    def __init__(
        self,
        run_id: str,
        *,
        event_level: Severity,
        destinations: tuple[EventDestination, ...],
        initial_sequence: int = 0,
    ) -> None:
        self.run_id = run_id
        self.event_level = Severity.parse(event_level)
        required = tuple(destination for destination in destinations if destination.role == "required")
        optional = tuple(destination for destination in destinations if destination.role == "optional")
        if not required:
            raise ValueError("at least one required event destination is required")
        self._destinations = required + optional
        self._disabled: set[int] = set()
        self._sequence = initial_sequence
        self._lock = threading.RLock()
        self._closed = False
        self._reporting_failure = False

    @staticmethod
    def _validate_identifier(value: str, kind: str) -> None:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"invalid event {kind}: {value!r}")

    def _event(
        self,
        stage: str,
        severity: Severity,
        name: str,
        fields: Mapping[str, object],
        span_id: str | None,
        parent_span_id: str | None,
    ) -> Event:
        self._sequence += 1
        return Event(
            1,
            datetime.now(timezone.utc).isoformat(),
            self.run_id,
            self._sequence,
            stage,
            severity.value,
            name,
            sanitize_fields(fields),
            span_id,
            parent_span_id,
        )

    def _deliver(self, event: Event, *, report_optional_failure: bool) -> None:
        severity = Severity.parse(event.severity)
        for index, destination in enumerate(self._destinations):
            if index in self._disabled or severity.rank < destination.threshold.rank:
                continue
            try:
                destination.accept(event)
            except Exception as exc:
                if destination.role == "required":
                    raise EventWriteError(
                        f"required event destination failed: {type(destination).__name__}"
                    ) from exc
                self._disabled.add(index)
                if report_optional_failure and not self._reporting_failure:
                    self._reporting_failure = True
                    try:
                        diagnostic = self._event(
                            "observability",
                            Severity.WARNING,
                            "observability.destination_disabled",
                            {
                                "destination": type(destination).__name__,
                                "error_type": type(exc).__name__,
                            },
                            None,
                            None,
                        )
                        self._deliver(diagnostic, report_optional_failure=False)
                    finally:
                        self._reporting_failure = False

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
        parsed = Severity.parse(severity)
        self._validate_identifier(stage, "stage")
        self._validate_identifier(name, "name")
        if parsed.rank < self.event_level.rank:
            return None
        with self._lock:
            if self._closed:
                raise EventWriteError("event router is closed")
            event = self._event(stage, parsed, name, fields, span_id, parent_span_id)
            self._deliver(event, report_optional_failure=True)
            return event

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

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            first_required_error: EventWriteError | None = None
            for index, destination in enumerate(self._destinations):
                if index in self._disabled:
                    continue
                try:
                    destination.flush()
                except Exception as exc:
                    if destination.role == "required" and first_required_error is None:
                        first_required_error = EventWriteError(
                            f"required event destination flush failed: {type(destination).__name__}"
                        )
                        first_required_error.__cause__ = exc
                    else:
                        self._disabled.add(index)
            for index in reversed(range(len(self._destinations))):
                destination = self._destinations[index]
                try:
                    destination.close()
                except Exception as exc:
                    if destination.role == "required" and first_required_error is None:
                        first_required_error = EventWriteError(
                            f"required event destination close failed: {type(destination).__name__}"
                        )
                        first_required_error.__cause__ = exc
            self._closed = True
            if first_required_error is not None:
                raise first_required_error

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                raise EventWriteError("event router is closed")
            for index, destination in enumerate(self._destinations):
                if index in self._disabled:
                    continue
                try:
                    destination.flush()
                except Exception as exc:
                    if destination.role == "required":
                        raise EventWriteError(
                            f"required event destination flush failed: {type(destination).__name__}"
                        ) from exc
                    self._disabled.add(index)


class JsonlEventSink:
    """Compatibility JSONL sink; events are flushed per emit but not fsynced."""

    def __init__(self, path: str | Path, run_id: str, observer: Callable[[Event], None] | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id
        initial_sequence, recovered_bytes = prepare_event_stream(self.path)
        destinations: list[EventDestination] = [JsonlEventDestination(self.path, Severity.DEBUG)]
        if observer is not None:
            destinations.append(CallbackEventDestination(observer))
        self._router = EventRouter(
            run_id,
            event_level=Severity.DEBUG,
            destinations=tuple(destinations),
            initial_sequence=initial_sequence,
        )
        if recovered_bytes:
            self.emit(
                "observability",
                "warning",
                "observability.tail_recovered",
                quarantined_bytes=recovered_bytes,
            )

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
        return self._router.emit(
            stage,
            severity,
            name,
            span_id=span_id,
            parent_span_id=parent_span_id,
            **fields,
        )

    @contextmanager
    def span(self, stage: str, name: str, parent_span_id: str | None = None, **fields: object) -> Iterator[str]:
        with self._router.span(stage, name, parent_span_id, **fields) as span_id:
            yield span_id

    def close(self) -> None:
        self._router.close()

    def __enter__(self) -> JsonlEventSink:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class ConsoleRenderer:
    """Compatibility callback using the shared deterministic line renderer."""

    def __call__(self, event: Event) -> None:
        print(render_event_line(event), flush=True)
