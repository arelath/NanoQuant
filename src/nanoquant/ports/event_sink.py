from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Severity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        return tuple(Severity).index(self)

    @classmethod
    def parse(cls, value: str | Severity) -> Severity:
        return value if isinstance(value, cls) else cls(value)


@dataclass(frozen=True, slots=True)
class Event:
    schema_version: int
    timestamp: str
    run_id: str
    sequence: int
    stage: str
    severity: str
    name: str
    fields: dict[str, object]
    span_id: str | None = None
    parent_span_id: str | None = None


class EventSink(Protocol):
    def emit(
        self,
        stage: str,
        severity: str,
        name: str,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        **fields: object,
    ) -> Event | None: ...


def capture_oom_if_supported(
    sink: EventSink,
    error: BaseException,
    *,
    stage: str | None = None,
    block: int | None = None,
    layer: str | None = None,
) -> None:
    """Invoke an infrastructure OOM observer without requiring it of every sink."""

    callback = getattr(sink, "capture_oom", None)
    if callable(callback):
        callback(error, stage=stage, block=block, layer=layer)
