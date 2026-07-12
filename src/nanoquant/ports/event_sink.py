from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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
    ) -> Event: ...
