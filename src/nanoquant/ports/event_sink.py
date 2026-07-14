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
