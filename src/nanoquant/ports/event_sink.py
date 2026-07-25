from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class LayerCommittedPayload:
    block: int
    layer: str
    artifact_id: str
    journal_sequence: int
    rank: int
    accepted_attempt: int
    actual_bits: int
    extra_retry_bits: int
    weighted_error: float
    raw_error: float


@dataclass(frozen=True, slots=True)
class LlamaCppQualityStartedPayload:
    gguf: str
    gguf_sha256: str
    parallel: int


@dataclass(frozen=True, slots=True)
class LlamaCppQualityCompletedPayload:
    output: str
    reused: bool
    wall_seconds: float | None


def emit_layer_committed(
    sink: EventSink,
    payload: LayerCommittedPayload,
    *,
    stage: str = "resident-quantization",
) -> Event | None:
    return sink.emit(stage, Severity.INFO.value, "layer.committed", **asdict(payload))


def emit_llamacpp_quality_started(
    sink: EventSink,
    payload: LlamaCppQualityStartedPayload,
) -> Event | None:
    return sink.emit("quality", Severity.INFO.value, "quality.llamacpp.started", **asdict(payload))


def emit_llamacpp_quality_completed(
    sink: EventSink,
    payload: LlamaCppQualityCompletedPayload,
) -> Event | None:
    return sink.emit("quality", Severity.INFO.value, "quality.llamacpp.completed", **asdict(payload))


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
