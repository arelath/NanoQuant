"""Composable timing and event spans for application business operations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
from nanoquant.ports.event_sink import EventSink, Severity


@dataclass(frozen=True, slots=True)
class TelemetryContext:
    """Combine phase recording and lifecycle events in one scoped operation."""

    events: EventSink
    stage: str
    recorder: PhaseRecorder = NULL_RECORDER

    @contextmanager
    def operation(
        self,
        name: str,
        *,
        phase: str | None = None,
        severity: Severity = Severity.INFO,
        fields: dict[str, object] | None = None,
        **extra_fields: object,
    ) -> Iterator[None]:
        payload = dict(fields or {})
        payload.update(extra_fields)
        phase_name = name if phase is None else phase
        started = time.perf_counter()
        cast(Any, self.events).emit(
            self.stage, severity.value, f"{name}.started", **payload
        )
        try:
            with self.recorder.phase(phase_name, **payload):
                yield
        except BaseException as error:
            if not hasattr(error, "nanoquant_operation"):
                try:
                    error.__dict__["nanoquant_operation"] = name
                except Exception:
                    pass
            cast(Any, self.events).emit(
                self.stage,
                Severity.ERROR.value,
                f"{name}.failed",
                error_type=type(error).__name__,
                error=str(error),
                wall_seconds=time.perf_counter() - started,
                **payload,
            )
            raise
        cast(Any, self.events).emit(
            self.stage,
            severity.value,
            f"{name}.completed",
            wall_seconds=time.perf_counter() - started,
            **payload,
        )

    def child(self, recorder: PhaseRecorder) -> TelemetryContext:
        """Keep the event stage while selecting a more specific phase recorder."""

        return TelemetryContext(self.events, self.stage, recorder)


__all__ = ["TelemetryContext"]
