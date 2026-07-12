"""Run application service with all side effects supplied through callbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nanoquant.config.schema import RunConfig
from nanoquant.ports.artifact_store import ArtifactStore
from nanoquant.ports.event_sink import EventSink


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    artifacts: ArtifactStore
    events: EventSink


LegacyRunner = Callable[[RunConfig, ApplicationContext], tuple[str, ...]]


class QuantizeApplication:
    """Shared orchestration boundary used by Python, CLI, and runfiles."""

    def run(
        self, config: RunConfig, context: ApplicationContext, runner: LegacyRunner | None = None
    ) -> tuple[str, ...]:
        context.events.emit("run", "info", "configuration.accepted")
        if runner is None:
            context.events.emit("quantize", "warning", "pipeline.not_configured", code="RUN002")
            return ()
        with (
            context.events.span("quantize", "legacy_compatibility") if hasattr(context.events, "span") else _nullspan()
        ):
            return runner(config, context)


class _nullspan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
