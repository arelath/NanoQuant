"""Generic typed stage lifecycle and explicit registry."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from nanoquant.config.codec import canonical_json
from nanoquant.domain.stages import HostInventory, ResourceEstimate, ValidationReport
from nanoquant.ports.artifact_store import ArtifactStore
from nanoquant.ports.event_sink import EventSink
from nanoquant.ports.executor import Executor
from nanoquant.ports.tensor_store import TensorStore

RequestT = TypeVar("RequestT", contravariant=True)
ResultT = TypeVar("ResultT")


class CancellationToken(Protocol):
    def raise_if_cancelled(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StageContext:
    run_id: str
    executor: Executor
    artifact_store: ArtifactStore
    tensor_store: TensorStore
    events: EventSink
    cancellation: CancellationToken


class Stage(Generic[RequestT, ResultT], Protocol):
    name: str
    version: str

    def estimate(self, request: RequestT, host: HostInventory) -> ResourceEstimate: ...
    def execute(self, request: RequestT, context: StageContext) -> ResultT: ...
    def validate(self, result: ResultT, context: StageContext) -> ValidationReport: ...


class StageRegistry:
    def __init__(self) -> None:
        self._stages: dict[str, Stage[object, object]] = {}

    def register(self, stage: Stage[object, object]) -> None:
        if not stage.name or not stage.version:
            raise ValueError("stage name and version are required")
        if stage.name in self._stages:
            raise ValueError(f"stage already registered: {stage.name}")
        self._stages[stage.name] = stage

    def get(self, name: str) -> Stage[object, object]:
        try:
            return self._stages[name]
        except KeyError as exc:
            raise KeyError(f"stage is not registered: {name}") from exc

    def semantic_key(self, name: str, request: object) -> str:
        stage = self.get(name)
        payload = canonical_json({"stage": stage.name, "version": stage.version, "request": request})
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def execute_stage(stage: Stage[RequestT, ResultT], request: RequestT, context: StageContext) -> ResultT:
    context.cancellation.raise_if_cancelled()
    context.events.emit(stage.name, "info", "stage.started", version=stage.version)
    try:
        result = stage.execute(request, context)
        report = stage.validate(result, context)
        if not report.valid:
            details = "; ".join(f"{finding.code}: {finding.message}" for finding in report.findings)
            raise ValueError(f"stage output validation failed: {details}")
    except BaseException as error:
        context.events.emit(
            stage.name,
            "error",
            "stage.failed",
            version=stage.version,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    context.events.emit(stage.name, "info", "stage.completed", version=stage.version)
    return result
