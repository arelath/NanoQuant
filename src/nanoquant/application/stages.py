"""Generic typed stage lifecycle and explicit registry."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from nanoquant.config.codec import canonical_json
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
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
    recorder: PhaseRecorder = NULL_RECORDER


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
    request_context: dict[str, object] = {"request_type": type(request).__name__}
    diagnostic_request = getattr(request, "request", request)
    layer = getattr(diagnostic_request, "layer", None)
    if layer is not None:
        path = getattr(layer, "path", None)
        request_context["layer"] = path if isinstance(path, str) else str(layer)
        block = getattr(getattr(layer, "block", None), "index", None)
        if isinstance(block, int):
            request_context["block"] = block
    rank = getattr(diagnostic_request, "rank", getattr(diagnostic_request, "probe_rank", None))
    if isinstance(rank, int):
        request_context["rank"] = rank
    started = time.perf_counter()
    with context.recorder.phase("stage", stage=stage.name, version=stage.version):
        with context.recorder.phase("cancellation"):
            context.cancellation.raise_if_cancelled()
        with context.recorder.phase("event"):
            cast(Any, context.events).emit(
                stage.name,
                "info",
                "stage.started",
                version=stage.version,
                **request_context,
            )
        active_phase = "execute"
        try:
            with context.recorder.phase("execute"):
                result = stage.execute(request, context)
            active_phase = "validate"
            with context.recorder.phase("validate"):
                report = stage.validate(result, context)
                if not report.valid:
                    details = "; ".join(f"{finding.code}: {finding.message}" for finding in report.findings)
                    raise ValueError(f"stage output validation failed: {details}")
        except BaseException as error:
            with context.recorder.phase("event"):
                cast(Any, context.events).emit(
                    stage.name,
                    "error",
                    "stage.failed",
                    version=stage.version,
                    active_phase=active_phase,
                    wall_seconds=time.perf_counter() - started,
                    error_type=type(error).__name__,
                    error=str(error),
                    **request_context,
                )
            raise
        with context.recorder.phase("event"):
            cast(Any, context.events).emit(
                stage.name,
                "info",
                "stage.completed",
                version=stage.version,
                wall_seconds=time.perf_counter() - started,
                **request_context,
            )
        return result
