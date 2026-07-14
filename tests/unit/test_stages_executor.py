import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from nanoquant.application.stages import StageContext, StageRegistry, execute_stage
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
from nanoquant.domain.stages import HostInventory, ResourceEstimate, ValidationFinding, ValidationReport
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.profiling import Profiler
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True)
class Request:
    value: int


class DoubleStage:
    name = "double"
    version = "1"

    def estimate(self, request: Request, host: HostInventory) -> ResourceEstimate:
        return ResourceEstimate(peak_cpu_bytes=8)

    def execute(self, request: Request, context: StageContext) -> int:
        return request.value * 2

    def validate(self, result: int, context: StageContext) -> ValidationReport:
        return ValidationReport(() if result >= 0 else (ValidationFinding("NEG001", "negative"),))


def _context(
    tmp_path: Path,
    cancellation: Cancellation | None = None,
    recorder: PhaseRecorder | None = None,
) -> StageContext:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    return StageContext(
        "run",
        ResidentExecutor(),
        artifacts,
        LocalTensorStore(artifacts),
        JsonlEventSink(tmp_path / "events.jsonl", "run"),
        cancellation or Cancellation(),
        recorder=recorder or NULL_RECORDER,
    )


def test_stage_registry_semantics_lifecycle_and_validation(tmp_path: Path) -> None:
    registry = StageRegistry()
    stage = DoubleStage()
    registry.register(stage)  # type: ignore[arg-type]
    assert registry.semantic_key("double", Request(2)) == registry.semantic_key("double", Request(2))
    assert registry.semantic_key("double", Request(2)) != registry.semantic_key("double", Request(3))
    assert execute_stage(stage, Request(2), _context(tmp_path)) == 4
    success_events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["name"] for event in success_events] == ["stage.started", "stage.completed"]
    assert success_events[0]["fields"]["request_type"] == "Request"
    assert success_events[1]["fields"]["wall_seconds"] >= 0
    with pytest.raises(ValueError, match="NEG001"):
        execute_stage(stage, Request(-1), _context(tmp_path / "bad"))
    failure_events = [
        json.loads(line)
        for line in (tmp_path / "bad" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed = next(event for event in failure_events if event["name"] == "stage.failed")
    assert failed["fields"]["active_phase"] == "validate"
    assert failed["fields"]["request_type"] == "Request"
    assert failed["fields"]["wall_seconds"] >= 0
    with pytest.raises(ValueError, match="already"):
        registry.register(stage)  # type: ignore[arg-type]


def test_resident_executor_reuses_buffers_and_scopes_tensor_transfers() -> None:
    executor = ResidentExecutor()
    first = executor.buffer("workspace", (2, 3), torch.float32, "cpu")
    second = executor.buffer("workspace", (2, 3), torch.float32, "cpu")
    assert first.data_ptr() == second.data_ptr()
    value = torch.ones(2)
    with executor.tensor_lease(value, "cpu") as leased:
        assert leased is value
    executor.release()
    third = executor.buffer("workspace", (2, 3), torch.float32, "cpu")
    assert third.data_ptr() != first.data_ptr()


def test_cancellation_prevents_stage_execution(tmp_path: Path) -> None:
    cancellation = Cancellation()
    cancellation.cancel()
    with pytest.raises(InterruptedError):
        execute_stage(DoubleStage(), Request(1), _context(tmp_path, cancellation))


def test_stage_executor_profiles_lifecycle_without_changing_result_or_failure(tmp_path: Path) -> None:
    profiler = Profiler(
        ProfilingConfig(level=ProfilingLevel.MACRO, emit_span_events=False),
        run_id="stage-profile",
    )
    assert execute_stage(DoubleStage(), Request(2), _context(tmp_path / "success", recorder=profiler)) == 4
    with pytest.raises(ValueError, match="NEG001"):
        execute_stage(
            DoubleStage(),
            Request(-1),
            _context(tmp_path / "failure", recorder=profiler),
        )

    payload = profiler.snapshot()
    phases = {str(phase["path"]): phase for phase in payload["phases"]}  # type: ignore[index]
    assert {"stage/cancellation", "stage/event", "stage/execute", "stage/validate"} <= set(phases)
    assert phases["stage"]["count"] == 2
    assert phases["stage"]["failed_count"] == 1
    assert phases["stage/validate"]["failed_count"] == 1
