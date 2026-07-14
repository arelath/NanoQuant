from pathlib import Path

import pytest
import torch

from nanoquant.application.runtime_fallback import PlacementState, run_with_runtime_fallback
from nanoquant.infrastructure.device_memory import ResourceEventSink
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.ports.event_sink import Severity


def test_runtime_oom_fallbacks_are_finite_and_label_algorithm_changes(tmp_path: Path) -> None:
    calls: list[PlacementState] = []

    def operation(state: PlacementState) -> str:
        calls.append(state)
        if len(calls) < 4:
            raise torch.OutOfMemoryError()
        return "ok"

    value, revisions = run_with_runtime_fallback(
        operation,
        PlacementState(8, "cuda", "online_fisher"),
        ("reduce_stage_batch_size", "move_activation_store_to_mmap", "forward_only", "fail"),
        JsonlEventSink(tmp_path / "events.jsonl", "run"),
    )

    assert value == "ok"
    assert [(state.batch_size, state.activation_tier, state.algorithm) for state in calls] == [
        (8, "cuda", "online_fisher"),
        (4, "cuda", "online_fisher"),
        (4, "mmap", "online_fisher"),
        (4, "mmap", "forward_only"),
    ]
    assert [revision.algorithm_changed for revision in revisions] == [False, False, True]


def test_runtime_fallback_terminates_and_rejects_unknown_actions(tmp_path: Path) -> None:
    sink = JsonlEventSink(tmp_path / "events.jsonl", "run")

    def operation(_state: PlacementState) -> None:
        raise torch.OutOfMemoryError()

    with pytest.raises(torch.OutOfMemoryError):
        run_with_runtime_fallback(operation, PlacementState(1, "mmap", "forward_only"), ("fail",), sink)
    with pytest.raises(ValueError, match="RES003"):
        run_with_runtime_fallback(operation, PlacementState(1, "cuda", "online_fisher"), ("magic",), sink)


def test_runtime_fallback_captures_forensics_before_retry(tmp_path: Path) -> None:
    captured: list[tuple[BaseException, str | None]] = []
    sink = ResourceEventSink(
        JsonlEventSink(tmp_path / "events.jsonl", "run"),
        Severity.INFO,
        oom_callback=lambda error, stage, _block, _layer: captured.append((error, stage)),
    )
    calls = 0

    def operation(_state: PlacementState) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return "ok"

    value, _revisions = run_with_runtime_fallback(
        operation,
        PlacementState(2, "cuda", "online_fisher"),
        ("reduce_stage_batch_size", "fail"),
        sink,
    )

    assert value == "ok"
    assert len(captured) == 1
    assert isinstance(captured[0][0], torch.OutOfMemoryError)
    assert captured[0][1] == "runtime"
