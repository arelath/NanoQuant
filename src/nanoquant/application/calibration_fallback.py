"""Finite, event-visible calibration OOM fallback policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

from nanoquant.ports.event_sink import EventSink

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CalibrationPlanRevision:
    attempt: int
    action: str
    method: str
    device: str
    algorithm_changed: bool


@dataclass(frozen=True, slots=True)
class CalibrationExecutionResult:
    value: object
    revisions: tuple[CalibrationPlanRevision, ...]


CalibrationOperation = Callable[[str, str], T]


def _is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or "cuda out of memory" in message


def run_with_oom_fallback(
    operation: CalibrationOperation[T], method: str, device: str, actions: tuple[str, ...], events: EventSink
) -> tuple[T, tuple[CalibrationPlanRevision, ...]]:
    current_method = method
    current_device = device
    revisions: list[CalibrationPlanRevision] = []
    attempted_actions: set[str] = set()
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation(current_method, current_device), tuple(revisions)
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            action = next((candidate for candidate in actions if candidate not in attempted_actions), "fail")
            attempted_actions.add(action)
            if action == "fail":
                events.emit("calibration", "error", "calibration.oom_unrecoverable", code="CAL005", attempt=attempt)
                raise
            if action == "cpu_offload":
                current_device = "cpu"
                algorithm_changed = False
            elif action == "forward_only":
                current_method = "forward_only"
                algorithm_changed = True
            else:
                events.emit("calibration", "error", "calibration.oom_invalid_fallback", code="CAL006", action=action)
                raise ValueError(f"CAL006 unsupported calibration fallback action: {action}") from error
            revision = CalibrationPlanRevision(attempt, action, current_method, current_device, algorithm_changed)
            revisions.append(revision)
            events.emit(
                "calibration",
                "warning",
                "calibration.oom_fallback",
                code="CAL005",
                attempt=attempt,
                action=action,
                method=current_method,
                device=current_device,
                algorithm_changed=algorithm_changed,
            )
