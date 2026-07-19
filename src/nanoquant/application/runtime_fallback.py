"""Finite placement/batch OOM fallbacks with explicit algorithm-change labels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

import torch

from nanoquant.ports.event_sink import EventSink, capture_oom_if_supported

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PlacementState:
    batch_size: int
    activation_tier: str
    algorithm: str


@dataclass(frozen=True, slots=True)
class FallbackRevision:
    attempt: int
    action: str
    before: PlacementState
    after: PlacementState
    algorithm_changed: bool


def _is_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in message


def run_with_runtime_fallback(
    operation: Callable[[PlacementState], T],
    initial: PlacementState,
    actions: tuple[str, ...],
    events: EventSink,
) -> tuple[T, tuple[FallbackRevision, ...]]:
    if initial.batch_size <= 0:
        raise ValueError("runtime fallback batch size must be positive")
    state = initial
    revisions: list[FallbackRevision] = []
    attempted: set[str] = set()
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation(state), tuple(revisions)
        except BaseException as error:
            if not _is_oom(error):
                raise
            capture_oom_if_supported(events, error, stage="runtime")
            action = next((candidate for candidate in actions if candidate not in attempted), "fail")
            attempted.add(action)
            before = state
            if action in {"reduce_stage_batch_size", "reduce_batch_size"} and state.batch_size > 1:
                state = replace(state, batch_size=max(1, state.batch_size // 2))
                algorithm_changed = False
            elif action == "move_activations_down_one_tier":
                next_tier = "ram" if state.activation_tier in {"cuda", "pinned_ram"} else "mmap"
                state = replace(state, activation_tier=next_tier)
                algorithm_changed = False
            elif action == "move_activation_store_to_pageable_ram":
                state = replace(state, activation_tier="ram")
                algorithm_changed = False
            elif action == "move_activation_store_to_mmap":
                state = replace(state, activation_tier="mmap")
                algorithm_changed = False
            elif action == "forward_only":
                state = replace(state, algorithm="forward_only")
                algorithm_changed = True
            elif action == "fail":
                events.emit("runtime", "error", "runtime.oom_unrecoverable", code="RES002", attempt=attempt)
                raise
            else:
                events.emit("runtime", "error", "runtime.oom_invalid_fallback", code="RES003", action=action)
                raise ValueError(f"RES003 unsupported runtime fallback action: {action}") from error
            revision = FallbackRevision(attempt, action, before, state, algorithm_changed)
            revisions.append(revision)
            events.emit(
                "runtime",
                "warning",
                "runtime.oom_fallback",
                code="RES002",
                attempt=attempt,
                action=action,
                batch_size=state.batch_size,
                activation_tier=state.activation_tier,
                algorithm=state.algorithm,
                algorithm_changed=algorithm_changed,
            )
