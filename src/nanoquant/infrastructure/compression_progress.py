"""Console-only compression progress and ETA projection from structured events."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Literal

from nanoquant.ports.event_sink import Event

ProgressAction = Literal["none", "refresh", "checkpoint", "finish"]

_BAR_WIDTH = 24
_EMA_ALPHA = 0.3
_CALIBRATION_CHECKPOINT_COUNT = 20


def _integer_field(event: Event, name: str) -> int | None:
    value = event.fields.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = int(value)
    return converted if converted >= 0 else None


def _float_field(event: Event, name: str) -> float | None:
    value = event.fields.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted >= 0 else None


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    rounded = max(0, int(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


class CompressionProgress:
    """Fold resident block/layer events into a tqdm-style progress line."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._phase: Literal["pending", "calibration", "compression"] = "pending"
        self._calibration_total_batches = 0
        self._calibration_completed_batches = 0
        self._calibration_started: float | None = None
        self._calibration_last_updated: float | None = None
        self._calibration_last_completed = 0
        self._seconds_per_calibration_batch: float | None = None
        self._total_blocks = 0
        self._completed_blocks = 0
        self._completed_wall_seconds = 0.0
        self._seconds_per_block: float | None = None
        self._active_block: int | None = None
        self._active_block_started: float | None = None
        self._layer_name: str | None = None
        self._layer_position = 0
        self._layer_count = 0
        self._layer_is_active = False
        self._seen_live_blocks: set[int] = set()
        self._active = False
        self._status = "pending"

    @property
    def active(self) -> bool:
        return self._active

    def observe(self, event: Event) -> ProgressAction:
        """Update progress state and report how a console should redraw it."""

        if event.name == "calibration.progress_initialized":
            total = _integer_field(event, "total_batches")
            if total is None or total <= 0:
                return "none"
            now = self._clock()
            self._phase = "calibration"
            self._calibration_total_batches = total
            self._calibration_completed_batches = 0
            self._calibration_started = now
            self._calibration_last_updated = now
            self._calibration_last_completed = 0
            self._seconds_per_calibration_batch = None
            self._active = True
            self._status = "running"
            return "checkpoint"

        if event.name == "calibration.progress_updated" and self._phase == "calibration":
            completed = _integer_field(event, "completed_batches")
            if completed is None:
                return "none"
            completed = min(self._calibration_total_batches, completed)
            now = self._clock()
            advanced = completed - self._calibration_last_completed
            if advanced > 0 and self._calibration_last_updated is not None:
                observed = max(0.0, now - self._calibration_last_updated) / advanced
                self._seconds_per_calibration_batch = (
                    observed
                    if self._seconds_per_calibration_batch is None
                    else (1.0 - _EMA_ALPHA) * self._seconds_per_calibration_batch
                    + _EMA_ALPHA * observed
                )
                self._calibration_last_updated = now
                self._calibration_last_completed = completed
            self._calibration_completed_batches = max(
                self._calibration_completed_batches,
                completed,
            )
            # Pipes such as ``python ... | tee`` are non-TTY streams. The
            # console prints checkpoints but deliberately suppresses refreshes,
            # so promote a bounded number of calibration updates to checkpoints
            # to make long Fisher passes visibly live in RunPod logs.
            checkpoint_stride = math.ceil(
                self._calibration_total_batches / _CALIBRATION_CHECKPOINT_COUNT
            )
            if (
                checkpoint_stride > 1
                and completed < self._calibration_total_batches
                and (completed == 1 or completed % checkpoint_stride == 0)
            ):
                return "checkpoint"
            return "refresh"

        if event.name == "calibration.progress_completed" and self._phase == "calibration":
            self._calibration_completed_batches = self._calibration_total_batches
            self._active = False
            self._status = "completed"
            return "finish"

        if event.name == "compression.progress_initialized":
            total = _integer_field(event, "total_blocks")
            completed = _integer_field(event, "completed_blocks")
            if total is None or total <= 0:
                return "none"
            self._phase = "compression"
            self._total_blocks = total
            self._completed_blocks = min(total, completed or 0)
            self._completed_wall_seconds = _float_field(event, "completed_wall_seconds") or 0.0
            self._seconds_per_block = _float_field(event, "mean_block_seconds")
            self._active_block = None
            self._active_block_started = None
            self._layer_name = None
            self._layer_position = 0
            self._layer_count = 0
            self._layer_is_active = False
            self._seen_live_blocks.clear()
            self._active = self._completed_blocks < self._total_blocks
            self._status = "running" if self._active else "completed"
            return "checkpoint" if self._active else "finish"

        if self._total_blocks <= 0:
            return "none"

        if event.name == "block.started":
            block = _integer_field(event, "block")
            if block is None:
                return "none"
            completed = _integer_field(event, "completed_blocks")
            if completed is not None:
                self._completed_blocks = min(
                    self._total_blocks,
                    max(self._completed_blocks, completed),
                )
            self._active_block = block
            self._active_block_started = self._clock()
            self._layer_name = None
            self._layer_position = 0
            self._layer_count = _integer_field(event, "layers") or 0
            self._layer_is_active = False
            self._active = True
            self._status = "running"
            return "checkpoint"

        if event.name == "layer.started" and self._active:
            position = _integer_field(event, "position")
            if position is not None:
                self._layer_position = min(self._layer_count, position)
            layer = event.fields.get("layer")
            self._layer_name = layer if isinstance(layer, str) else self._layer_name
            self._layer_is_active = True
            return "refresh"

        if event.name == "layer.completed" and self._active:
            if self._layer_count:
                self._layer_position = min(self._layer_count, self._layer_position + 1)
            layer = event.fields.get("layer")
            self._layer_name = layer if isinstance(layer, str) else self._layer_name
            self._layer_is_active = False
            return "checkpoint"

        if event.name == "block.completed":
            block = _integer_field(event, "block")
            duration = _float_field(event, "wall_seconds")
            if block is not None and block not in self._seen_live_blocks:
                self._seen_live_blocks.add(block)
                self._completed_blocks = min(
                    self._total_blocks,
                    max(self._completed_blocks + 1, block + 1),
                )
                if duration is not None:
                    self._completed_wall_seconds += duration
                    self._seconds_per_block = (
                        duration
                        if self._seconds_per_block is None
                        else (1.0 - _EMA_ALPHA) * self._seconds_per_block + _EMA_ALPHA * duration
                    )
            self._active_block = None
            self._active_block_started = None
            self._layer_name = None
            self._layer_position = 0
            self._layer_count = 0
            self._layer_is_active = False
            if self._completed_blocks >= self._total_blocks:
                self._active = False
                self._status = "completed"
                return "finish"
            self._active = True
            return "checkpoint"

        if event.name in {"run.failed", "run.interrupted"} and self._active:
            self._active = False
            self._status = event.name.removeprefix("run.")
            return "finish"
        return "none"

    def _progress_units(self) -> float:
        progress = float(self._completed_blocks)
        if self._active_block is not None and self._layer_count > 0:
            progress += min(0.99, self._layer_position / self._layer_count)
        return min(float(self._total_blocks), progress)

    def _elapsed_seconds(self) -> float:
        elapsed = self._completed_wall_seconds
        if self._active_block_started is not None:
            elapsed += max(0.0, self._clock() - self._active_block_started)
        return elapsed

    def render(self) -> str:
        """Render the current state as one bounded, carriage-return-safe line."""

        if self._phase == "calibration":
            total = self._calibration_total_batches
            completed = self._calibration_completed_batches
            fraction = completed / total if total else 0.0
            filled = min(_BAR_WIDTH, int(fraction * _BAR_WIDTH))
            bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
            elapsed = (
                None
                if self._calibration_started is None
                else max(0.0, self._clock() - self._calibration_started)
            )
            eta = (
                None
                if self._seconds_per_calibration_batch is None
                else (total - completed) * self._seconds_per_calibration_batch
            )
            rate = (
                "?s/batch"
                if self._seconds_per_calibration_batch is None
                else f"{self._seconds_per_calibration_batch:.1f}s/batch"
            )
            suffix = ""
            if self._status not in {"pending", "running", "completed"}:
                suffix = f" status={self._status}"
            return (
                f"Calibrating: {fraction * 100:5.1f}%|{bar}| "
                f"{completed}/{total} "
                f"[{_duration(elapsed)}<{_duration(eta)}, {rate}]{suffix}"
            )

        progress = self._progress_units()
        fraction = progress / self._total_blocks if self._total_blocks else 0.0
        filled = min(_BAR_WIDTH, int(fraction * _BAR_WIDTH))
        bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
        eta = (
            None
            if self._seconds_per_block is None
            else max(0.0, self._total_blocks - progress) * self._seconds_per_block
        )
        rate = "?s/block" if self._seconds_per_block is None else f"{self._seconds_per_block:.1f}s/block"
        suffix = ""
        if self._active_block is not None:
            suffix = f" block={self._active_block + 1}/{self._total_blocks}"
            if self._layer_name is not None and self._layer_count:
                visible_position = min(
                    self._layer_count,
                    self._layer_position + int(self._layer_is_active),
                )
                suffix += f" layer={visible_position}/{self._layer_count} {self._layer_name}"
        if self._status not in {"pending", "running", "completed"}:
            suffix += f" status={self._status}"
        return (
            f"Compressing Layers: {fraction * 100:5.1f}%|{bar}| "
            f"{self._completed_blocks}/{self._total_blocks} "
            f"[{_duration(self._elapsed_seconds())}<{_duration(eta)}, {rate}]{suffix}"
        )
