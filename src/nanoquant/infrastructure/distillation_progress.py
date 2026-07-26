"""Rate-limited console progress for long-running global distillation."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from typing import TextIO

_MIB = 1024 * 1024


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    rounded = max(0, int(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def _integer(fields: Mapping[str, object], name: str) -> int:
    value = fields.get(name)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _float(fields: Mapping[str, object], name: str) -> float:
    value = fields.get(name)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


class DistillationProgressLogger:
    """Render useful distillation state without printing every training batch."""

    def __init__(
        self,
        *,
        interval_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        stream: TextIO | None = None,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("distillation progress interval cannot be negative")
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._stream = stream
        self._teacher_epoch_started = 0.0
        self._training_started = 0.0
        self._last_teacher_update = float("-inf")
        self._last_training_update = float("-inf")
        self._starting_step = 0

    def _write(self, message: str) -> None:
        print(f"[distillation] {message}", file=self._stream or sys.stdout, flush=True)

    def status(self, message: str) -> None:
        """Emit a phase-boundary message immediately."""

        self._write(message)

    def __call__(self, event: str, fields: Mapping[str, object]) -> None:
        now = self._clock()
        if event == "teacher_cache.epoch_started":
            self._teacher_epoch_started = now
            self._last_teacher_update = float("-inf")
            epoch = _integer(fields, "epoch") + 1
            epochs = _integer(fields, "epochs")
            batches = _integer(fields, "total_batches")
            self._write(f"teacher cache epoch {epoch}/{epochs} started ({batches} batches)")
            return

        if event == "teacher_cache.batch_completed":
            completed = _integer(fields, "completed_batches")
            total = _integer(fields, "total_batches")
            if completed != 1 and now - self._last_teacher_update < self._interval_seconds:
                return
            self._last_teacher_update = now
            elapsed = max(0.0, now - self._teacher_epoch_started)
            eta = elapsed / completed * (total - completed) if completed else None
            epoch = _integer(fields, "epoch") + 1
            epochs = _integer(fields, "epochs")
            tokens = _integer(fields, "selected_tokens")
            cache_mib = _integer(fields, "cache_bytes") / _MIB
            self._write(
                f"teacher cache epoch {epoch}/{epochs}: batch {completed}/{total}, "
                f"tokens={tokens}, cache={cache_mib:.1f} MiB, "
                f"elapsed={_duration(elapsed)}, eta={_duration(eta)}"
            )
            return

        if event == "teacher_cache.epoch_completed":
            elapsed = max(0.0, now - self._teacher_epoch_started)
            epoch = _integer(fields, "epoch") + 1
            epochs = _integer(fields, "epochs")
            batches = _integer(fields, "completed_batches")
            cache_mib = _integer(fields, "cache_bytes") / _MIB
            self._write(
                f"teacher cache epoch {epoch}/{epochs} complete: "
                f"{batches} batches, cache={cache_mib:.1f} MiB, elapsed={_duration(elapsed)}"
            )
            return

        if event == "training.started":
            self._training_started = now
            self._last_training_update = float("-inf")
            self._starting_step = _integer(fields, "completed_steps")
            total_steps = _integer(fields, "total_steps")
            epochs = _integer(fields, "epochs")
            parameters = _integer(fields, "selected_parameters")
            self._write(
                f"training started at step {self._starting_step}/{total_steps} "
                f"for {epochs} epochs ({parameters} trainable tensors)"
            )
            return

        if event == "training.epoch_started":
            epoch = _integer(fields, "epoch") + 1
            epochs = _integer(fields, "epochs")
            batches = _integer(fields, "total_batches")
            self._write(f"training epoch {epoch}/{epochs} started ({batches} batches)")
            return

        if event == "training.batch_completed":
            completed = _integer(fields, "completed_batches")
            if completed != 1 and now - self._last_training_update < self._interval_seconds:
                return
            self._last_training_update = now
            step = _integer(fields, "completed_steps")
            total_steps = _integer(fields, "total_steps")
            local_steps = step - self._starting_step
            elapsed = max(0.0, now - self._training_started)
            eta = elapsed / local_steps * (total_steps - step) if local_steps > 0 else None
            epoch = _integer(fields, "epoch") + 1
            epochs = _integer(fields, "epochs")
            total_batches = _integer(fields, "total_batches")
            batch_loss = _float(fields, "batch_loss")
            mean_loss = _float(fields, "epoch_mean_loss")
            learning_rate = _float(fields, "learning_rate")
            self._write(
                f"training epoch {epoch}/{epochs}: batch {completed}/{total_batches}, "
                f"step {step}/{total_steps}, loss={batch_loss:.6f}, "
                f"epoch_mean={mean_loss:.6f}, lr={learning_rate:.3e}, "
                f"elapsed={_duration(elapsed)}, eta={_duration(eta)}"
            )
            return

        if event == "training.epoch_completed":
            epoch = _integer(fields, "epoch") + 1
            epochs = _integer(fields, "epochs")
            step = _integer(fields, "completed_steps")
            total_steps = _integer(fields, "total_steps")
            mean_loss = _float(fields, "epoch_mean_loss")
            self._write(
                f"training epoch {epoch}/{epochs} complete: step {step}/{total_steps}, "
                f"mean_loss={mean_loss:.6f}"
            )
            return

        if event == "training.completed":
            elapsed = max(0.0, now - self._training_started)
            steps = _integer(fields, "completed_steps")
            final_loss = fields.get("final_epoch_loss")
            loss_suffix = (
                ""
                if not isinstance(final_loss, (int, float)) or isinstance(final_loss, bool)
                else f", final_epoch_loss={float(final_loss):.6f}"
            )
            self._write(f"training complete: {steps} steps{loss_suffix}, elapsed={_duration(elapsed)}")
