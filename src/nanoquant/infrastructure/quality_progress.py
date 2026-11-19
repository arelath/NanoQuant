"""Rate-limited console progress for PyTorch base/frozen quality evaluation."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from typing import TextIO

_GIB = 1024**3
_BATCH_EVENTS = frozenset(
    {"wikitext_batch_completed", "task_batch_completed", "reasoning_batch_completed"}
)


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


def _number(fields: Mapping[str, object], name: str) -> float | None:
    value = fields.get(name)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _seconds(fields: Mapping[str, object]) -> str:
    return _duration(_number(fields, "elapsed_seconds"))


def _gib(fields: Mapping[str, object], name: str) -> str:
    value = _number(fields, name)
    return "?" if value is None else f"{value / _GIB:.2f} GiB"


def _scope(event: str, fields: Mapping[str, object]) -> tuple[str, str]:
    model = str(fields.get("model", "?"))
    if event.startswith("wikitext"):
        return model, "wikitext"
    if event.startswith("task"):
        return model, f"task:{fields.get('task', '?')}"
    return model, f"reasoning:{fields.get('mode', '?')}"


class QualityProgressLogger:
    """Print quality-evaluation phases immediately and batch progress periodically."""

    def __init__(
        self,
        *,
        interval_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        stream: TextIO | None = None,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("quality progress interval cannot be negative")
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._stream = stream
        self._phase_started: dict[tuple[str, str], float] = {}
        self._last_batch_update: dict[tuple[str, str], float] = {}

    def _write(self, message: str) -> None:
        print(f"[quality] {message}", file=self._stream or sys.stdout, flush=True)

    def _start_phase(self, event: str, fields: Mapping[str, object]) -> None:
        scope = _scope(event, fields)
        self._phase_started[scope] = self._clock()
        self._last_batch_update[scope] = float("-inf")

    def _batch(self, event: str, fields: Mapping[str, object]) -> None:
        scope = _scope(event, fields)
        now = self._clock()
        completed = _integer(fields, "completed_batches")
        total = _integer(fields, "total_batches")
        last = self._last_batch_update.get(scope, float("-inf"))
        if completed not in {1, total} and now - last < self._interval_seconds:
            return
        self._last_batch_update[scope] = now
        started = self._phase_started.setdefault(scope, now)
        elapsed = max(0.0, now - started)
        eta = elapsed / completed * (total - completed) if completed else None
        model, phase = scope
        if event == "task_batch_completed":
            phase = (
                f"{fields.get('task', '?')} "
                f"({_integer(fields, 'task_index')}/{_integer(fields, 'task_count')})"
            )
        elif event == "reasoning_batch_completed":
            phase = f"reasoning {fields.get('mode', '?')}"
        percent = 100.0 * completed / total if total else 0.0
        self._write(
            f"{model} {phase}: batch {completed}/{total} ({percent:.0f}%), "
            f"elapsed={_duration(elapsed)}, eta={_duration(eta)}"
        )

    def __call__(self, event: str, fields: Mapping[str, object]) -> None:
        if event in _BATCH_EVENTS:
            self._batch(event, fields)
            return
        model = fields.get("model", "?")
        message: str | None
        if event == "input_preparation_started":
            message = (
                "reusing prepared evaluation inputs"
                if fields.get("reused")
                else "preparing evaluation inputs"
            )
        elif event == "input_preparation_completed":
            message = f"evaluation inputs ready ({_integer(fields, 'task_count')} tasks)"
        elif event == "wikitext_input_dataset_started":
            message = f"loading WikiText dataset {fields.get('dataset', '?')}"
        elif event == "wikitext_input_dataset_completed":
            message = f"WikiText dataset loaded ({_integer(fields, 'rows')} rows)"
        elif event == "wikitext_input_tokenization_started":
            message = "tokenizing WikiText"
        elif event == "wikitext_input_tokenization_completed":
            message = f"WikiText tokenized ({_integer(fields, 'tokens')} tokens)"
        elif event == "tokenizer_load_started":
            message = "loading tokenizer"
        elif event == "tokenizer_hash_started":
            message = "hashing tokenizer snapshot"
        elif event == "task_input_started":
            message = (
                f"preparing task {fields.get('task', '?')} "
                f"({_integer(fields, 'task_index')}/{_integer(fields, 'task_count')})"
            )
        elif event == "task_input_completed":
            message = f"task {fields.get('task', '?')} prepared ({_integer(fields, 'examples')} examples)"
        elif event == "reasoning_input_started":
            message = f"preparing reasoning inputs ({fields.get('mode', '?')})"
        elif event == "reasoning_input_completed":
            message = f"reasoning inputs ready ({fields.get('mode', '?')})"
        elif event == "device_lease_started":
            message = f"waiting for device lease on {fields.get('device', '?')}"
        elif event == "device_lease_acquired":
            message = f"device lease acquired on {fields.get('device', '?')}"
        elif event == "model_load_started":
            message = f"loading {model} model"
        elif event == "model_load_completed":
            message = f"{model} model loaded in {_seconds(fields)}"
        elif event == "model_evaluation_started":
            message = f"evaluating {model} model"
        elif event == "model_evaluation_reused":
            message = f"reusing {model} model evaluation"
        elif event == "wikitext_started":
            self._start_phase(event, fields)
            message = f"{model} WikiText perplexity started"
        elif event == "wikitext_completed":
            perplexity = _number(fields, "perplexity")
            message = (
                f"{model} WikiText perplexity="
                f"{'?' if perplexity is None else f'{perplexity:.4f}'} "
                f"({_integer(fields, 'token_count')} tokens) in {_seconds(fields)}"
            )
        elif event == "task_started":
            self._start_phase(event, fields)
            message = (
                f"{model} task {fields.get('task', '?')} "
                f"({_integer(fields, 'task_index')}/{_integer(fields, 'task_count')}) started, "
                f"{_integer(fields, 'sample_count')} samples"
            )
        elif event == "task_completed":
            value = _number(fields, "value")
            message = (
                f"{model} task {fields.get('task', '?')} "
                f"{fields.get('metric', '?')}={'?' if value is None else f'{value:.4f}'} "
                f"in {_seconds(fields)}"
            )
        elif event == "reasoning_started":
            self._start_phase(event, fields)
            message = (
                f"{model} reasoning {fields.get('mode', '?')} started, "
                f"{_integer(fields, 'sample_count')} samples"
            )
        elif event == "reasoning_completed":
            perplexity = _number(fields, "perplexity")
            message = (
                f"{model} reasoning {fields.get('mode', '?')} perplexity="
                f"{'?' if perplexity is None else f'{perplexity:.4f}'} "
                f"({_integer(fields, 'token_count')} tokens) in {_seconds(fields)}"
            )
        elif event == "model_evaluation_completed":
            message = (
                f"{model} model evaluated in {_seconds(fields)}, "
                f"peak device={_gib(fields, 'peak_device_bytes')}, "
                f"peak host={_gib(fields, 'peak_host_bytes')}"
            )
        elif event == "quality_evaluation_completed":
            message = f"quality evaluation complete in {_duration(_number(fields, 'wall_seconds'))}"
        elif event in {"tokenizer_load_completed", "tokenizer_hash_completed"}:
            message = None
        else:
            details = ", ".join(f"{key}={value}" for key, value in fields.items())
            message = event if not details else f"{event}: {details}"
        if message is not None:
            self._write(message)


__all__ = ["QualityProgressLogger"]
