"""Reusable explicit VRAM-budget assertion for opt-in CUDA tests."""

from __future__ import annotations

import json
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType

import pytest
import torch

from nanoquant.infrastructure.device_memory import CudaMemoryHistory, PeakWindow, sample_device_memory

_REPORT_LOCK = threading.Lock()


class VramBudgetWindow(AbstractContextManager["VramBudgetWindow"]):
    def __init__(self, node_id: str, budget: int, report: Path | None, device: str) -> None:
        self.node_id = node_id
        self.budget = budget
        self.report = report
        self.device = device
        self.peak_increment_bytes = 0
        self.peak_allocated_increment_bytes = 0
        self._baseline_allocated = 0
        self._baseline_reserved = 0
        self._window = PeakWindow(device)
        history_root = report.parent / "vram-history" if report is not None else Path(".pytest-vram")
        self._history = CudaMemoryHistory(history_root, False)

    def __enter__(self) -> VramBudgetWindow:
        if not torch.cuda.is_available():
            pytest.skip("VRAM budget requires CUDA")
        torch.cuda.empty_cache()
        baseline = sample_device_memory()
        self._baseline_allocated = baseline.get("cuda.allocated_bytes", 0)
        self._baseline_reserved = baseline.get("cuda.reserved_bytes", 0)
        self._history.start()
        self._window.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._window.finish()
        self.peak_allocated_increment_bytes = max(
            0,
            self._window.peak_allocated_bytes - self._baseline_allocated,
        )
        self.peak_increment_bytes = max(
            self.peak_allocated_increment_bytes,
            self._window.peak_reserved_bytes - self._baseline_reserved,
            0,
        )
        exceeded = self.peak_increment_bytes > self.budget
        history_snapshot = self._history.dump("budget-failure") if exceeded else None
        self._history.stop()
        payload = {
            "test_id": self.node_id,
            "budget_bytes": self.budget,
            "peak_increment_bytes": self.peak_increment_bytes,
            "peak_allocated_increment_bytes": self.peak_allocated_increment_bytes,
            "cuda.window_peak_allocated_bytes": self._window.peak_allocated_bytes,
            "cuda.window_peak_reserved_bytes": self._window.peak_reserved_bytes,
            "device_name": torch.cuda.get_device_name(self.device),
            "torch_version": torch.__version__,
            "vram_history_path": None if history_snapshot is None else str(history_snapshot),
        }
        if self.report is not None:
            self.report.parent.mkdir(parents=True, exist_ok=True)
            with _REPORT_LOCK, self.report.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        if exc_type is None and exceeded:
            pytest.fail(
                "VRAM budget exceeded\n"
                f"  budget_bytes:                  {self.budget}\n"
                f"  peak_increment_bytes:          {self.peak_increment_bytes}\n"
                f"  peak_allocated_increment_bytes:{self.peak_allocated_increment_bytes}\n"
                f"  window_peak_allocated_bytes:   {self._window.peak_allocated_bytes}\n"
                f"  window_peak_reserved_bytes:    {self._window.peak_reserved_bytes}",
                pytrace=False,
            )
        return None


class VramBudgetFactory:
    def __init__(self, node_id: str, report: Path | None) -> None:
        self.node_id = node_id
        self.report = report

    def __call__(self, *, peak_increment_bytes: int, device: str = "cuda:0") -> VramBudgetWindow:
        if peak_increment_bytes < 0:
            raise ValueError("VRAM budget must not be negative")
        return VramBudgetWindow(self.node_id, peak_increment_bytes, self.report, device)
