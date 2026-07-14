from __future__ import annotations

import json
from pathlib import Path

import pytest

import tests.support.vram_budget as budget_module
from tests.support.vram_budget import VramBudgetWindow


class FakePeakWindow:
    peak_allocated_bytes = 450
    peak_reserved_bytes = 600

    def start(self) -> FakePeakWindow:
        return self

    def finish(self) -> None:
        return None


def test_vram_budget_writes_standard_jsonl_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(budget_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(budget_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(budget_module.torch.cuda, "get_device_name", lambda _device: "fixture-gpu")
    monkeypatch.setattr(
        budget_module,
        "sample_device_memory",
        lambda: {"cuda.allocated_bytes": 100, "cuda.reserved_bytes": 200},
    )
    report = tmp_path / "vram.jsonl"
    window = VramBudgetWindow("test_fixture", 500, report, "cuda:0")
    window._window = FakePeakWindow()  # type: ignore[assignment]

    with window:
        pass

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["test_id"] == "test_fixture"
    assert payload["peak_increment_bytes"] == 400
    assert payload["cuda.window_peak_allocated_bytes"] == 450
    assert payload["cuda.window_peak_reserved_bytes"] == 600
