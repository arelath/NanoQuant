from pathlib import Path

import pytest
import torch

from nanoquant.application.calibration_fallback import run_with_oom_fallback
from nanoquant.infrastructure.events import JsonlEventSink


def test_oom_fallback_is_finite_visible_and_distinguishes_algorithm_change(tmp_path: Path) -> None:
    calls = []

    def operation(method: str, device: str) -> str:
        calls.append((method, device))
        if len(calls) < 3:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return "ok"

    sink = JsonlEventSink(tmp_path / "events.jsonl", "run")
    value, revisions = run_with_oom_fallback(
        operation, "online_fisher", "cuda:0", ("cpu_offload", "forward_only", "fail"), sink
    )
    assert value == "ok"
    assert calls == [("online_fisher", "cuda:0"), ("online_fisher", "cpu"), ("forward_only", "cpu")]
    assert [revision.algorithm_changed for revision in revisions] == [False, True]
    assert len((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_non_oom_is_not_swallowed_and_fail_terminates(tmp_path: Path) -> None:
    sink = JsonlEventSink(tmp_path / "events.jsonl", "run")
    with pytest.raises(KeyError):
        run_with_oom_fallback(
            lambda _method, _device: (_ for _ in ()).throw(KeyError("bad")), "online_fisher", "cuda:0", ("fail",), sink
        )
    with pytest.raises(torch.OutOfMemoryError):
        run_with_oom_fallback(
            lambda _method, _device: (_ for _ in ()).throw(torch.OutOfMemoryError()),
            "online_fisher",
            "cuda:0",
            ("fail",),
            sink,
        )


def test_unknown_fallback_is_rejected(tmp_path: Path) -> None:
    sink = JsonlEventSink(tmp_path / "events.jsonl", "run")
    with pytest.raises(ValueError, match="CAL006"):
        run_with_oom_fallback(
            lambda _method, _device: (_ for _ in ()).throw(torch.OutOfMemoryError()),
            "online_fisher",
            "cuda:0",
            ("magic",),
            sink,
        )
