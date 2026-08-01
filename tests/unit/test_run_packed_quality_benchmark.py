from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def _tool_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("run_packed_quality_benchmark")


def test_packed_quality_runner_preserves_complete_default_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _tool_module()
    captured = {}

    def execute(request, *, progress):  # type: ignore[no-untyped-def]
        captured["request"] = request
        progress("started", {"fixture": True})
        return {"passed": True}

    monkeypatch.setattr(tool, "execute_quality_evaluation", execute)
    output = tmp_path / "quality.json"
    args = argparse.Namespace(
        packed_artifact=tmp_path / "packed",
        component_overlay=None,
        run_output=tmp_path / "run",
        snapshot=tmp_path / "snapshot",
        source="fixture/model",
        revision="revision",
        output=output,
        device="cuda:0",
        backend="factorized",
        wikitext_samples=64,
        wikitext_sequence_length=128,
        wikitext_batch_size=8,
        task=[],
        task_limit=200,
        task_batch_size=4,
        maximum_wddm_shared_bytes=805_306_368,
        local_files_only=True,
        no_global_tuning=False,
    )

    assert tool.run(args) == 0

    request = captured["request"]
    assert request.wikitext_samples == 64
    assert request.wikitext_batch_size == 8
    assert request.task_names == tool.DEFAULT_TASKS
    assert request.task_limit == 200
    assert request.task_batch_size == 4
    assert request.packed_artifact == args.packed_artifact
    assert request.component_overlay is None
    assert json.loads(output.read_text(encoding="utf-8")) == {"passed": True}
