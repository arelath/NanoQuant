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
        product_codebook_overlay=tmp_path / "product-overlay",
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
        base_quality=None,
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
    assert request.product_codebook_overlay == args.product_codebook_overlay
    assert json.loads(output.read_text(encoding="utf-8")) == {"passed": True}


def test_packed_quality_runner_reuses_only_a_protocol_matched_base(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _tool_module()
    captured = {}
    base_result = {
        "wikitext": {"perplexity": 12.0},
        "tasks": {name: {"primary_metric_value": 0.5} for name in tool.DEFAULT_TASKS},
    }
    base_quality = tmp_path / "base-quality.json"
    base_quality.write_text(
        json.dumps(
            {
                "model": {"source": "fixture/model", "revision": "revision"},
                "protocol": {
                    "wikitext_samples": 64,
                    "wikitext_sequence_length": 128,
                    "wikitext_batch_size": 8,
                    "task_names": list(tool.DEFAULT_TASKS),
                    "task_limit": 200,
                    "task_batch_size": 4,
                },
                "results": {"base": base_result},
            }
        ),
        encoding="utf-8",
    )

    def execute(request, *, progress, base_result):  # type: ignore[no-untyped-def]
        captured["base_result"] = base_result
        return {"passed": True}

    monkeypatch.setattr(tool, "execute_quality_evaluation", execute)
    output = tmp_path / "quality.json"
    args = argparse.Namespace(
        packed_artifact=tmp_path / "packed",
        component_overlay=None,
        product_codebook_overlay=None,
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
        base_quality=base_quality,
    )

    assert tool.run(args) == 0
    assert captured["base_result"] == base_result
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["base_result_source"] == str(base_quality.resolve())
