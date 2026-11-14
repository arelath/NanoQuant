from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from nanoquant.quality_evaluation import QualityEvaluationRequest

REPOSITORY = Path(__file__).resolve().parents[2]
LAUNCHER = next((REPOSITORY / "experiments").glob("074-*.py"))


def _tool() -> ModuleType:
    tools = str(REPOSITORY / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("export_pre_global_tuning")


def test_plan_exports_the_uniform_control_to_new_pre_kd_paths() -> None:
    tool = _tool()
    plan = tool.build_plan(LAUNCHER, uniform_control=True)

    assert plan.experiment_number == 74
    assert plan.run_output == (REPOSITORY / "evidence/074/074-d2-uniform-control-gemma-3-270m-it").resolve()
    assert plan.config.distillation.enabled
    export = plan.export
    assert export.use_active_global_tuning is False
    assert export.publication_experiment_number == 74
    assert export.huggingface is None
    assert export.logical_output == Path("outputs/074/pre-kd/logical")
    assert export.packed_output == Path("outputs/074/pre-kd/packed")
    assert export.checkpoint_output == Path("outputs/074/pre-kd/llamacpp-checkpoint")
    assert export.gguf_output == Path("Results/074/gemma-3-270m-it-nanoquant-pre-kd.gguf")
    assert plan.quality_output.name.endswith("-it-pre-kd-quality.json")
    assert plan.quality_markdown_output.parent == (REPOSITORY / "Results/074").resolve()
    assert plan.quality_markdown_output.name.endswith("-it-pre-kd-quality.md")
    assert plan.published_quality.name.endswith("-it-quality.json")
    tool._require_separate_outputs(plan, REPOSITORY)


def test_plan_refuses_outputs_that_would_replace_the_published_export() -> None:
    tool = _tool()
    plan = tool.build_plan(LAUNCHER, uniform_control=True)
    reused = replace(plan, export=replace(plan.export, packed_output=plan.workflow.export.packed_output))

    with pytest.raises(ValueError, match="replace published outputs"):
        tool._require_separate_outputs(reused, REPOSITORY)
    with pytest.raises(ValueError, match="plain name"):
        tool.build_plan(LAUNCHER, uniform_control=True, suffix="../x")


def test_plan_only_prints_paths_without_writing(capsys) -> None:
    tool = _tool()
    args = argparse.Namespace(launcher=LAUNCHER, uniform_control=True, suffix="pre-kd", plan_only=True)

    assert tool.run(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["export"]["use_active_global_tuning"] is False
    assert report["export"]["gguf_output"].endswith("gemma-3-270m-it-nanoquant-pre-kd.gguf")


def _request(**overrides: object) -> QualityEvaluationRequest:
    fields: dict[str, object] = {
        "snapshot": Path("snapshot"),
        "source": "unsloth/gemma-3-270m-it",
        "revision": "r",
        "run_output": Path("run"),
        "wikitext_samples": 64,
        "wikitext_sequence_length": 128,
        "wikitext_batch_size": 8,
        "task_names": ("piqa", "boolq"),
        "task_limit": 1000,
        "task_batch_size": 4,
    }
    fields.update(overrides)
    return QualityEvaluationRequest(**fields)  # type: ignore[arg-type]


def test_base_result_is_reused_only_for_a_matching_protocol(tmp_path: Path) -> None:
    tool = _tool()
    published = tmp_path / "quality.json"
    published.write_text(json.dumps({
        "model": {"source": "unsloth/gemma-3-270m-it", "revision": "r"},
        "protocol": {
            "wikitext_samples": 64,
            "wikitext_sequence_length": 128,
            "wikitext_batch_size": 8,
            "task_names": ["piqa", "boolq"],
            "task_limit": 1000,
            "task_batch_size": 4,
        },
        "results": {"base": {"wikitext": {}, "tasks": []}},
    }), encoding="utf-8")

    assert tool.reusable_base_result(published, _request()) == {"wikitext": {}, "tasks": []}
    assert tool.reusable_base_result(published, _request(task_limit=200)) is None
    assert tool.reusable_base_result(published, _request(task_names=("piqa",))) is None
    assert tool.reusable_base_result(published, _request(revision="other")) is None
    assert tool.reusable_base_result(tmp_path / "missing.json", _request()) is None
