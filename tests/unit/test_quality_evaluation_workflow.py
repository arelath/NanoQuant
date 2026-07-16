from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import nanoquant.quality_evaluation_workflow as workflow
from nanoquant.quality_evaluation import QualityEvaluationRequest
from nanoquant.quality_evaluation_workflow import (
    QualityEvaluationExperiment,
    execute_quality_evaluation_experiment,
    resolve_quality_evaluation_experiment,
)
from nanoquant.recipes import (
    EXPERIMENT_002_CONFIG,
    EXPERIMENT_002_EVALUATION,
    EXPERIMENT_003_CONFIG,
    EXPERIMENT_003_EVALUATION,
)


def test_quality_evaluation_request_rejects_ambiguous_protocols(tmp_path: Path) -> None:
    request = QualityEvaluationRequest(tmp_path, "model", "revision", tmp_path / "run")

    with pytest.raises(ValueError, match="WikiText"):
        replace(request, wikitext_samples=0)
    with pytest.raises(ValueError, match="task names"):
        replace(request, task_names=("piqa", "piqa"))
    with pytest.raises(ValueError, match="unsupported"):
        replace(request, task_names=("gsm8k",))


def test_experiment003_preserves_legacy_quality_smoke_protocol() -> None:
    request = EXPERIMENT_003_EVALUATION.request

    assert request.wikitext_samples == 16
    assert request.wikitext_sequence_length == 128
    assert request.wikitext_batch_size == 1
    assert request.task_names == ("piqa", "arc_easy", "boolq")
    assert request.task_limit == 25
    assert request.task_batch_size == 1
    assert request.backend == "factorized"
    assert request.use_global_tuning


def test_experiment002_uses_the_full_common_quality_protocol() -> None:
    request = EXPERIMENT_002_EVALUATION.request

    assert request.wikitext_samples == 64
    assert request.wikitext_sequence_length == 128
    assert request.wikitext_batch_size == 1
    assert request.task_names == (
        "piqa",
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "boolq",
    )
    assert request.task_limit == 200
    assert request.task_batch_size == 1
    assert request.backend == "factorized"
    assert request.use_global_tuning


def test_002_benchmark_runfile_imports_canonical_recipe_objects() -> None:
    namespace = runpy.run_path("experiments/002-benchmark-gemma-3-1b-it.py")

    assert namespace["CONFIG"] is EXPERIMENT_002_CONFIG
    assert namespace["EVALUATION"] is EXPERIMENT_002_EVALUATION


def test_quality_experiment_resolution_is_pinned_and_repository_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "repo" / "experiments" / "003-example.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(
        workflow,
        "snapshot_download",
        lambda *, repo_id, revision: str(snapshot),
    )

    resolved = resolve_quality_evaluation_experiment(
        EXPERIMENT_003_CONFIG,
        EXPERIMENT_003_EVALUATION,
        launcher_path=launcher,
    )

    assert resolved.request.snapshot == snapshot.resolve()
    assert resolved.request.run_output == (
        tmp_path / "repo" / "evidence/m4/gemma-pageable-v28-four-block-canary"
    )
    assert resolved.result_path == tmp_path / "repo" / "evidence/m9/003-gemma-3-1b-it-quality.json"


def test_quality_workflow_records_config_and_launcher_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    request = QualityEvaluationRequest(tmp_path, "model", "revision", tmp_path / "run")
    experiment = QualityEvaluationExperiment(request, output)
    observed: list[QualityEvaluationRequest] = []
    launcher = tmp_path / "003-evaluate-gemma-3-1b-it-quality.py"
    launcher.write_text("# provenance fixture\n", encoding="utf-8")

    def evaluate(resolved: QualityEvaluationRequest) -> dict[str, Any]:
        observed.append(resolved)
        return {"schema_version": 1, "passed": True, "results": {}}

    monkeypatch.setattr(workflow, "execute_quality_evaluation", evaluate)
    payload = execute_quality_evaluation_experiment(
        EXPERIMENT_003_CONFIG,
        experiment,
        launcher_path=launcher,
    )

    assert observed == [replace(request, source="google/gemma-3-1b-it", revision=EXPERIMENT_003_CONFIG.model.revision)]
    assert payload["experiment"]["launcher"]["experiment_number"] == 3
    assert payload["experiment"]["resolved_config"]["intent"]["name"] == (
        "003-evaluate-gemma-3-1b-it-quality"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload
