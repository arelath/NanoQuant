from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from recipes import (
    EXPERIMENT_002_CONFIG,
    EXPERIMENT_002_EVALUATION,
)

import nanoquant.quality_evaluation_workflow as workflow
from nanoquant.quality_evaluation import QualityEvaluationRequest
from nanoquant.quality_evaluation_workflow import (
    QualityEvaluationExperiment,
    execute_quality_evaluation_experiment,
    render_quality_evaluation_markdown,
    resolve_quality_evaluation_experiment,
)


def test_quality_evaluation_request_rejects_ambiguous_protocols(tmp_path: Path) -> None:
    request = QualityEvaluationRequest(tmp_path, "model", "revision", tmp_path / "run")

    with pytest.raises(ValueError, match="WikiText"):
        replace(request, wikitext_samples=0)
    with pytest.raises(ValueError, match="task names"):
        replace(request, task_names=("piqa", "piqa"))
    with pytest.raises(ValueError, match="unsupported"):
        replace(request, task_names=("gsm8k",))


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
    assert EXPERIMENT_002_EVALUATION.markdown_path == Path(
        "evidence/m9/002-gemma-3-1b-it-quality-benchmark.md"
    )


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

    experiment = replace(
        EXPERIMENT_002_EVALUATION,
        request=replace(
            EXPERIMENT_002_EVALUATION.request,
            packed_artifact=Path("outputs/candidate/packed"),
        ),
    )
    resolved = resolve_quality_evaluation_experiment(
        EXPERIMENT_002_CONFIG,
        experiment,
        launcher_path=launcher,
    )
    assert resolved.request.snapshot == snapshot.resolve()
    assert resolved.markdown_path == (
        tmp_path / "repo" / "evidence/m9/002-gemma-3-1b-it-quality-benchmark.md"
    )
    assert resolved.request.packed_artifact == tmp_path / "repo" / "outputs/candidate/packed"


def test_quality_workflow_records_config_and_launcher_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    request = QualityEvaluationRequest(tmp_path, "model", "revision", tmp_path / "run")
    experiment = QualityEvaluationExperiment(request, output)
    observed: list[QualityEvaluationRequest] = []
    launcher = tmp_path / "002-evaluate-gemma-3-1b-it-quality.py"
    launcher.write_text("# provenance fixture\n", encoding="utf-8")

    def evaluate(resolved: QualityEvaluationRequest) -> dict[str, Any]:
        observed.append(resolved)
        return {"schema_version": 1, "passed": True, "results": {}}

    monkeypatch.setattr(workflow, "execute_quality_evaluation", evaluate)
    published = []
    monkeypatch.setattr(
        workflow,
        "publish_experiment_artifacts",
        lambda root, number, artifacts: published.append((root, number, tuple(artifacts))),
    )
    payload = execute_quality_evaluation_experiment(
        EXPERIMENT_002_CONFIG,
        experiment,
        launcher_path=launcher,
    )

    assert observed == [replace(request, source="google/gemma-3-1b-it", revision=EXPERIMENT_002_CONFIG.model.revision)]
    assert payload["experiment"]["launcher"]["experiment_number"] == 2
    assert payload["experiment"]["resolved_config"]["intent"]["name"] == (
        "002-benchmark-gemma-3-1b-it"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert published[0][1] == 2
    assert [artifact.source for artifact in published[0][2]] == [output]


def _quality_result() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "passed": True,
        "model": {"source": "fixture/model", "revision": "revision", "snapshot": "snapshot"},
        "candidate": {
            "run_output": "run",
            "commit_identity": {"config_hash": "config", "model_hash": "model", "plan_hash": "plan"},
            "global_tuning": None,
            "backend": "factorized",
        },
        "protocol": {
            "wikitext_samples": 64,
            "wikitext_sequence_length": 128,
            "wikitext_batch_size": 1,
            "wikitext_token_hash": "sha256:tokens",
            "task_names": ("piqa",),
            "task_limit": 200,
            "task_batch_size": 1,
            "tokenizer_hash": "sha256:tokenizer",
        },
        "results": {
            "base": {
                "elapsed_seconds": 2.0,
                "peak_device_bytes": 100,
                "peak_host_bytes": 200,
            },
            "frozen": {
                "elapsed_seconds": 3.0,
                "peak_device_bytes": 110,
                "peak_host_bytes": 220,
            },
        },
        "comparison": {
            "wikitext": {
                "base_perplexity": 10.0,
                "frozen_perplexity": 12.5,
                "ratio": 1.25,
                "relative_change": 0.25,
            },
            "tasks": [
                {
                    "task_name": "piqa",
                    "metric": "acc_norm",
                    "base": 0.75,
                    "frozen": 0.70,
                    "delta": -0.05,
                    "ratio": 0.7 / 0.75,
                }
            ],
        },
        "wall_seconds": 5.5,
    }


def test_quality_workflow_writes_deterministic_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    markdown = tmp_path / "result.md"
    launcher = tmp_path / "002-quality.py"
    launcher.write_text("# provenance fixture\n", encoding="utf-8")
    request = QualityEvaluationRequest(tmp_path, "model", "revision", tmp_path / "run")
    experiment = QualityEvaluationExperiment(request, output, markdown_path=markdown)
    monkeypatch.setattr(workflow, "execute_quality_evaluation", lambda _request: _quality_result())
    published = []
    monkeypatch.setattr(
        workflow,
        "publish_experiment_artifacts",
        lambda root, number, artifacts: published.append((root, number, tuple(artifacts))),
    )

    payload = execute_quality_evaluation_experiment(
        EXPERIMENT_002_CONFIG,
        experiment,
        launcher_path=launcher,
    )

    rendered = markdown.read_text(encoding="utf-8")
    assert rendered == render_quality_evaluation_markdown(payload)
    assert "| WikiText-2 | perplexity ↓ | 10.000000 | 12.500000 | +2.500000 (+25.00%) | 1.2500x |" in rendered
    assert "| piqa | acc_norm ↑ | 0.7500 | 0.7000 | -0.0500 | 0.9333x |" in rendered
    assert "`completed` means all evaluators returned finite metrics" in rendered
    assert [artifact.source for artifact in published[0][2]] == [output, markdown]
