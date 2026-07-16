from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from recipes import EXPERIMENT_011_BENCHMARK, EXPERIMENT_011_CONFIG

import nanoquant.benchmark_workflow as workflow
from nanoquant.benchmark_workflow import (
    RuntimeBenchmarkExperiment,
    execute_runtime_benchmark_experiment,
    resolve_runtime_benchmark_experiment,
)
from nanoquant.runtime_benchmark import RuntimeBenchmarkRequest


def test_runtime_benchmark_request_rejects_invalid_protocols(tmp_path: Path) -> None:
    base = RuntimeBenchmarkRequest(tmp_path / "packed", tmp_path / "model")

    with pytest.raises(ValueError, match="warmups"):
        replace(base, warmups=-1)
    with pytest.raises(ValueError, match="suite"):
        replace(base, suite=("unknown",))
    with pytest.raises(ValueError, match="prompts"):
        replace(base, prompt=())


def test_experiment011_preserves_legacy_generation_workload_on_packed_runtime() -> None:
    request = EXPERIMENT_011_BENCHMARK.request

    assert request.suite == ("end-to-end",)
    assert request.input_dtype == "bfloat16"
    assert request.cache_dtype == "bfloat16"
    assert request.warmups == 1
    assert request.repetitions == 3
    assert request.max_new_tokens == 128
    assert not request.chat_template
    assert request.ignore_eos
    assert request.prompt == (
        "Explain why compact language models are useful for local inference.",
    )
    assert request.packed_artifact == Path(
        "evidence/m6/gemma-pageable-v28-runtime-bundle/packed"
    )
    assert request.run_output == Path("evidence/m4/gemma-pageable-v28-four-block-canary")


def test_benchmark_experiment_resolution_is_repository_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "repo" / "experiments" / "011-example.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(
        workflow,
        "snapshot_download",
        lambda *, repo_id, revision: str(snapshot),
    )

    resolved = resolve_runtime_benchmark_experiment(
        EXPERIMENT_011_CONFIG,
        EXPERIMENT_011_BENCHMARK,
        launcher_path=launcher,
    )

    assert resolved.request.packed_artifact == (
        tmp_path / "repo" / "evidence/m6/gemma-pageable-v28-runtime-bundle/packed"
    )
    assert resolved.request.model == snapshot.resolve()
    assert resolved.result_path == tmp_path / "repo" / "evidence/m9/011-generation-tps.json"


def test_benchmark_workflow_records_config_and_launcher_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    request = RuntimeBenchmarkRequest(
        tmp_path / "packed",
        tmp_path / "model",
        suite=("end-to-end",),
    )
    experiment = RuntimeBenchmarkExperiment(request, output)
    observed: list[RuntimeBenchmarkRequest] = []
    launcher = tmp_path / "011-benchmark-generation-tps.py"
    launcher.write_text("# provenance fixture\n", encoding="utf-8")

    def benchmark(resolved: RuntimeBenchmarkRequest) -> dict[str, Any]:
        observed.append(resolved)
        return {"schema_version": 1, "passed": True, "cases": []}

    monkeypatch.setattr(workflow, "run_runtime_benchmark", benchmark)
    payload = execute_runtime_benchmark_experiment(
        EXPERIMENT_011_CONFIG,
        experiment,
        launcher_path=launcher,
    )

    assert observed == [request]
    assert payload["experiment"]["launcher"]["experiment_number"] == 11
    assert payload["experiment"]["launcher"]["repository_relative_path"] is None
    assert payload["experiment"]["resolved_config"]["intent"]["name"] == (
        "011-benchmark-generation-tps"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload
