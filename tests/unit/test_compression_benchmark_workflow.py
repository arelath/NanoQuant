from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from recipes import EXPERIMENT_001, EXPERIMENT_001_CONFIG

import nanoquant.compression_benchmark_workflow as workflow
from nanoquant.compression_benchmark_workflow import (
    CompressionBenchmarkExperiment,
    ResolvedCompressionBenchmarkExperiment,
    execute_compression_benchmark_experiment,
    resolve_compression_benchmark_experiment,
)
from nanoquant.compression_export_workflow import (
    CompleteCompressionResult,
    CompressionExportRecipe,
    CompressionExportResult,
    ResolvedCompressionExportRecipe,
)
from nanoquant.infrastructure.commits import CommitIdentity
from nanoquant.infrastructure.gguf_export import GgufExportResult
from nanoquant.quality_evaluation import QualityEvaluationRequest
from nanoquant.resident_workflow import ResolvedResidentInputs


def _inputs(tmp_path: Path, launcher: Path) -> ResolvedResidentInputs:
    tokens = torch.zeros((256, 8), dtype=torch.long)
    return ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=tmp_path / "runs" / "001",
        registry_root=tmp_path / "runs",
        token_ids=tokens,
        quality_token_ids=tokens[:1],
        launcher_path=launcher,
        pad_token_id=0,
    )


def test_compression_benchmark_resolution_is_repository_relative(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    launcher = tmp_path / "repo" / "experiments" / "001-example.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    inputs = _inputs(tmp_path, launcher)
    monkeypatch.setattr(workflow, "resolve_resident_experiment_inputs", lambda *_args, **_kwargs: inputs)
    experiment = CompressionBenchmarkExperiment(
        CompressionExportRecipe(
            Path("outputs/logical"),
            Path("outputs/packed"),
            Path("outputs/checkpoint"),
            Path("outputs/model.gguf"),
            Path(r"D:\reference\llama.cpp"),
        ),
        Path("outputs/benchmark.json"),
    )

    resolved = resolve_compression_benchmark_experiment(
        EXPERIMENT_001_CONFIG,
        experiment,
        launcher_path=launcher,
    )

    assert resolved.export.gguf_output == tmp_path / "repo" / "outputs" / "model.gguf"
    assert resolved.benchmark_output == tmp_path / "repo" / "outputs" / "benchmark.json"
    assert resolved.export.llama_cpp_root == Path(r"D:\reference\llama.cpp")


def test_compression_benchmark_executes_export_before_shared_quality_comparison(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    launcher = tmp_path / "repo" / "experiments" / "001.py"
    inputs = _inputs(tmp_path, launcher)
    resolved = ResolvedCompressionBenchmarkExperiment(
        inputs,
        ResolvedCompressionExportRecipe(
            tmp_path / "logical",
            tmp_path / "packed",
            tmp_path / "checkpoint",
            tmp_path / "model.gguf",
            tmp_path / "llama.cpp",
            "gemma3",
        ),
        tmp_path / "benchmark.json",
    )
    quantization = SimpleNamespace(
        inventory=SimpleNamespace(
            blocks=tuple(range(26)),
            model=SimpleNamespace(config_hash="sha256:model"),
        ),
        identity=CommitIdentity("config", "model", "plan"),
        frozen_model=SimpleNamespace(effective_bpw=0.996),
        peak_device_bytes=100,
        peak_host_bytes=200,
        artifact_bytes=300,
        elapsed_seconds=400.0,
        reused_commit_count=26,
    )
    resident_result = SimpleNamespace(quantization=quantization, distillation=None)
    calls: list[str] = []
    requests: list[QualityEvaluationRequest] = []

    monkeypatch.setattr(
        workflow,
        "execute_complete_compression",
        lambda *_args, **_kwargs: calls.append("complete")
        or CompleteCompressionResult(
            resident_result,
            CompressionExportResult(
                {"exact": True},
                {"exact": True},
                GgufExportResult(
                    resolved.export.gguf_output,
                    resolved.export.checkpoint_output,
                    resolved.export.llama_cpp_root / "convert_nanoquant_to_gguf.py",
                    123,
                    "digest",
                    False,
                ),
                tmp_path / "export-summary.json",
            ),
        ),
    )

    def quality(request: QualityEvaluationRequest) -> dict[str, object]:
        calls.append("quality")
        requests.append(request)
        return {"passed": True, "comparison": {}}

    monkeypatch.setattr(workflow, "execute_quality_evaluation", quality)
    published = []
    monkeypatch.setattr(
        workflow,
        "publish_experiment_artifacts",
        lambda root, number, artifacts: published.append((root, number, tuple(artifacts))),
    )

    payload = execute_compression_benchmark_experiment(
        EXPERIMENT_001_CONFIG,
        EXPERIMENT_001,
        resolved,
    )

    assert calls == ["complete", "quality"]
    assert requests[0].wikitext_samples == 64
    assert requests[0].task_names == (
        "piqa",
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "boolq",
    )
    assert requests[0].task_limit == 200
    assert requests[0].packed_artifact == resolved.export.packed_output
    assert payload["experiment"]["comparison_labels"] == {
        "base": "bf16",
        "frozen": "nanoquant",
    }
    assert json.loads(resolved.benchmark_output.read_text(encoding="utf-8")) == payload
    assert published[0][1] == 1
    assert [artifact.source for artifact in published[0][2]][:4] == [
        resolved.export.gguf_output,
        tmp_path / "export-summary.json",
        resolved.export.gguf_output.with_suffix(".gguf.export.json"),
        resolved.benchmark_output,
    ]
