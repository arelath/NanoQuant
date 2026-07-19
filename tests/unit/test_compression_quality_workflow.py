from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import nanoquant.compression_quality_workflow as workflow
from nanoquant.compression_export_workflow import CompleteCompressionResult, CompressionExportResult
from nanoquant.compression_quality_workflow import (
    ResolvedCompressionQualityExperiment,
    execute_compression_quality_experiment,
)
from nanoquant.infrastructure.commits import CommitIdentity
from nanoquant.infrastructure.gguf_export import GgufExportResult
from nanoquant.infrastructure.huggingface_upload import (
    HuggingFaceUploadConfig,
    HuggingFaceUploadResult,
)
from nanoquant.infrastructure.mmproj_export import MmprojExportResult
from nanoquant.resident_workflow import ResolvedResidentInputs
from tests.support.experiments import load_experiment

_DEFINITION = load_experiment(3)
_CONFIG = _DEFINITION.config
_EXPERIMENT = _DEFINITION.workflow


def test_compression_quality_runs_quality_before_huggingface_upload_and_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    launcher = tmp_path / "repo" / "experiments" / "003.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    inputs = ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=run,
        registry_root=tmp_path / "registry",
        token_ids=torch.zeros((1, 8), dtype=torch.long),
        quality_token_ids=None,
        launcher_path=launcher,
        pad_token_id=0,
    )
    resolved = ResolvedCompressionQualityExperiment(
        inputs,
        tmp_path / "summary.json",
        tmp_path / "quality.json",
        tmp_path / "quality.md",
    )
    quantization = SimpleNamespace(
        inventory=SimpleNamespace(blocks=tuple(range(34))),
        identity=CommitIdentity("config", "model", "plan"),
        frozen_model=SimpleNamespace(effective_bpw=0.998),
        peak_device_bytes=100,
        peak_host_bytes=200,
        artifact_bytes=300,
        reused_commit_count=0,
        elapsed_seconds=400.0,
    )
    resident = SimpleNamespace(quantization=quantization, distillation=None)
    gguf = tmp_path / "repo" / "outputs" / "model.gguf"
    mmproj = gguf.parent / "mmproj-BF16.gguf"
    export = CompressionExportResult(
        {"exact": True},
        {"exact": True},
        GgufExportResult(
            gguf,
            tmp_path / "checkpoint",
            tmp_path / "converter.py",
            123,
            "digest",
            False,
            mmproj=MmprojExportResult(
                mmproj,
                tmp_path / "convert_hf_to_gguf.py",
                456,
                "mmproj-digest",
                7,
                ("bf16", "f32"),
                False,
            ),
        ),
        tmp_path / "export-summary.json",
    )
    calls: list[str] = []
    quality_requests = []
    upload_config = HuggingFaceUploadConfig("owner/model")
    experiment = replace(
        _EXPERIMENT,
        export=replace(_EXPERIMENT.export, huggingface=upload_config),
    )
    monkeypatch.setattr(
        workflow,
        "ensure_huggingface_model_repository",
        lambda config: calls.append("preflight") or config.repo_id,
    )

    def complete(*_args, **kwargs):  # type: ignore[no-untyped-def]
        assert "defer_huggingface" not in kwargs
        calls.append("complete")
        return CompleteCompressionResult(resident, export)

    monkeypatch.setattr(workflow, "execute_complete_compression", complete)
    def evaluate(request):  # type: ignore[no-untyped-def]
        calls.append("quality")
        quality_requests.append(request)
        return {"passed": True, "comparison": {}, "resource_limits": {}}

    monkeypatch.setattr(workflow, "execute_quality_evaluation", evaluate)
    monkeypatch.setattr(workflow, "render_quality_evaluation_markdown", lambda _payload: "# quality\n")

    def upload(result, config, artifacts):  # type: ignore[no-untyped-def]
        calls.append("upload")
        assert config is upload_config
        assert tuple(artifacts) == (
            (resolved.quality_markdown_output, "README.md"),
            (resolved.quality_output, "quality.json"),
        )
        assert resolved.quality_output.is_file()
        assert resolved.quality_markdown_output.read_text(encoding="utf-8") == "# quality\n"
        return replace(
            result,
            huggingface=HuggingFaceUploadResult(
                "owner/model",
                "https://huggingface.co/owner/model",
                "a" * 40,
                f"https://huggingface.co/owner/model/commit/{'a' * 40}",
                None,
                config.commit_message,
                (),
                gguf.with_suffix(".gguf.huggingface.json"),
            ),
        )

    monkeypatch.setattr(workflow, "complete_deferred_huggingface_upload", upload)
    published = []
    monkeypatch.setattr(
        workflow,
        "publish_experiment_artifacts",
        lambda root, number, artifacts: published.append((root, number, tuple(artifacts))),
    )

    payload = execute_compression_quality_experiment(
        _CONFIG,
        experiment,
        resolved,
    )

    assert calls == ["preflight", "complete", "quality", "upload"]
    assert quality_requests[0].packed_artifact == tmp_path / "repo" / "outputs/003/packed"
    assert not quality_requests[0].stream_base_model
    assert quality_requests[0].local_files_only is False
    assert payload["exports"]["gguf"]["output"] == str(gguf)
    assert payload["exports"]["mmproj"]["output"] == str(mmproj)
    assert payload["exports"]["huggingface"]["commit_oid"] == "a" * 40
    assert published[0][1] == 3
    assert [artifact.source for artifact in published[0][2]][:5] == [
        gguf,
        tmp_path / "export-summary.json",
        gguf.with_suffix(".gguf.export.json"),
        mmproj,
        mmproj.with_suffix(".gguf.export.json"),
    ]


def test_large_model_guard_rejects_resident_recipe_before_compression(
    tmp_path: Path,
) -> None:
    inputs = ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=tmp_path / "run",
        registry_root=tmp_path / "registry",
        token_ids=torch.zeros((1, 8), dtype=torch.long),
        quality_token_ids=None,
        launcher_path=tmp_path / "experiments/003.py",
        pad_token_id=0,
    )
    resolved = ResolvedCompressionQualityExperiment(
        inputs,
        tmp_path / "summary.json",
        tmp_path / "quality.json",
        tmp_path / "quality.md",
    )
    guarded = replace(_EXPERIMENT, large_model_guards=True)

    with pytest.raises(ValueError, match="cpu_offload or streaming"):
        execute_compression_quality_experiment(_CONFIG, guarded, resolved)
