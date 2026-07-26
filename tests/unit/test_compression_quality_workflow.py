from __future__ import annotations

from contextlib import nullcontext
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


def test_llamacpp_is_required_when_pytorch_candidate_quality_is_disabled() -> None:
    with pytest.raises(ValueError, match="requires llama.cpp quality"):
        replace(_EXPERIMENT, quality_backend=None)


def test_reasoning_sequence_length_override_must_be_valid() -> None:
    with pytest.raises(ValueError, match="at least two"):
        replace(_EXPERIMENT, reasoning_sequence_length_override=1)


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
        tmp_path / "gguf-quality.json",
    )
    quantization = SimpleNamespace(
        inventory=SimpleNamespace(
            blocks=tuple(range(34)),
            total_source_bytes=1_000,
        ),
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
        {"exact": True, "packed_weight_bytes": 25},
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
        quality_backend=None,
        llamacpp_quality=True,
        llama_cpp_root=tmp_path / "llama.cpp",
        reasoning_sequence_length_override=1024,
    )
    def complete(*_args, **kwargs):  # type: ignore[no-untyped-def]
        assert "defer_huggingface" not in kwargs
        calls.append("complete")
        return CompleteCompressionResult(resident, export)

    monkeypatch.setattr(workflow, "execute_complete_compression", complete)
    prepared_quality = SimpleNamespace()
    monkeypatch.setattr(
        workflow,
        "prepare_quality_inputs",
        lambda _request: calls.append("prepare-quality") or prepared_quality,
    )

    def evaluate(request, *, prepared, evaluate_candidate):  # type: ignore[no-untyped-def]
        calls.append("quality")
        assert prepared is prepared_quality
        assert evaluate_candidate is False
        quality_requests.append(request)
        return {
            "passed": True,
            "candidate": None,
            "comparison": {},
            "resource_limits": {},
            "results": {"base": {"tasks": [], "wikitext": {"perplexity": 1.0}}},
            "protocol": {"task_names": (), "wikitext_token_hash": "sha256:tokens"},
            "wall_seconds": 2.0,
        }

    monkeypatch.setattr(workflow, "execute_quality_evaluation", evaluate)
    def evaluate_llamacpp(request, quality_request, prepared, base, protocol):  # type: ignore[no-untyped-def]
        calls.append("llamacpp-quality")
        assert request.gguf == gguf
        assert request.output == resolved.llamacpp_quality_output
        assert quality_request is quality_requests[0]
        assert prepared is prepared_quality
        assert base["wikitext"]["perplexity"] == 1.0
        assert protocol["wikitext_token_hash"] == "sha256:tokens"
        request.output.write_text("{}\n", encoding="utf-8")
        return {
            "passed": True,
            "comparison": {"wikitext": {}, "tasks": []},
            "gguf": {"path": str(gguf), "bytes": 123, "sha256": "digest"},
            "runtime": {"git": {"commit": "a" * 40}},
            "results": {
                "gguf": {
                    "label": "gguf",
                    "tasks": [],
                    "wikitext": {"perplexity": 1.1},
                    "elapsed_seconds": 3.0,
                }
            },
            "wall_seconds": 3.0,
        }

    monkeypatch.setattr(
        workflow,
        "execute_llamacpp_quality_evaluation",
        evaluate_llamacpp,
    )
    monkeypatch.setattr(workflow, "render_llamacpp_quality_markdown", lambda _payload: "# GGUF\n")
    rendered_payloads = []
    monkeypatch.setattr(
        workflow,
        "render_quality_evaluation_markdown",
        lambda payload: rendered_payloads.append(payload) or "# quality\n",
    )

    emitted_events = []
    upload_events = SimpleNamespace(
        emit=lambda component, severity, name, **fields: emitted_events.append(
            (component, severity, name, fields)
        )
    )
    monkeypatch.setattr(
        workflow,
        "open_run_event_append_session",
        lambda *_args, **_kwargs: nullcontext(upload_events),
    )

    def upload(result, config, artifacts, *, model_card_metadata, events):  # type: ignore[no-untyped-def]
        calls.append("upload")
        assert events is upload_events
        assert config is upload_config
        assert model_card_metadata.base_model == _CONFIG.model.source
        assert model_card_metadata.base_model_revision == _CONFIG.model.revision
        assert tuple(artifacts) == (
            (resolved.quality_markdown_output, "README.md"),
            (resolved.quality_output, "quality.json"),
            (resolved.llamacpp_quality_output, "gguf-quality.json"),
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

    assert calls == [
        "complete",
        "prepare-quality",
        "quality",
        "llamacpp-quality",
        "upload",
    ]
    assert quality_requests[0].packed_artifact is None
    assert not quality_requests[0].stream_base_model
    assert quality_requests[0].local_files_only is False
    assert quality_requests[0].reasoning_sequence_length == 1024
    assert quality_requests[0].reasoning_batch_size == 1
    assert rendered_payloads[0]["deployment_storage"] == {
        "bf16_checkpoint_bytes": 1_000,
        "packed_quantized_layer_bytes": 25,
        "gguf_bytes": 123,
    }
    assert rendered_payloads[0]["candidate"]["backend"] == "llama.cpp"
    assert rendered_payloads[0]["results"]["frozen"]["label"] == "gguf"
    assert rendered_payloads[0]["comparison"] == {"wikitext": {}, "tasks": []}
    assert payload["exports"]["gguf"]["output"] == str(gguf)
    assert payload["exports"]["mmproj"]["output"] == str(mmproj)
    assert payload["exports"]["huggingface"]["commit_oid"] == "a" * 40
    assert payload["quality"]["gguf_json"] == str(resolved.llamacpp_quality_output)
    assert [event[2] for event in emitted_events[:2]] == [
        "quality.llamacpp.started",
        "quality.llamacpp.completed",
    ]
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
