from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import ModelCard

import nanoquant.compression_export_workflow as workflow
from nanoquant.compression_export_workflow import (
    CompressionExportRecipe,
    CompressionExportResult,
    HuggingFaceUploadConfig,
    HuggingFaceUploadResult,
    complete_deferred_huggingface_upload,
    execute_compression_export,
    resolve_compression_export_recipe,
)
from nanoquant.infrastructure.gguf_export import GgufExportResult
from nanoquant.infrastructure.huggingface_model_card import HuggingFaceModelCardMetadata
from nanoquant.infrastructure.mmproj_export import MmprojExportResult
from nanoquant.resident_workflow import ResolvedResidentInputs
from nanoquant.runtime import RuntimeModelMetadata
from tests.support.experiments import load_experiment

_CONFIG = load_experiment(3).config


def _recipe() -> CompressionExportRecipe:
    return CompressionExportRecipe(
        Path("outputs/logical"),
        Path("outputs/packed"),
        Path("outputs/checkpoint"),
        Path("Results/003/model.gguf"),
        Path(r"D:\reference\llama.cpp"),
    )


def test_compression_export_recipe_resolves_all_material_paths(tmp_path: Path) -> None:
    resolved = resolve_compression_export_recipe(_recipe(), tmp_path)

    assert resolved.logical_output == tmp_path / "outputs" / "logical"
    assert resolved.packed_output == tmp_path / "outputs" / "packed"
    assert resolved.checkpoint_output == tmp_path / "outputs" / "checkpoint"
    assert resolved.gguf_output == tmp_path / "Results" / "003" / "model.gguf"
    assert resolved.llama_cpp_root == Path(r"D:\reference\llama.cpp")
    assert resolved.token_embedding_type == "q8_0"


def test_complete_compression_export_runs_validated_stages_in_order(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    calls: list[object] = []
    recipe = replace(_recipe(), huggingface=HuggingFaceUploadConfig("owner/model"))
    resolved = resolve_compression_export_recipe(recipe, tmp_path)
    monkeypatch.setattr(
        workflow,
        "_runtime_metadata",
        lambda *_args: RuntimeModelMetadata("source", "revision", "gemma3", "config", "tokenizer"),
    )
    monkeypatch.setattr(
        workflow,
        "_ensure_logical_export",
        lambda *_args, **kwargs: calls.append(("logical", kwargs["use_global_tuning"]))
        or {"exact": True},
    )
    monkeypatch.setattr(
        workflow,
        "_ensure_packed_export",
        lambda *_args: calls.append("packed") or {"exact": True},
    )
    monkeypatch.setattr(
        workflow,
        "export_llamacpp_gguf",
        lambda *_args, **kwargs: calls.append(("gguf", kwargs["token_embedding_type"]))
        or GgufExportResult(
            resolved.gguf_output,
            resolved.checkpoint_output,
            resolved.llama_cpp_root / "convert_nanoquant_to_gguf.py",
            123,
            "a" * 64,
            False,
            mmproj=MmprojExportResult(
                resolved.gguf_output.parent / "mmproj-BF16.gguf",
                resolved.llama_cpp_root / "convert_hf_to_gguf.py",
                456,
                "b" * 64,
                7,
                ("bf16", "f32"),
                False,
            ),
        ),
    )
    monkeypatch.setattr(
        workflow,
        "upload_validated_model_artifacts",
        lambda *_args, **_kwargs: pytest.fail("local export must not contact Hugging Face"),
    )

    result = execute_compression_export(
        _CONFIG,
        recipe,
        repository_root=tmp_path,
        run_output=tmp_path / "run",
        snapshot=tmp_path / "snapshot",
        expected_blocks=34,
    )

    assert calls == [("logical", True), "packed", ("gguf", "q8_0")]
    assert result.logical == {"exact": True}
    assert result.packed == {"exact": True}
    assert result.gguf.output == resolved.gguf_output
    assert result.summary_output == resolved.gguf_output.with_suffix(".export-summary.json")
    assert result.summary_output.is_file()
    summary = json.loads(result.summary_output.read_text(encoding="utf-8"))
    assert summary["logical"] == {"exact": True}
    assert summary["packed"] == {"exact": True}
    assert summary["gguf"]["sha256"] == "a" * 64
    assert summary["gguf"]["token_embedding_type"] == "q8_0"
    assert summary["schema_version"] == 4
    assert summary["mmproj"]["output"] == str(resolved.gguf_output.parent / "mmproj-BF16.gguf")
    assert summary["mmproj"]["sha256"] == "b" * 64
    assert summary["huggingface"] is None
    assert result.huggingface is None


def test_deferred_huggingface_upload_includes_quality_documents_and_refreshes_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    gguf = tmp_path / "Results" / "009" / "model.gguf"
    mmproj = gguf.parent / "mmproj-BF16.gguf"
    quality_markdown = gguf.parent / "quality.md"
    quality_json = tmp_path / "outputs" / "009" / "quality.json"
    summary = gguf.with_suffix(".export-summary.json")
    for path, content in (
        (gguf, b"model"),
        (mmproj, b"projector"),
        (quality_markdown, b"# Quality\n"),
        (quality_json, b'{"passed":true}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    summary.write_text('{"schema_version":4,"huggingface":null}\n', encoding="utf-8")
    result = CompressionExportResult(
        {},
        {},
        GgufExportResult(
            gguf,
            tmp_path / "checkpoint",
            tmp_path / "converter.py",
            gguf.stat().st_size,
            "a" * 64,
            True,
            mmproj=MmprojExportResult(
                mmproj,
                tmp_path / "mmproj-converter.py",
                mmproj.stat().st_size,
                "b" * 64,
                1,
                ("bf16",),
                True,
            ),
        ),
        summary,
    )
    captured = []

    def upload(config, artifacts, *, receipt_output):  # type: ignore[no-untyped-def]
        captured.append((config, tuple(artifacts), receipt_output))
        return HuggingFaceUploadResult(
            "owner/model",
            "https://huggingface.co/owner/model",
            "c" * 40,
            f"https://huggingface.co/owner/model/commit/{'c' * 40}",
            False,
            config.commit_message,
            (),
            receipt_output,
        )

    monkeypatch.setattr(workflow, "upload_validated_model_artifacts", upload)

    with pytest.raises(ValueError, match="requires model-card metadata"):
        complete_deferred_huggingface_upload(
            result,
            HuggingFaceUploadConfig("owner/model", private=False),
        )

    completed = complete_deferred_huggingface_upload(
        result,
        HuggingFaceUploadConfig("owner/model", private=False),
        (
            (quality_markdown, "README.md"),
            (quality_json, "quality.json"),
        ),
        model_card_metadata=HuggingFaceModelCardMetadata("owner/base-model", "revision"),
    )

    artifacts = captured[0][1]
    model_card = gguf.with_suffix(".model-card.md")
    assert [artifact.source for artifact in artifacts] == [
        gguf,
        mmproj,
        model_card,
        quality_json,
    ]
    assert [artifact.path_in_repo for artifact in artifacts] == [
        "model.gguf",
        "mmproj-BF16.gguf",
        "README.md",
        "quality.json",
    ]
    card = ModelCard.load(model_card)
    assert card.data.get("base_model") == "owner/base-model"
    assert card.data.get("base_model_relation") == "quantized"
    assert card.data.get("pipeline_tag") == "image-text-to-text"
    assert card.text.strip() == "# Quality"
    assert quality_markdown.read_bytes() == b"# Quality\n"
    assert completed.huggingface is not None
    assert json.loads(summary.read_text(encoding="utf-8"))["huggingface"]["commit_oid"] == "c" * 40


def test_deferred_huggingface_upload_generates_a_card_when_no_report_body_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    gguf = tmp_path / "Results" / "001" / "model.gguf"
    quality_json = tmp_path / "outputs" / "001" / "quality.json"
    summary = gguf.with_suffix(".export-summary.json")
    for path, content in ((gguf, b"model"), (quality_json, b'{"passed":true}\n')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    summary.write_text('{"schema_version":4,"huggingface":null}\n', encoding="utf-8")
    result = CompressionExportResult(
        {},
        {},
        GgufExportResult(
            gguf,
            tmp_path / "checkpoint",
            tmp_path / "converter.py",
            gguf.stat().st_size,
            "a" * 64,
            True,
        ),
        summary,
    )
    captured = []

    def upload(config, artifacts, *, receipt_output):  # type: ignore[no-untyped-def]
        captured.extend(artifacts)
        return HuggingFaceUploadResult(
            config.repo_id,
            f"https://huggingface.co/{config.repo_id}",
            "d" * 40,
            f"https://huggingface.co/{config.repo_id}/commit/{'d' * 40}",
            None,
            config.commit_message,
            (),
            receipt_output,
        )

    monkeypatch.setattr(workflow, "upload_validated_model_artifacts", upload)

    complete_deferred_huggingface_upload(
        result,
        HuggingFaceUploadConfig("owner/model"),
        ((quality_json, "quality.json"),),
        model_card_metadata=HuggingFaceModelCardMetadata("owner/base-model", "revision"),
    )

    assert [artifact.path_in_repo for artifact in captured] == [
        "model.gguf",
        "README.md",
        "quality.json",
    ]
    card = ModelCard.load(gguf.with_suffix(".model-card.md"))
    assert card.data.get("pipeline_tag") == "text-generation"
    assert "# model" in card.text
    assert "`revision`" in card.text


def test_base_compression_requires_export_after_resident_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    launcher = tmp_path / "repo" / "experiments" / "003.py"
    inputs = ResolvedResidentInputs(
        tmp_path / "snapshot",
        tmp_path / "run",
        tmp_path / "registry",
        ((1, 2, 3),),
        None,
        launcher_path=launcher,
    )
    resident = SimpleNamespace(quantization=SimpleNamespace(inventory=SimpleNamespace(blocks=(0, 1))))
    export = SimpleNamespace()
    calls: list[str] = []
    monkeypatch.setattr(
        workflow,
        "execute_resident_workflow",
        lambda *_args: calls.append("compress") or resident,
    )
    monkeypatch.setattr(
        workflow,
        "execute_compression_export",
        lambda *_args, **_kwargs: calls.append("export") or export,
    )

    result = workflow.execute_complete_compression(
        _CONFIG,
        inputs,
        _recipe(),
        expected_blocks=2,
    )

    assert calls == ["compress", "export"]
    assert result.workflow is resident
    assert result.exports is export
    assert (inputs.output / "weight-errors.md").is_file()
    assert (launcher.parent.parent / "Results" / "003" / "weight-errors.md").is_file()


def test_compression_export_rejects_gguf_outside_numbered_results(tmp_path: Path) -> None:
    recipe = replace(_recipe(), gguf_output=Path("outputs/003/model.gguf"))

    with pytest.raises(ValueError, match="numbered Results directory"):
        execute_compression_export(
            _CONFIG,
            recipe,
            repository_root=tmp_path,
            run_output=tmp_path / "run",
            snapshot=tmp_path / "snapshot",
            expected_blocks=34,
        )


def test_compression_export_adopts_validated_legacy_gguf_without_copying(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    recipe = _recipe()
    resolved = resolve_compression_export_recipe(recipe, tmp_path)
    legacy_output = tmp_path / "outputs" / "003" / "model.gguf"
    legacy_receipt = legacy_output.with_suffix(".gguf.export.json")
    legacy_output.parent.mkdir(parents=True)
    legacy_output.write_bytes(b"validated model")
    legacy_receipt.write_text('{"schema_version":3}\n', encoding="utf-8")
    monkeypatch.setattr(
        workflow,
        "_runtime_metadata",
        lambda *_args: RuntimeModelMetadata("source", "revision", "gemma3", "config", "tokenizer"),
    )
    monkeypatch.setattr(workflow, "_ensure_logical_export", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow, "_ensure_packed_export", lambda *_args: {})
    calls: list[Path] = []

    def export(*args, **_kwargs):  # type: ignore[no-untyped-def]
        output = Path(args[3])
        calls.append(output)
        return GgufExportResult(
            output,
            resolved.checkpoint_output,
            resolved.llama_cpp_root / "convert_nanoquant_to_gguf.py",
            output.stat().st_size,
            "a" * 64,
            True,
        )

    monkeypatch.setattr(workflow, "export_llamacpp_gguf", export)

    result = execute_compression_export(
        _CONFIG,
        recipe,
        repository_root=tmp_path,
        run_output=tmp_path / "run",
        snapshot=tmp_path / "snapshot",
        expected_blocks=34,
    )

    assert calls == [legacy_output.resolve(), resolved.gguf_output]
    assert os.path.samefile(legacy_output, resolved.gguf_output)
    assert os.path.samefile(
        legacy_receipt,
        resolved.gguf_output.with_suffix(".gguf.export.json"),
    )
    assert result.gguf.output == resolved.gguf_output


def test_complete_compression_reuses_terminal_completed_workflow_before_export(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    launcher = tmp_path / "repo" / "experiments" / "003.py"
    inputs = ResolvedResidentInputs(
        tmp_path / "snapshot",
        tmp_path / "run",
        tmp_path / "registry",
        ((1, 2, 3),),
        None,
        launcher_path=launcher,
    )
    resident = SimpleNamespace(quantization=SimpleNamespace(inventory=SimpleNamespace(blocks=(0, 1))))
    export = SimpleNamespace()
    calls: list[str] = []
    monkeypatch.setattr(
        workflow,
        "load_completed_resident_workflow",
        lambda *_args: calls.append("load-completed") or resident,
    )
    monkeypatch.setattr(
        workflow,
        "initialize_live_weight_error_report",
        lambda *_args, **_kwargs: calls.append("initialize-report"),
    )
    monkeypatch.setattr(
        workflow,
        "execute_resident_workflow",
        lambda *_args: calls.append("compress") or resident,
    )
    monkeypatch.setattr(
        workflow,
        "execute_compression_export",
        lambda *_args, **_kwargs: calls.append("export") or export,
    )

    result = workflow.execute_complete_compression(
        _CONFIG,
        inputs,
        _recipe(),
        expected_blocks=2,
    )

    assert calls == ["load-completed", "export"]
    assert result.workflow is resident
    assert result.exports is export
