from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from recipes import EXPERIMENT_003_CONFIG

import nanoquant.compression_export_workflow as workflow
from nanoquant.compression_export_workflow import (
    CompressionExportRecipe,
    execute_compression_export,
    resolve_compression_export_recipe,
)
from nanoquant.infrastructure.gguf_export import GgufExportResult
from nanoquant.infrastructure.mmproj_export import MmprojExportResult
from nanoquant.resident_workflow import ResolvedResidentInputs
from nanoquant.runtime import RuntimeModelMetadata


def _recipe() -> CompressionExportRecipe:
    return CompressionExportRecipe(
        Path("outputs/logical"),
        Path("outputs/packed"),
        Path("outputs/checkpoint"),
        Path("outputs/model.gguf"),
        Path(r"D:\reference\llama.cpp"),
    )


def test_compression_export_recipe_resolves_all_material_paths(tmp_path: Path) -> None:
    resolved = resolve_compression_export_recipe(_recipe(), tmp_path)

    assert resolved.logical_output == tmp_path / "outputs" / "logical"
    assert resolved.packed_output == tmp_path / "outputs" / "packed"
    assert resolved.checkpoint_output == tmp_path / "outputs" / "checkpoint"
    assert resolved.gguf_output == tmp_path / "outputs" / "model.gguf"
    assert resolved.llama_cpp_root == Path(r"D:\reference\llama.cpp")
    assert resolved.token_embedding_type == "q8_0"


def test_complete_compression_export_runs_validated_stages_in_order(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    calls: list[object] = []
    recipe = _recipe()
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
            "sha256:gguf",
            False,
            mmproj=MmprojExportResult(
                resolved.gguf_output.parent / "mmproj-BF16.gguf",
                resolved.llama_cpp_root / "convert_hf_to_gguf.py",
                456,
                "sha256:mmproj",
                7,
                ("bf16", "f32"),
                False,
            ),
        ),
    )

    result = execute_compression_export(
        EXPERIMENT_003_CONFIG,
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
    assert summary["gguf"]["sha256"] == "sha256:gguf"
    assert summary["gguf"]["token_embedding_type"] == "q8_0"
    assert summary["schema_version"] == 3
    assert summary["mmproj"]["output"] == str(resolved.gguf_output.parent / "mmproj-BF16.gguf")
    assert summary["mmproj"]["sha256"] == "sha256:mmproj"


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
        EXPERIMENT_003_CONFIG,
        inputs,
        _recipe(),
        expected_blocks=2,
    )

    assert calls == ["compress", "export"]
    assert result.workflow is resident
    assert result.exports is export
