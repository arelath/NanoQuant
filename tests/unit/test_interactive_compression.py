from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from recipes import INTERACTIVE_RECOMMENDED_MODELS

import nanoquant.interactive_compression as interactive
from nanoquant.infrastructure.huggingface_upload import HuggingFaceUploadConfig
from nanoquant.interactive_compression import (
    INTERACTIVE_COMPLETION_NAME,
    INTERACTIVE_SETTINGS_NAME,
    create_interactive_settings,
    discover_interactive_runs,
    execute_interactive_run,
    load_interactive_settings,
    resolve_custom_model,
    run_interactive_launcher,
    write_interactive_settings,
)


def _launcher(root: Path) -> Path:
    launcher = root / "tools" / "compress_model.py"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    return launcher


def _inputs(values: list[str]) -> Any:
    iterator = iter(values)

    def read(_prompt: str) -> str:
        return next(iterator)

    return read


def _model(variant: str = "qwen3-0-6b"):  # type: ignore[no-untyped-def]
    return next(item for item in INTERACTIVE_RECOMMENDED_MODELS if item.variant == variant)


@pytest.fixture(autouse=True)
def _resolve_promoted_model_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    configs = {
        "Qwen/Qwen3-0.6B": ("qwen3", 28),
        "Qwen/Qwen3-8B": ("qwen3", 36),
        "unsloth/gemma-3-270m-it": ("gemma3_text", 18),
        "google/gemma-3-1b-it": ("gemma3_text", 26),
        "google/gemma-3-4b-it": ("gemma3", 34),
        "unsloth/gemma-3-12b-it": ("gemma3", 48),
        "meta-llama/Meta-Llama-3-8B-Instruct": ("llama", 32),
        "meta-llama/Llama-3.2-1B-Instruct": ("llama", 16),
        "meta-llama/Llama-3.2-3B-Instruct": ("llama", 28),
    }

    def resolve(source: str, revision: str | None):  # type: ignore[no-untyped-def]
        architecture, blocks = configs[source]
        config = {"model_type": architecture, "num_hidden_layers": blocks}
        if architecture == "gemma3":
            config["text_config"] = {"num_hidden_layers": blocks}
        return config, revision or "resolved-revision"

    monkeypatch.setattr(interactive, "_model_config", resolve)


def _persist(root: Path, settings) -> Path:  # type: ignore[no-untyped-def]
    path = root / settings.paths.run_output / INTERACTIVE_SETTINGS_NAME
    path.parent.mkdir(parents=True)
    write_interactive_settings(path, settings)
    return path


def test_promoted_catalog_groups_family_then_size_and_keeps_qwen_dual_mode() -> None:
    families = interactive._families(INTERACTIVE_RECOMMENDED_MODELS)

    assert families == (
        ("qwen3", "Qwen3"),
        ("qwen3-5", "Qwen3.5"),
        ("qwen3-6", "Qwen3.6"),
        ("gemma3", "Gemma3"),
        ("llama3", "Llama3"),
        ("llama3-2", "Llama3.2"),
    )
    assert [item.variant_label for item in interactive._variants(
        INTERACTIVE_RECOMMENDED_MODELS, "qwen3"
    )] == [
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
    ]
    assert [item.variant_label for item in interactive._variants(
        INTERACTIVE_RECOMMENDED_MODELS, "qwen3-5"
    )] == [
        "Qwen3.5-0.8B",
        "Qwen3.5-2B",
        "Qwen3.5-4B",
        "Qwen3.5-9B",
        "Qwen3.5-27B",
    ]
    assert [item.variant_label for item in interactive._variants(
        INTERACTIVE_RECOMMENDED_MODELS, "qwen3-6"
    )] == [
        "Qwen3.6-27B",
    ]
    assert all(
        item.template.dataset.behavior_slices
        for item in INTERACTIVE_RECOMMENDED_MODELS
        if item.family.startswith("qwen3")
    )


def test_new_launcher_run_uses_family_size_prompts_and_persists_before_dispatch(
    tmp_path: Path,
) -> None:
    launcher = _launcher(tmp_path)
    outputs: list[str] = []
    executed: list[Path] = []

    def execute(path: Path, observed_launcher: Path) -> int:
        assert observed_launcher == launcher
        settings, _digest = load_interactive_settings(path)
        assert settings.selection.model_family == "qwen3"
        assert settings.selection.model_variant == "qwen3-8b"
        assert settings.selection.target_bpw == 1.25
        assert not settings.selection.quality_requested
        assert settings.selection.huggingface is None
        assert settings.run_config.allocation.target_bpw == 1.25
        assert path.is_file()
        executed.append(path)
        return 0

    result = run_interactive_launcher(
        tmp_path,
        launcher,
        INTERACTIVE_RECOMMENDED_MODELS,
        input_fn=_inputs(["", "4", "1.25", "n", "", ""]),
        write=outputs.append,
        execute=execute,
    )

    assert result == 0
    assert len(executed) == 1
    assert executed[0].parent.parent == tmp_path / "evidence" / "interactive"
    assert "Choose a model family:" in outputs
    assert "Choose a Qwen3 model:" in outputs
    assert any(line.startswith("Settings written:") for line in outputs)


def test_second_invocation_defaults_to_continuing_previous_settings(tmp_path: Path) -> None:
    launcher = _launcher(tmp_path)
    settings = create_interactive_settings(
        _model("llama-3-2-3b-instruct"),
        target_bpw=0.9,
        quality_requested=True,
        huggingface=None,
        llama_cpp_root=tmp_path / "llama.cpp",
        now=datetime(2026, 7, 25, tzinfo=timezone.utc),
        run_name="previous-llama-run",
    )
    settings_path = _persist(tmp_path, settings)
    dispatched: list[Path] = []

    assert run_interactive_launcher(
        tmp_path,
        launcher,
        INTERACTIVE_RECOMMENDED_MODELS,
        input_fn=_inputs([""]),
        write=lambda _line: None,
        execute=lambda path, _launcher_path: dispatched.append(path) or 0,
    ) == 0

    assert dispatched == [settings_path.resolve()]


def test_previous_model_becomes_family_and_variant_default_for_new_run(
    tmp_path: Path,
) -> None:
    launcher = _launcher(tmp_path)
    previous = create_interactive_settings(
        _model("qwen3-8b"),
        target_bpw=1.0,
        quality_requested=True,
        huggingface=None,
        llama_cpp_root=tmp_path / "llama.cpp",
        now=datetime(2026, 7, 25, tzinfo=timezone.utc),
        run_name="previous-qwen-run",
    )
    _persist(tmp_path, previous)
    selected: list[str] = []

    def execute(path: Path, _launcher_path: Path) -> int:
        settings, _digest = load_interactive_settings(path)
        selected.append(settings.selection.model_variant)
        return 0

    # Start new, accept the previous family, accept the previous variant, then
    # accept BPW/quality/upload/confirmation defaults.
    run_interactive_launcher(
        tmp_path,
        launcher,
        INTERACTIVE_RECOMMENDED_MODELS,
        input_fn=_inputs(["2", "", "", "", "", "", ""]),
        write=lambda _line: None,
        execute=execute,
    )

    assert selected == ["qwen3-8b"]


def test_settings_are_strict_and_tamper_evident(tmp_path: Path) -> None:
    settings = create_interactive_settings(
        _model(),
        target_bpw=1.0,
        quality_requested=True,
        huggingface=HuggingFaceUploadConfig("owner/model", private=True),
        llama_cpp_root=tmp_path / "llama.cpp",
        run_name="strict-settings",
    )
    path = _persist(tmp_path, settings)
    loaded, digest = load_interactive_settings(path)
    assert loaded == settings
    assert digest.startswith("sha256:")
    text = path.read_text(encoding="utf-8").replace("target_bpw: 1.0", "target_bpw: 2.0")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_interactive_settings(path)
    assert discover_interactive_runs(tmp_path) == ()


def test_schema_one_settings_migrate_the_derived_block_count_on_load(tmp_path: Path) -> None:
    settings = create_interactive_settings(
        _model(),
        target_bpw=1.0,
        quality_requested=False,
        huggingface=None,
        llama_cpp_root=tmp_path / "llama.cpp",
        run_name="legacy-settings",
    )
    body = interactive._settings_body(settings)
    body["schema_version"] = 1
    resolved_source = body["resolved_source"]
    resolved_source["expected_blocks"] = resolved_source.pop("block_count")
    digest = interactive.semantic_hash(body)
    path = tmp_path / "settings.yaml"
    path.write_text(
        yaml.safe_dump({"settings_hash": digest, **body}, sort_keys=False),
        encoding="utf-8",
    )

    loaded, observed = load_interactive_settings(path)

    assert observed == digest
    assert loaded.schema_version == 2
    assert loaded.resolved_source.block_count == 28


def test_custom_model_inherits_nearest_compatible_family_profile(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        interactive,
        "_model_config",
        lambda _source, _revision: (
            {"model_type": "qwen3", "num_hidden_layers": 40},
            "a" * 40,
        ),
    )

    resolved = resolve_custom_model(
        "owner/Qwen3-Custom",
        INTERACTIVE_RECOMMENDED_MODELS,
    )

    assert resolved.family == "qwen3"
    assert interactive._resolved_source_for_model(resolved).block_count == 40
    assert resolved.source == "owner/Qwen3-Custom"
    assert resolved.variant == resolved.release_name == "qwen3-custom"
    assert resolved.variant_label == "Qwen3-Custom"
    assert resolved.revision == "a" * 40
    assert resolved.template.dataset.behavior_slices
    assert resolved.template.model.source == "owner/Qwen3-Custom"


def test_quality_dispatch_uses_non_numbered_quality_workflow_and_writes_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    launcher = _launcher(tmp_path)
    settings = create_interactive_settings(
        _model(),
        target_bpw=1.0,
        quality_requested=True,
        huggingface=None,
        llama_cpp_root=tmp_path / "llama.cpp",
        run_name="quality-run",
    )
    settings_path = _persist(tmp_path, settings)
    summary = tmp_path / settings.paths.summary_output
    gguf = tmp_path / settings.paths.gguf_output
    calls = []

    def run_quality(config, experiment, *, launcher_path):  # type: ignore[no-untyped-def]
        calls.append((config, experiment, launcher_path))
        summary.parent.mkdir(parents=True, exist_ok=True)
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"gguf")
        summary.write_text(
            json.dumps(
                {
                    "passed": True,
                    "exports": {
                        "gguf": {
                            "bytes": gguf.stat().st_size,
                            "sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
                        },
                        "huggingface": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(interactive, "run_compression_quality_experiment", run_quality)

    assert execute_interactive_run(settings_path, launcher) == 0
    assert calls[0][0].intent.experiment_number is None
    assert settings.resolved_source.block_count == 28
    assert calls[0][1].llamacpp_quality
    completion = json.loads(
        (settings_path.parent / INTERACTIVE_COMPLETION_NAME).read_text(encoding="utf-8")
    )
    assert completion["passed"] is True
    assert completion["gguf"]["path"] == str(gguf)


def test_completed_interactive_run_is_not_dispatched_again(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    launcher = _launcher(tmp_path)
    settings = create_interactive_settings(
        _model(),
        target_bpw=1.0,
        quality_requested=False,
        huggingface=None,
        llama_cpp_root=tmp_path / "llama.cpp",
        run_name="completed-run",
    )
    settings_path = _persist(tmp_path, settings)
    summary = tmp_path / settings.paths.summary_output
    gguf = tmp_path / settings.paths.gguf_output
    summary.parent.mkdir(parents=True, exist_ok=True)
    gguf.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("{}\n", encoding="utf-8")
    gguf.write_bytes(b"gguf")
    _settings, digest = load_interactive_settings(settings_path)
    (settings_path.parent / INTERACTIVE_COMPLETION_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "settings_hash": digest,
                "summary": str(summary),
                "gguf": {
                    "path": str(gguf),
                    "bytes": gguf.stat().st_size,
                    "sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
                },
                "huggingface_receipt": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        interactive,
        "_execute_without_quality",
        lambda *_args, **_kwargs: pytest.fail("completed run must not execute"),
    )

    assert execute_interactive_run(settings_path, launcher) == 0
