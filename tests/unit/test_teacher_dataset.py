from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import build_teacher_dataset as teacher_tool
import pytest
import yaml

import nanoquant.teacher_dataset as teacher_dataset
from nanoquant.config.schema import ReasoningMode
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.teacher_trace_generation import PreparedTeacherTraces
from nanoquant.teacher_dataset import (
    TEACHER_DATASET_COMPLETION_NAME,
    TEACHER_DATASET_DIRECTORY_NAME,
    TEACHER_DATASET_MANIFEST_NAME,
    TEACHER_DATASET_SETTINGS_NAME,
    TeacherDatasetGeneration,
    TeacherDatasetUpload,
    TeacherModel,
    TeacherPromptSource,
    execute_teacher_dataset,
    load_teacher_dataset_settings,
    new_teacher_dataset_settings,
    normalize_prompt_record,
    run_interactive_teacher_dataset,
    write_teacher_dataset_settings,
)


def _settings():
    return new_teacher_dataset_settings(
        prompt_source=TeacherPromptSource(
            "HuggingFaceH4/ultrachat_200k",
            "dataset-revision",
            "train_sft",
        ),
        teacher=TeacherModel(
            "unsloth/Qwen3-8B-GGUF",
            "teacher-revision",
            "unsloth/Qwen3-8B",
            "tokenizer-revision",
            "Qwen3-8B-BF16.gguf",
            "llamacpp-server-greedy-qwen3-v1",
            "cpu",
        ),
        generation=TeacherDatasetGeneration(
            samples_per_mode=2,
            sequence_length=512,
            maximum_new_tokens=256,
            minimum_new_tokens=1,
        ),
        upload=None,
        created_at="2026-07-28T00:00:00+00:00",
    )


def _inputs(values: list[str]):
    iterator = iter(values)

    def read(_prompt: str) -> str:
        return next(iterator)

    return read


def test_teacher_dataset_settings_are_tamper_evident(tmp_path: Path) -> None:
    path = tmp_path / TEACHER_DATASET_SETTINGS_NAME
    settings = _settings()
    digest = write_teacher_dataset_settings(path, settings)

    loaded, observed = load_teacher_dataset_settings(path)

    assert loaded == settings
    assert observed == digest
    path.write_text(
        path.read_text(encoding="utf-8").replace("samples_per_mode: 2", "samples_per_mode: 3"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_teacher_dataset_settings(path)


def test_schema_one_settings_migrate_teacher_tokenizer_identity(tmp_path: Path) -> None:
    body = teacher_dataset._settings_body(_settings())
    body["schema_version"] = 1
    teacher = body["teacher"]
    teacher["source"] = "Qwen/Qwen3-8B"
    teacher["revision"] = "official-revision"
    teacher.pop("tokenizer_source")
    teacher.pop("tokenizer_revision")
    teacher.pop("gguf_filename")
    digest = teacher_dataset.semantic_hash(body)
    path = tmp_path / TEACHER_DATASET_SETTINGS_NAME
    path.write_text(
        yaml.safe_dump({"settings_hash": digest, **body}, sort_keys=False),
        encoding="utf-8",
    )

    loaded, observed = load_teacher_dataset_settings(path)

    assert observed == digest
    assert loaded.teacher.tokenizer_source == "Qwen/Qwen3-8B"
    assert loaded.teacher.tokenizer_revision == "official-revision"
    assert loaded.teacher.gguf_filename is None


def test_prompt_normalizer_accepts_common_chat_schemas_and_rejects_incomplete_turns() -> None:
    normalized = normalize_prompt_record(
        {
            "id": "row-1",
            "conversation": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Original answer"},
            ],
        },
        messages_column="conversation",
    )

    assert normalized == {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Original answer"},
        ],
        "source_record_id": "row-1",
    }
    with pytest.raises(ValueError, match="does not end in an assistant"):
        normalize_prompt_record(
            {"messages": [{"role": "assistant", "content": "A"}, {"role": "user", "content": "Q"}]},
            messages_column="messages",
        )


def test_parameterized_tool_exposes_teacher_source_subset_mode_resume_and_upload_controls() -> None:
    args = teacher_tool._parser().parse_args(
        [
            "--output",
            "evidence/teacher-datasets/example",
            "--teacher-model",
            "unsloth/Qwen3-8B-GGUF",
            "--teacher-tokenizer",
            "unsloth/Qwen3-8B",
            "--teacher-gguf-file",
            "Qwen3-8B-BF16.gguf",
            "--source-dataset",
            "owner/conversations",
            "--source-config",
            "default",
            "--source-split",
            "train",
            "--messages-column",
            "conversation",
            "--mode",
            "thinking",
            "--samples-per-mode",
            "64",
            "--backend",
            "transformers",
            "--hub-repo",
            "owner/generated-responses",
            "--public",
        ]
    )

    assert args.teacher_model == "unsloth/Qwen3-8B-GGUF"
    assert args.teacher_tokenizer == "unsloth/Qwen3-8B"
    assert args.teacher_gguf_file == "Qwen3-8B-BF16.gguf"
    assert args.source_dataset == "owner/conversations"
    assert args.source_config == "default"
    assert args.messages_column == "conversation"
    assert args.mode == "thinking"
    assert args.samples_per_mode == 64
    assert args.backend == "transformers"
    assert args.hub_repo == "owner/generated-responses"
    assert args.public


def test_unsloth_gguf_snapshot_downloads_only_the_selected_bf16_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    shard_root = snapshot / "BF16"
    shard_root.mkdir(parents=True)
    for index in (1, 2):
        (shard_root / f"Qwen3-32B-BF16-{index:05d}-of-00002.gguf").write_bytes(b"gguf")
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(snapshot)

    monkeypatch.setattr(teacher_dataset, "snapshot_download", download)
    teacher = TeacherModel(
        "unsloth/Qwen3-32B-GGUF",
        "model-revision",
        "unsloth/Qwen3-32B",
        "tokenizer-revision",
        "BF16/Qwen3-32B-BF16-00001-of-00002.gguf",
    )

    assert teacher_dataset.resolve_teacher_snapshot(teacher) == snapshot.resolve()
    assert calls == [
        {
            "repo_id": "unsloth/Qwen3-32B-GGUF",
            "revision": "model-revision",
            "allow_patterns": ["BF16/Qwen3-32B-BF16-*-of-00002.gguf"],
        }
    ]


def test_gguf_auto_detection_ignores_mmproj_and_selects_the_first_model_shard() -> None:
    class Api:
        @staticmethod
        def list_repo_files(_source: str, *, revision: str) -> list[str]:
            assert revision == "revision"
            return [
                "BF16/Qwen3.5-27B-BF16-00002-of-00002.gguf",
                "mmproj-BF16.gguf",
                "BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf",
            ]

    assert teacher_dataset.resolve_gguf_filename(
        "unsloth/Qwen3.5-27B-GGUF",
        "revision",
        None,
        api=Api(),  # type: ignore[arg-type]
    ) == "BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf"


class _Behavior:
    supported_modes = (ReasoningMode.THINKING, ReasoningMode.NON_THINKING)


class _TokenizerFactory:
    @staticmethod
    def from_pretrained(_snapshot: Path, *, local_files_only: bool) -> object:
        assert local_files_only is False
        return object()


def _fake_trace_preparer(
    _snapshot: Path,
    output: Path,
    item: Any,
    _tokenizer: object,
    _behavior: object,
    _records: object,
    **kwargs: object,
) -> PreparedTeacherTraces:
    assert kwargs["count"] == 2
    assert str(kwargs["source_adapter_identity"]).startswith("sha256:")
    assert kwargs["teacher_gguf_file"] == "Qwen3-8B-BF16.gguf"
    assert kwargs["teacher_tokenizer_source"] == "unsloth/Qwen3-8B"
    assert kwargs["teacher_tokenizer_revision"] == "tokenizer-revision"
    mode = item.mode
    values = []
    messages = []
    hashes = []
    for index in range(2):
        response = {"role": "assistant", "content": f"{mode.value} answer {index}"}
        if mode is ReasoningMode.THINKING:
            response["reasoning_content"] = f"reasoning {index}"
        turn = (
            {"role": "user", "content": f"prompt {index}"},
            response,
        )
        response_hash = f"sha256:{mode.value}-{index}"
        values.append(
            {
                "source_hash": f"sha256:source-{index}",
                "messages": turn,
                "prompt_token_hash": f"sha256:prompt-{index}",
                "response_token_hash": response_hash,
                "complete_token_hash": f"sha256:complete-{mode.value}-{index}",
                "prompt_tokens": 10,
                "response_tokens": 20,
                "stop_reason": "eos",
            }
        )
        messages.append(turn)
        hashes.append(response_hash)
    artifacts = LocalArtifactStore(Path(output) / "artifacts")
    with artifacts.begin_write("teacher-trace-dataset") as writer:
        (writer.path / "records.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    return PreparedTeacherTraces(
        tuple(messages),
        ArtifactRef("teacher-trace-dataset", descriptor.artifact_id, 1),
        f"identity-{mode.value}",
        tuple(hashes),
    )


def test_execution_publishes_huggingface_ready_mode_configs_and_reuses_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    settings_path = run / TEACHER_DATASET_SETTINGS_NAME
    write_teacher_dataset_settings(settings_path, _settings())
    monkeypatch.setattr(teacher_dataset, "AutoTokenizer", _TokenizerFactory)
    monkeypatch.setattr(teacher_dataset, "chat_behavior_for_snapshot", lambda _path: _Behavior())

    events: list[str] = []
    assert execute_teacher_dataset(
        settings_path,
        snapshot_resolver=lambda _teacher: tmp_path / "snapshot",
        tokenizer_snapshot_resolver=lambda _source, _revision: tmp_path / "tokenizer",
        source_loader=lambda _source, _seed: (),
        trace_preparer=_fake_trace_preparer,
        progress=lambda event, _fields: events.append(event),
    ) == 0

    dataset_root = run / TEACHER_DATASET_DIRECTORY_NAME
    manifest = json.loads(
        (dataset_root / TEACHER_DATASET_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["record_counts"] == {"thinking": 2, "non_thinking": 2}
    thinking = [
        json.loads(line)
        for line in (dataset_root / "data" / "thinking.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert thinking[0]["messages"][-1]["reasoning_content"] == "reasoning 0"
    assert thinking[0]["teacher_model"] == "unsloth/Qwen3-8B-GGUF"
    assert thinking[0]["teacher_gguf_file"] == "Qwen3-8B-BF16.gguf"
    assert thinking[0]["tokenizer_model"] == "unsloth/Qwen3-8B"
    card = (dataset_root / "README.md").read_text(encoding="utf-8")
    assert "config_name: all" in card
    assert "config_name: thinking" in card
    assert "config_name: non_thinking" in card
    assert (run / TEACHER_DATASET_COMPLETION_NAME).is_file()
    from datasets import load_dataset  # type: ignore[import-untyped]

    loaded = load_dataset(str(dataset_root), "thinking", split="train")
    assert len(loaded) == 2
    assert loaded[0]["messages"][-1]["reasoning_content"] == "reasoning 0"

    assert execute_teacher_dataset(
        settings_path,
        snapshot_resolver=lambda _source, _revision: (_ for _ in ()).throw(
            AssertionError("completed dataset should not reload the teacher")
        ),
        progress=lambda event, _fields: events.append(event),
    ) == 0
    assert "teacher_dataset_local_reused" in events


class _UploadApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_repo(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))
        return SimpleNamespace(repo_id=repo_id)

    def upload_folder(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("upload_folder", dict(kwargs)))
        return SimpleNamespace(oid="dataset-commit")


def test_completed_dataset_uploads_as_a_private_dataset_repository_and_reuses_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    settings = replace(
        _settings(),
        upload=TeacherDatasetUpload("owner/teacher-data", private=True),
    )
    settings_path = run / TEACHER_DATASET_SETTINGS_NAME
    write_teacher_dataset_settings(settings_path, settings)
    monkeypatch.setattr(teacher_dataset, "AutoTokenizer", _TokenizerFactory)
    monkeypatch.setattr(teacher_dataset, "chat_behavior_for_snapshot", lambda _path: _Behavior())
    api = _UploadApi()

    assert execute_teacher_dataset(
        settings_path,
        api=api,  # type: ignore[arg-type]
        snapshot_resolver=lambda _teacher: tmp_path / "snapshot",
        tokenizer_snapshot_resolver=lambda _source, _revision: tmp_path / "tokenizer",
        source_loader=lambda _source, _seed: (),
        trace_preparer=_fake_trace_preparer,
        progress=lambda _event, _fields: None,
    ) == 0

    assert api.calls[0] == (
        "create_repo",
        {
            "repo_id": "owner/teacher-data",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        },
    )
    assert api.calls[1][0] == "upload_folder"
    assert api.calls[1][1]["repo_type"] == "dataset"
    completion = json.loads((run / TEACHER_DATASET_COMPLETION_NAME).read_text(encoding="utf-8"))
    assert completion["upload"]["commit_oid"] == "dataset-commit"

    assert execute_teacher_dataset(
        settings_path,
        api=api,  # type: ignore[arg-type]
        progress=lambda _event, _fields: None,
    ) == 0
    assert len(api.calls) == 2


class _RevisionApi:
    @staticmethod
    def model_info(source: str) -> SimpleNamespace:
        revisions = {
            "unsloth/Qwen3-8B-GGUF": "resolved-teacher-revision",
            "unsloth/Qwen3-8B": "resolved-tokenizer-revision",
        }
        return SimpleNamespace(sha=revisions[source])

    @staticmethod
    def list_repo_files(source: str, *, revision: str) -> list[str]:
        assert source == "unsloth/Qwen3-8B-GGUF"
        assert revision == "resolved-teacher-revision"
        return ["Qwen3-8B-BF16.gguf"]


def test_interactive_menu_defaults_to_small_dual_mode_larger_teacher_and_persists_before_run(
    tmp_path: Path,
) -> None:
    catalog = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "teacher_dataset_models.yaml"
    )
    dispatched: list[Path] = []
    outputs: list[str] = []

    # UltraChat, Qwen3, default Unsloth Qwen3-8B GGUF, both modes, 512 rows, 2048 tokens,
    # llama.cpp, CUDA, no upload, start.
    result = run_interactive_teacher_dataset(
        tmp_path,
        catalog,
        input_fn=_inputs(["", "", "", "", "", "", "", "", "n", ""]),
        write=outputs.append,
        execute=lambda path: dispatched.append(path) or 0,
        api=_RevisionApi(),  # type: ignore[arg-type]
    )

    assert result == 0
    assert len(dispatched) == 1
    settings, _digest = load_teacher_dataset_settings(dispatched[0])
    assert settings.teacher.source == "unsloth/Qwen3-8B-GGUF"
    assert settings.teacher.revision == "resolved-teacher-revision"
    assert settings.teacher.tokenizer_source == "unsloth/Qwen3-8B"
    assert settings.teacher.tokenizer_revision == "resolved-tokenizer-revision"
    assert settings.teacher.gguf_filename == "Qwen3-8B-BF16.gguf"
    assert settings.generation.samples_per_mode == 512
    assert settings.generation.modes == (
        ReasoningMode.THINKING,
        ReasoningMode.NON_THINKING,
    )
    assert settings.upload is None
    assert dispatched[0].is_file()
    assert any(line.startswith("Settings written:") for line in outputs)

    resumed: list[Path] = []
    assert run_interactive_teacher_dataset(
        tmp_path,
        catalog,
        input_fn=_inputs([""]),
        write=lambda _line: None,
        execute=lambda path: resumed.append(path) or 0,
    ) == 0
    assert resumed == dispatched
