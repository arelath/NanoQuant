from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import RepoUrl

import nanoquant.infrastructure.huggingface_upload as upload_module
from nanoquant.infrastructure.huggingface_upload import (
    HUGGINGFACE_UPLOAD_SCHEMA_VERSION,
    HuggingFaceUploadConfig,
    ValidatedModelArtifact,
    ensure_huggingface_model_repository,
    upload_validated_model_artifacts,
)


class _FakeHfApi:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.uploaded: dict[str, bytes] = {}

    def create_repo(self, repo_id: str, **kwargs: object) -> RepoUrl:
        self.calls.append(("create_repo", repo_id, kwargs))
        return RepoUrl(f"https://huggingface.co/resolved-owner/{repo_id.split('/')[-1]}")

    def create_commit(self, repo_id: str, operations, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        self.calls.append(("create_commit", repo_id, kwargs))
        for operation in operations:
            handle = operation.path_or_fileobj
            self.uploaded[operation.path_in_repo] = handle.read()
            handle.seek(0)
        return SimpleNamespace(
            oid="a" * 40,
            commit_url=f"https://huggingface.co/{repo_id}/commit/{'a' * 40}",
        )


def _artifact(path: Path, *, path_in_repo: str = "") -> ValidatedModelArtifact:
    payload = path.read_bytes()
    return ValidatedModelArtifact(
        path,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        path_in_repo,
    )


def test_huggingface_upload_commits_exact_validated_model_files_and_writes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.gguf"
    projector = tmp_path / "mmproj-BF16.gguf"
    model.write_bytes(b"language-model")
    projector.write_bytes(b"projector")
    receipt = tmp_path / "model.gguf.huggingface.json"
    api = _FakeHfApi()
    progress: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setenv("HF_TOKEN", "must-not-appear")
    monkeypatch.setattr(upload_module, "HUGGINGFACE_VALIDATION_PROGRESS_BYTES", 4)

    result = upload_validated_model_artifacts(
        HuggingFaceUploadConfig("requested-owner/model", private=True),
        (_artifact(model), _artifact(projector)),
        receipt_output=receipt,
        api=api,  # type: ignore[arg-type]
        progress=lambda severity, name, fields: progress.append((severity, name, fields)),
    )

    assert result.repo_id == "resolved-owner/model"
    assert result.commit_oid == "a" * 40
    assert api.uploaded == {
        "model.gguf": b"language-model",
        "mmproj-BF16.gguf": b"projector",
    }
    assert api.calls == [
        (
            "create_repo",
            "requested-owner/model",
            {"repo_type": "model", "private": True, "exist_ok": True},
        ),
        (
            "create_commit",
            "resolved-owner/model",
            {"repo_type": "model", "commit_message": "Upload validated NanoQuant GGUF"},
        ),
    ]
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema_version"] == HUGGINGFACE_UPLOAD_SCHEMA_VERSION
    assert payload["repo_id"] == "resolved-owner/model"
    assert [artifact["path_in_repo"] for artifact in payload["artifacts"]] == [
        "model.gguf",
        "mmproj-BF16.gguf",
    ]
    assert "must-not-appear" not in receipt.read_text(encoding="utf-8")
    names = [name for _severity, name, _fields in progress]
    assert names == [
        "huggingface.upload.started",
        "huggingface.artifact_validation.started",
        "huggingface.artifact_validation.progress",
        "huggingface.artifact_validation.completed",
        "huggingface.artifact_validation.started",
        "huggingface.artifact_validation.progress",
        "huggingface.artifact_validation.completed",
        "huggingface.repository.started",
        "huggingface.repository.completed",
        "huggingface.commit.started",
        "huggingface.commit.completed",
        "huggingface.receipt.completed",
        "huggingface.upload.completed",
    ]
    validation_progress = [
        fields
        for _severity, name, fields in progress
        if name == "huggingface.artifact_validation.progress"
    ]
    assert validation_progress[0]["processed_bytes"] == len(b"language-model")
    assert validation_progress[0]["percent"] == 100.0
    assert all("token" not in json.dumps(fields).lower() for _severity, _name, fields in progress)


def test_huggingface_upload_fails_before_hub_mutation_when_validated_hash_changed(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"changed")
    artifact = ValidatedModelArtifact(model, len(b"changed"), "0" * 64)
    api = _FakeHfApi()

    with pytest.raises(ValueError, match="SHA-256 changed"):
        upload_validated_model_artifacts(
            HuggingFaceUploadConfig("owner/model"),
            (artifact,),
            receipt_output=tmp_path / "receipt.json",
            api=api,  # type: ignore[arg-type]
        )

    assert api.calls == []
    assert not (tmp_path / "receipt.json").exists()


def test_huggingface_upload_reports_heartbeat_during_long_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    api = _FakeHfApi()
    original_create_commit = api.create_commit

    def slow_create_commit(*args, **kwargs):  # type: ignore[no-untyped-def]
        time.sleep(0.03)
        return original_create_commit(*args, **kwargs)

    monkeypatch.setattr(api, "create_commit", slow_create_commit)
    monkeypatch.setattr(upload_module, "HUGGINGFACE_COMMIT_HEARTBEAT_SECONDS", 0.005)
    progress: list[str] = []

    upload_validated_model_artifacts(
        HuggingFaceUploadConfig("owner/model"),
        (_artifact(model),),
        receipt_output=tmp_path / "receipt.json",
        api=api,  # type: ignore[arg-type]
        progress=lambda _severity, name, _fields: progress.append(name),
    )

    assert "huggingface.commit.progress" in progress
    assert not any(thread.name == "nanoquant-huggingface-upload-progress" for thread in threading.enumerate())


def test_huggingface_upload_configuration_and_paths_fail_closed(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    with pytest.raises(ValueError, match="repository ID"):
        HuggingFaceUploadConfig("https://huggingface.co/owner/model")
    with pytest.raises(ValueError, match="one line"):
        HuggingFaceUploadConfig("owner/model", commit_message="first\nsecond")
    with pytest.raises(ValueError, match="unsafe"):
        _artifact(model, path_in_repo="../model.gguf")
    with pytest.raises(ValueError, match="must not overwrite"):
        upload_validated_model_artifacts(
            HuggingFaceUploadConfig("owner/model"),
            (_artifact(model),),
            receipt_output=model,
            api=_FakeHfApi(),  # type: ignore[arg-type]
        )


def test_huggingface_preflight_requires_exact_token_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload_module.os, "environ", {"HF_Token": "wrong-case"})

    with pytest.raises(RuntimeError, match="use HF_TOKEN, not HF_Token"):
        ensure_huggingface_model_repository(HuggingFaceUploadConfig("owner/model"))
