"""Hugging Face publication of already validated deployment model files."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError, HFValidationError
from huggingface_hub.utils import validate_repo_id  # type: ignore[attr-defined]

from nanoquant.infrastructure.io_utils import atomic_write_json

HUGGINGFACE_UPLOAD_SCHEMA_VERSION = 1
HUGGINGFACE_VALIDATION_PROGRESS_BYTES = 256 * 1024 * 1024
HUGGINGFACE_COMMIT_HEARTBEAT_SECONDS = 30.0
HuggingFaceProgress = Callable[[str, str, dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class HuggingFaceUploadConfig:
    """Non-secret destination settings for one model-repository upload."""

    repo_id: str
    private: bool | None = None
    commit_message: str = "Upload validated NanoQuant GGUF"

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or self.repo_id != self.repo_id.strip():
            raise ValueError("Hugging Face repository ID must be a trimmed string")
        try:
            validate_repo_id(self.repo_id)
        except HFValidationError as error:
            raise ValueError(f"invalid Hugging Face repository ID: {self.repo_id!r}") from error
        if self.private is not None and not isinstance(self.private, bool):
            raise ValueError("Hugging Face private setting must be true, false, or None")
        if not isinstance(self.commit_message, str) or not self.commit_message.strip():
            raise ValueError("Hugging Face commit message is required")
        if "\n" in self.commit_message or "\r" in self.commit_message:
            raise ValueError("Hugging Face commit message must be one line")
        object.__setattr__(self, "commit_message", self.commit_message.strip())


@dataclass(frozen=True, slots=True)
class ValidatedModelArtifact:
    """A local model file bound to the size and digest established by export."""

    source: Path
    bytes: int
    sha256: str
    path_in_repo: str = ""

    def __post_init__(self) -> None:
        source = Path(self.source)
        if not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes <= 0:
            raise ValueError("validated model artifact byte count must be positive")
        if not isinstance(self.sha256, str):
            raise ValueError("validated model artifact SHA-256 must be a string")
        digest = self.sha256.strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("validated model artifact SHA-256 must be 64 hexadecimal characters")
        if not isinstance(self.path_in_repo, str):
            raise ValueError("Hugging Face repository path must be a string")
        path_in_repo = self.path_in_repo or source.name
        segments = path_in_repo.split("/")
        if (
            not path_in_repo
            or path_in_repo.startswith("/")
            or path_in_repo.endswith("/")
            or "\\" in path_in_repo
            or any(segment in {"", ".", ".."} for segment in segments)
        ):
            raise ValueError(f"Hugging Face repository path is unsafe: {path_in_repo!r}")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "path_in_repo", path_in_repo)


@dataclass(frozen=True, slots=True)
class UploadedModelArtifact:
    source: Path
    path_in_repo: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class HuggingFaceUploadResult:
    repo_id: str
    repo_url: str
    commit_oid: str
    commit_url: str
    requested_private: bool | None
    commit_message: str
    artifacts: tuple[UploadedModelArtifact, ...]
    receipt_output: Path


def huggingface_upload_summary(result: HuggingFaceUploadResult) -> dict[str, object]:
    """Return the token-free JSON representation shared by receipts and reports."""

    return {
        "repo_id": result.repo_id,
        "repo_url": result.repo_url,
        "commit_oid": result.commit_oid,
        "commit_url": result.commit_url,
        "requested_private": result.requested_private,
        "commit_message": result.commit_message,
        "receipt": str(result.receipt_output),
        "artifacts": [
            {
                "source": str(artifact.source),
                "path_in_repo": artifact.path_in_repo,
                "bytes": artifact.bytes,
                "sha256": artifact.sha256,
            }
            for artifact in result.artifacts
        ],
    }


def _notify(
    progress: HuggingFaceProgress | None,
    severity: str,
    name: str,
    **fields: object,
) -> None:
    if progress is not None:
        progress(severity, name, fields)


@contextmanager
def _commit_heartbeat(
    progress: HuggingFaceProgress | None,
    *,
    repo_id: str,
    artifact_count: int,
    total_bytes: int,
) -> Iterator[None]:
    if progress is None:
        yield
        return
    stopped = threading.Event()
    started = time.perf_counter()

    def report() -> None:
        while not stopped.wait(HUGGINGFACE_COMMIT_HEARTBEAT_SECONDS):
            _notify(
                progress,
                "info",
                "huggingface.commit.progress",
                repo_id=repo_id,
                artifact_count=artifact_count,
                total_bytes=total_bytes,
                elapsed_seconds=time.perf_counter() - started,
            )

    worker = threading.Thread(
        target=report,
        name="nanoquant-huggingface-upload-progress",
        daemon=True,
    )
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def _measure_open_file(
    source: BinaryIO,
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    next_progress = HUGGINGFACE_VALIDATION_PROGRESS_BYTES
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        byte_count += len(chunk)
        digest.update(chunk)
        if progress is not None and byte_count >= next_progress:
            progress(byte_count)
            while next_progress <= byte_count:
                next_progress += HUGGINGFACE_VALIDATION_PROGRESS_BYTES
    return byte_count, digest.hexdigest()


def _authenticated_api(api: HfApi | None) -> HfApi:
    if api is not None:
        return api
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        suffix = (
            " Environment variable names are case-sensitive; use HF_TOKEN, not HF_Token."
            if os.environ.get("HF_Token")
            else ""
        )
        raise RuntimeError(f"Hugging Face publication requires the HF_TOKEN environment variable.{suffix}")
    return HfApi(token=token)


def ensure_huggingface_model_repository(
    config: HuggingFaceUploadConfig,
    *,
    api: HfApi | None = None,
    progress: HuggingFaceProgress | None = None,
) -> str:
    """Verify explicit-token write access and create the model repository if needed."""

    client = _authenticated_api(api)
    started = time.perf_counter()
    _notify(
        progress,
        "info",
        "huggingface.repository.started",
        repo_id=config.repo_id,
        requested_private=config.private,
    )
    try:
        repo_url = client.create_repo(
            config.repo_id,
            repo_type="model",
            private=config.private,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        _notify(
            progress,
            "error",
            "huggingface.repository.failed",
            repo_id=config.repo_id,
            error_type=type(exc).__name__,
            wall_seconds=time.perf_counter() - started,
        )
        raise RuntimeError(
            f"HF_TOKEN cannot create or access model repository {config.repo_id!r}; "
            "use a token with write permission for the destination namespace"
        ) from exc
    _notify(
        progress,
        "info",
        "huggingface.repository.completed",
        requested_repo_id=config.repo_id,
        repo_id=repo_url.repo_id,
        wall_seconds=time.perf_counter() - started,
    )
    return repo_url.repo_id


def upload_validated_model_artifacts(
    config: HuggingFaceUploadConfig,
    artifacts: Iterable[ValidatedModelArtifact],
    *,
    receipt_output: str | Path,
    api: HfApi | None = None,
    progress: HuggingFaceProgress | None = None,
) -> HuggingFaceUploadResult:
    """Upload the exact validated file handles in one model-repository commit."""

    requested = tuple(artifacts)
    if not requested:
        raise ValueError("at least one validated model artifact is required")
    repository_paths = tuple(artifact.path_in_repo for artifact in requested)
    if len(set(repository_paths)) != len(repository_paths):
        raise ValueError("Hugging Face repository paths must be unique")

    receipt = Path(receipt_output).resolve()
    if receipt.exists() and not receipt.is_file():
        raise ValueError("Hugging Face receipt output must be a regular file")
    client = _authenticated_api(api)
    total_bytes = sum(artifact.bytes for artifact in requested)
    upload_started = time.perf_counter()
    _notify(
        progress,
        "info",
        "huggingface.upload.started",
        repo_id=config.repo_id,
        artifact_count=len(requested),
        total_bytes=total_bytes,
        receipt=str(receipt),
    )
    try:
        with ExitStack() as opened:
            operations: list[CommitOperationAdd] = []
            uploaded: list[UploadedModelArtifact] = []
            for index, artifact in enumerate(requested, start=1):
                source = artifact.source.resolve(strict=True)
                if not source.is_file():
                    raise ValueError(f"validated model artifact is not a regular file: {source}")
                if source == receipt:
                    raise ValueError("Hugging Face receipt must not overwrite a model artifact")
                validation_started = time.perf_counter()
                _notify(
                    progress,
                    "info",
                    "huggingface.artifact_validation.started",
                    artifact_index=index,
                    artifact_count=len(requested),
                    path_in_repo=artifact.path_in_repo,
                    expected_bytes=artifact.bytes,
                )
                handle = opened.enter_context(source.open("rb"))

                def report_validation(
                    processed_bytes: int,
                    *,
                    item: ValidatedModelArtifact = artifact,
                    item_index: int = index,
                ) -> None:
                    _notify(
                        progress,
                        "info",
                        "huggingface.artifact_validation.progress",
                        artifact_index=item_index,
                        artifact_count=len(requested),
                        path_in_repo=item.path_in_repo,
                        processed_bytes=min(processed_bytes, item.bytes),
                        expected_bytes=item.bytes,
                        percent=min(100.0, processed_bytes * 100.0 / item.bytes),
                    )

                observed_bytes, observed_sha256 = _measure_open_file(
                    handle,
                    progress=report_validation,
                )
                if observed_bytes != artifact.bytes:
                    raise ValueError(
                        f"validated model artifact byte count changed before upload: "
                        f"{observed_bytes} != {artifact.bytes}"
                    )
                if observed_sha256 != artifact.sha256:
                    raise ValueError("validated model artifact SHA-256 changed before upload")
                handle.seek(0)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=artifact.path_in_repo,
                        path_or_fileobj=handle,
                    )
                )
                uploaded.append(
                    UploadedModelArtifact(
                        source,
                        artifact.path_in_repo,
                        artifact.bytes,
                        artifact.sha256,
                    )
                )
                _notify(
                    progress,
                    "info",
                    "huggingface.artifact_validation.completed",
                    artifact_index=index,
                    artifact_count=len(requested),
                    path_in_repo=artifact.path_in_repo,
                    bytes=observed_bytes,
                    wall_seconds=time.perf_counter() - validation_started,
                )

            repo_id = ensure_huggingface_model_repository(config, api=client, progress=progress)
            commit_started = time.perf_counter()
            _notify(
                progress,
                "info",
                "huggingface.commit.started",
                repo_id=repo_id,
                artifact_count=len(operations),
                total_bytes=total_bytes,
                commit_message=config.commit_message,
            )
            try:
                with _commit_heartbeat(
                    progress,
                    repo_id=repo_id,
                    artifact_count=len(operations),
                    total_bytes=total_bytes,
                ):
                    commit = client.create_commit(
                        repo_id,
                        operations,
                        repo_type="model",
                        commit_message=config.commit_message,
                    )
            except BaseException as exc:
                _notify(
                    progress,
                    "error",
                    "huggingface.commit.failed",
                    repo_id=repo_id,
                    error_type=type(exc).__name__,
                    wall_seconds=time.perf_counter() - commit_started,
                )
                raise
            _notify(
                progress,
                "info",
                "huggingface.commit.completed",
                repo_id=repo_id,
                commit_oid=commit.oid,
                commit_url=commit.commit_url,
                wall_seconds=time.perf_counter() - commit_started,
            )
    except BaseException as exc:
        _notify(
            progress,
            "error",
            "huggingface.upload.failed",
            repo_id=config.repo_id,
            error_type=type(exc).__name__,
            wall_seconds=time.perf_counter() - upload_started,
        )
        raise

    result = HuggingFaceUploadResult(
        repo_id,
        f"https://huggingface.co/{repo_id}",
        commit.oid,
        commit.commit_url,
        config.private,
        config.commit_message,
        tuple(uploaded),
        receipt,
    )
    atomic_write_json(
        receipt,
        {
            "schema_version": HUGGINGFACE_UPLOAD_SCHEMA_VERSION,
            **huggingface_upload_summary(result),
        },
    )
    _notify(
        progress,
        "info",
        "huggingface.receipt.completed",
        repo_id=repo_id,
        receipt=str(receipt),
        artifact_count=len(uploaded),
    )
    _notify(
        progress,
        "info",
        "huggingface.upload.completed",
        repo_id=repo_id,
        commit_oid=commit.oid,
        artifact_count=len(uploaded),
        total_bytes=total_bytes,
        wall_seconds=time.perf_counter() - upload_started,
    )
    return result


__all__ = [
    "HUGGINGFACE_UPLOAD_SCHEMA_VERSION",
    "HUGGINGFACE_COMMIT_HEARTBEAT_SECONDS",
    "HUGGINGFACE_VALIDATION_PROGRESS_BYTES",
    "HuggingFaceUploadConfig",
    "HuggingFaceProgress",
    "HuggingFaceUploadResult",
    "UploadedModelArtifact",
    "ValidatedModelArtifact",
    "ensure_huggingface_model_repository",
    "huggingface_upload_summary",
    "upload_validated_model_artifacts",
]
