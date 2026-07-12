from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    schema_version: int
    artifact_type: str
    artifact_id: str
    content_hash: str
    files: tuple[ArtifactFile, ...]
    committed_at: float


class ArtifactWriter(AbstractContextManager["ArtifactWriter"], Protocol):
    path: Path

    def commit(self) -> ArtifactDescriptor: ...


class ArtifactStore(Protocol):
    def begin_write(self, artifact_type: str, schema_version: int = 1) -> ArtifactWriter: ...
    def validate(self, artifact_id: str) -> ArtifactDescriptor: ...
