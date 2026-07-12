"""Local immutable content-addressed artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from nanoquant.ports.artifact_store import ArtifactDescriptor, ArtifactFile


class ArtifactCorruptionError(IOError):
    pass


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalArtifactWriter:
    def __init__(self, store: LocalArtifactStore, artifact_type: str, schema_version: int) -> None:
        self.store = store
        self.artifact_type = artifact_type
        self.schema_version = schema_version
        self.path = Path(tempfile.mkdtemp(prefix="lease-", dir=store.temporary_root))
        (self.path / ".lease.json").write_text(
            json.dumps({"pid": os.getpid(), "created_at": time.time()}), encoding="utf-8"
        )
        self._committed = False

    def commit(self) -> ArtifactDescriptor:
        if self._committed:
            raise RuntimeError("writer was already committed")
        files: list[ArtifactFile] = []
        for path in sorted(
            candidate for candidate in self.path.rglob("*") if candidate.is_file() and candidate.name != ".lease.json"
        ):
            relative = path.relative_to(self.path).as_posix()
            if relative.startswith("../") or Path(relative).is_absolute():
                raise ValueError("artifact path traversal is not allowed")
            files.append(ArtifactFile(relative, path.stat().st_size, _hash_file(path)))
        identity_payload = json.dumps(
            {"type": self.artifact_type, "schema": self.schema_version, "files": [asdict(item) for item in files]},
            sort_keys=True,
            separators=(",", ":"),
        )
        content_hash = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        artifact_id = f"sha256-{content_hash}"
        descriptor = ArtifactDescriptor(
            self.schema_version, self.artifact_type, artifact_id, f"sha256:{content_hash}", tuple(files), time.time()
        )
        (self.path / ".lease.json").unlink()
        (self.path / "descriptor.json").write_text(
            json.dumps(asdict(descriptor), sort_keys=True, indent=2), encoding="utf-8"
        )
        destination = self.store.root / content_hash[:2] / artifact_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(self.path)
        else:
            os.replace(self.path, destination)
        self.store._remember_validation(descriptor, destination)
        self._committed = True
        return descriptor

    def __enter__(self) -> LocalArtifactWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._committed and self.path.exists():
            shutil.rmtree(self.path)


class LocalArtifactStore:
    def __init__(self, root: str | Path, temporary_root: str | Path | None = None) -> None:
        self.root = Path(root)
        self.temporary_root = Path(temporary_root) if temporary_root else self.root / ".tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        self._validated: dict[str, tuple[ArtifactDescriptor, tuple[tuple[str, int, int], ...]]] = {}
        self._validation_cache_path = self.root / ".validation-cache.json"
        try:
            cached = json.loads(self._validation_cache_path.read_text(encoding="utf-8"))
            self._persistent_validation = cached if isinstance(cached, dict) else {}
        except (OSError, json.JSONDecodeError):
            self._persistent_validation = {}

    def _persist_validation(self) -> None:
        temporary = self._validation_cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._persistent_validation, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self._validation_cache_path)

    def _remember_validation(self, descriptor: ArtifactDescriptor, root: Path) -> None:
        signatures = tuple(
            (item.path, (root / item.path).stat().st_size, (root / item.path).stat().st_mtime_ns)
            for item in descriptor.files
        )
        descriptor_stat = (root / "descriptor.json").stat()
        persistent_signature = {
            "descriptor": [descriptor_stat.st_size, descriptor_stat.st_mtime_ns],
            "members": [list(signature) for signature in signatures],
        }
        self._validated[descriptor.artifact_id] = (descriptor, signatures)
        self._persistent_validation[descriptor.artifact_id] = persistent_signature
        self._persist_validation()

    def begin_write(self, artifact_type: str, schema_version: int = 1) -> LocalArtifactWriter:
        if not artifact_type or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in artifact_type
        ):
            raise ValueError("artifact type must be lowercase alphanumeric with '-' or '_'")
        return LocalArtifactWriter(self, artifact_type, schema_version)

    def path_for(self, artifact_id: str) -> Path:
        if not artifact_id.startswith("sha256-") or len(artifact_id) != 71:
            raise ValueError("invalid artifact id")
        return self.root / artifact_id[7:9] / artifact_id

    def validate(self, artifact_id: str) -> ArtifactDescriptor:
        root = self.path_for(artifact_id)
        cached = self._validated.get(artifact_id)
        if cached is not None:
            descriptor, signatures = cached
            current = tuple(
                (
                    item.path,
                    (root / item.path).stat().st_size,
                    (root / item.path).stat().st_mtime_ns,
                )
                for item in descriptor.files
                if (root / item.path).is_file()
            )
            if current == signatures:
                return descriptor
        try:
            raw = json.loads((root / "descriptor.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactCorruptionError(f"ART001 descriptor unavailable for {artifact_id}") from exc
        files = tuple(ArtifactFile(**item) for item in raw["files"])
        descriptor = ArtifactDescriptor(
            raw["schema_version"],
            raw["artifact_type"],
            raw["artifact_id"],
            raw["content_hash"],
            files,
            raw["committed_at"],
        )
        if descriptor.artifact_id != artifact_id:
            raise ArtifactCorruptionError("ART001 descriptor identity mismatch")
        identity_payload = json.dumps(
            {
                "type": descriptor.artifact_type,
                "schema": descriptor.schema_version,
                "files": [asdict(item) for item in descriptor.files],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_hash = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        if descriptor.content_hash != f"sha256:{expected_hash}" or artifact_id != f"sha256-{expected_hash}":
            raise ArtifactCorruptionError("ART001 descriptor content identity mismatch")
        signatures = tuple(
            (item.path, (root / item.path).stat().st_size, (root / item.path).stat().st_mtime_ns)
            for item in files
            if (root / item.path).is_file()
        )
        descriptor_stat = (root / "descriptor.json").stat()
        persistent_signature = {
            "descriptor": [descriptor_stat.st_size, descriptor_stat.st_mtime_ns],
            "members": [list(signature) for signature in signatures],
        }
        if self._persistent_validation.get(artifact_id) == persistent_signature and len(signatures) == len(files):
            self._validated[artifact_id] = (descriptor, signatures)
            return descriptor
        for item in files:
            path = (root / item.path).resolve()
            if root.resolve() not in path.parents:
                raise ArtifactCorruptionError("ART001 descriptor path traversal")
            if not path.is_file() or path.stat().st_size != item.bytes or _hash_file(path) != item.sha256:
                raise ArtifactCorruptionError(f"ART001 corrupt artifact member: {item.path}")
        self._remember_validation(descriptor, root)
        return descriptor

    def cleanup_abandoned(self, older_than_seconds: float = 3600) -> int:
        removed = 0
        cutoff = time.time() - older_than_seconds
        for lease in self.temporary_root.glob("lease-*/.lease.json"):
            try:
                created = float(json.loads(lease.read_text(encoding="utf-8"))["created_at"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                created = 0
            if created < cutoff:
                shutil.rmtree(lease.parent, ignore_errors=True)
                removed += 1
        return removed
