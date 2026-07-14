"""Conservative mark-and-sweep cleanup for local content-addressed artifacts."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from nanoquant.infrastructure.io_utils import safe_replace

ARTIFACT_ID_PATTERN = re.compile(r"sha256-[0-9a-f]{64}")
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"}


@dataclass(frozen=True, slots=True)
class ArtifactGarbageCollectionPlan:
    artifact_root: Path
    root_artifacts: tuple[str, ...]
    external_evidence_reference_count: int
    reachable_artifacts: tuple[str, ...]
    candidate_artifacts: tuple[str, ...]
    retained_for_age: tuple[str, ...]
    candidate_logical_bytes: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArtifactGarbageCollectionResult:
    deleted_artifacts: tuple[str, ...]
    deleted_logical_bytes: int


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _is_ignored(path: Path, ignored: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == item or _is_within(resolved, item) for item in ignored)


def _artifact_directories(root: Path) -> dict[str, Path]:
    result = {}
    for prefix in root.iterdir() if root.exists() else ():
        if not prefix.is_dir() or not re.fullmatch(r"[0-9a-f]{2}", prefix.name):
            continue
        for candidate in prefix.iterdir():
            if not candidate.is_dir() or not ARTIFACT_ID_PATTERN.fullmatch(candidate.name):
                continue
            if candidate.name[7:9] != prefix.name:
                continue
            result[candidate.name] = candidate
    return result


def _text_artifact_ids(path: Path) -> set[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return set()
    try:
        return set(ARTIFACT_ID_PATTERN.findall(path.read_text(encoding="utf-8", errors="ignore")))
    except OSError:
        return set()


def _evidence_roots(evidence_roots: tuple[Path, ...], ignored: tuple[Path, ...]) -> set[str]:
    result = set()
    for evidence_root in evidence_roots:
        if not evidence_root.exists() or _is_ignored(evidence_root, ignored):
            continue
        if evidence_root.is_file():
            result.update(_text_artifact_ids(evidence_root))
            continue
        for directory, names, filenames in os.walk(evidence_root, followlinks=False):
            current = Path(directory)
            names[:] = [
                name
                for name in names
                if name not in {".git", ".tmp", "artifacts"}
                and not _is_ignored(current / name, ignored)
            ]
            if _is_ignored(current, ignored):
                names[:] = []
                continue
            for filename in filenames:
                path = current / filename
                if not _is_ignored(path, ignored):
                    result.update(_text_artifact_ids(path))
    return result


def _artifact_references(path: Path) -> set[str]:
    result = set()
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.name != "descriptor.json":
            result.update(_text_artifact_ids(candidate))
    return result


def _logical_bytes(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def plan_artifact_gc(
    artifact_root: str | Path,
    evidence_roots: tuple[str | Path, ...],
    *,
    ignored_evidence_paths: tuple[str | Path, ...] = (),
    keep_artifacts: tuple[str, ...] = (),
    minimum_age_seconds: float = 24 * 60 * 60,
    now: float | None = None,
) -> ArtifactGarbageCollectionPlan:
    root = Path(artifact_root).resolve()
    if minimum_age_seconds < 0:
        raise ValueError("artifact GC minimum age cannot be negative")
    ignored = tuple(Path(path).resolve() for path in ignored_evidence_paths)
    inventory = _artifact_directories(root)
    observed_roots = _evidence_roots(
        tuple(Path(path).resolve() for path in evidence_roots), ignored
    )
    roots = observed_roots & inventory.keys()
    external_evidence_reference_count = len(observed_roots - inventory.keys())
    for artifact_id in keep_artifacts:
        if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
            raise ValueError(f"invalid explicitly retained artifact ID: {artifact_id}")
        if artifact_id not in inventory:
            raise ValueError(f"explicitly retained artifact is absent: {artifact_id}")
        roots.add(artifact_id)
    reachable = set()
    warnings = []
    pending = list(roots)
    while pending:
        artifact_id = pending.pop()
        if artifact_id in reachable:
            continue
        path = inventory[artifact_id]
        reachable.add(artifact_id)
        references = _artifact_references(path) - reachable
        for reference in references:
            if reference in inventory:
                pending.append(reference)
            else:
                warnings.append(
                    f"reachable artifact {artifact_id} references absent artifact: {reference}"
                )
    cutoff = (time.time() if now is None else now) - minimum_age_seconds
    candidates = []
    retained_for_age = []
    candidate_bytes = 0
    for artifact_id, path in inventory.items():
        if artifact_id in reachable:
            continue
        try:
            modified = max(candidate.stat().st_mtime for candidate in path.rglob("*") if candidate.is_file())
        except (OSError, ValueError):
            modified = path.stat().st_mtime
        if modified > cutoff:
            retained_for_age.append(artifact_id)
            continue
        candidates.append(artifact_id)
        candidate_bytes += _logical_bytes(path)
    return ArtifactGarbageCollectionPlan(
        root,
        tuple(sorted(roots)),
        external_evidence_reference_count,
        tuple(sorted(reachable)),
        tuple(sorted(candidates)),
        tuple(sorted(retained_for_age)),
        candidate_bytes,
        tuple(sorted(set(warnings))),
    )


def _delete_artifact_directory(root: Path, artifact_id: str) -> int:
    if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
        raise ValueError(f"invalid artifact deletion candidate: {artifact_id}")
    path = root / artifact_id[7:9] / artifact_id
    if path.resolve().parent != (root / artifact_id[7:9]).resolve() or not _is_within(path, root):
        raise ValueError("artifact deletion candidate escaped the artifact root")
    logical_bytes = _logical_bytes(path)
    for candidate in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink()
        elif candidate.is_dir():
            candidate.rmdir()
    path.rmdir()
    prefix = path.parent
    try:
        prefix.rmdir()
    except OSError:
        pass
    return logical_bytes


def apply_artifact_gc(plan: ArtifactGarbageCollectionPlan) -> ArtifactGarbageCollectionResult:
    deleted = []
    deleted_bytes = 0
    root = plan.artifact_root.resolve()
    for artifact_id in plan.candidate_artifacts:
        path = root / artifact_id[7:9] / artifact_id
        if not path.exists():
            continue
        deleted_bytes += _delete_artifact_directory(root, artifact_id)
        deleted.append(artifact_id)
    cache_path = root / ".validation-cache.json"
    if cache_path.exists() and deleted:
        import json

        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cache, dict):
                for artifact_id in deleted:
                    cache.pop(artifact_id, None)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
                safe_replace(temporary, cache_path)
        except (OSError, json.JSONDecodeError):
            pass
    return ArtifactGarbageCollectionResult(tuple(deleted), deleted_bytes)
