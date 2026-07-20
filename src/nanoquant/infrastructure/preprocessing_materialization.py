"""Validated local materialization of a retained resident preprocessing graph."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifact_gc import ARTIFACT_ID_PATTERN, TEXT_SUFFIXES
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.io_utils import atomic_write_json, safe_replace


@dataclass(frozen=True, slots=True)
class MaterializedResidentPreprocessing:
    calibration: ArtifactRef
    objectives: ArtifactRef
    plan: ArtifactRef
    artifact_count: int
    logical_bytes: int


def _artifact_references(store: LocalArtifactStore, artifact_id: str) -> set[str]:
    descriptor = store.validate(artifact_id)
    root = store.path_for(artifact_id)
    references: set[str] = set()
    for member in descriptor.files:
        path = root / member.path
        if path.suffix.lower() in TEXT_SUFFIXES:
            references.update(ARTIFACT_ID_PATTERN.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return references


def _validated_closure(store: LocalArtifactStore, roots: tuple[str, ...]) -> tuple[str, ...]:
    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        artifact_id = pending.pop()
        if artifact_id in reachable:
            continue
        store.validate(artifact_id)
        reachable.add(artifact_id)
        for reference in _artifact_references(store, artifact_id) - reachable:
            if not store.path_for(reference).is_dir():
                raise ValueError(f"retained preprocessing references absent artifact {reference}")
            pending.append(reference)
    return tuple(sorted(reachable))


def _copy_validated_artifact(
    source: LocalArtifactStore,
    destination: LocalArtifactStore,
    artifact_id: str,
) -> int:
    descriptor = source.validate(artifact_id)
    target = destination.path_for(artifact_id)
    if target.is_dir():
        destination.validate(artifact_id)
        return sum(member.bytes for member in descriptor.files)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="preprocessing-import-", dir=destination.temporary_root))
    try:
        shutil.copytree(source.path_for(artifact_id), staging, dirs_exist_ok=True)
        if target.exists():
            shutil.rmtree(staging)
        else:
            safe_replace(staging, target)
        destination.validate(artifact_id)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return sum(member.bytes for member in descriptor.files)


def materialize_resident_preprocessing(
    source_run: str | Path,
    destination_run: str | Path,
) -> MaterializedResidentPreprocessing:
    """Copy and freshly validate the transitive preprocessing graph into a run-local store."""

    source_root = Path(source_run).resolve()
    destination_root = Path(destination_run).resolve()
    try:
        state = json.loads((source_root / "state" / "preprocessing.json").read_text(encoding="utf-8"))
        calibration = from_dict(ArtifactRef, state["calibration"], path="preprocessing.calibration")
        objectives = from_dict(ArtifactRef, state["objectives"], path="preprocessing.objectives")
        plan = from_dict(ArtifactRef, state["plan"], path="preprocessing.plan")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("retained resident preprocessing state is invalid") from exc
    source_store = LocalArtifactStore(source_root / "artifacts", use_persistent_validation_cache=False)
    destination_store = LocalArtifactStore(
        destination_root / "artifacts",
        use_persistent_validation_cache=False,
    )
    closure = _validated_closure(
        source_store,
        (calibration.artifact_id, objectives.artifact_id, plan.artifact_id),
    )
    logical_bytes = sum(
        _copy_validated_artifact(source_store, destination_store, artifact_id) for artifact_id in closure
    )
    atomic_write_json(
        destination_root / "preprocessing-input.json",
        {
            "schema_version": 1,
            "materialized_from": str(source_root),
            "calibration": to_dict(calibration),
            "objectives": to_dict(objectives),
            "plan": to_dict(plan),
            "artifact_count": len(closure),
            "logical_bytes": logical_bytes,
        },
    )
    return MaterializedResidentPreprocessing(
        calibration,
        objectives,
        plan,
        len(closure),
        logical_bytes,
    )
