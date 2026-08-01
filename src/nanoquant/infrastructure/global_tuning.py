"""Immutable global-tuning result persistence and active-pointer management."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef, GlobalTuningResult

from .artifacts import LocalArtifactStore
from .io_utils import safe_replace


def _write_pointer(path: Path, reference: ArtifactRef, *, temporary_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=temporary_prefix, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(to_dict(reference), stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        safe_replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _stage_pointer_path(run_output: str | Path, state_namespace: str) -> Path:
    if not state_namespace or Path(state_namespace).name != state_namespace or state_namespace in {".", ".."}:
        raise ValueError("global-tuning state namespace must be a safe filename stem")
    return Path(run_output) / f"{state_namespace}-result.json"


@dataclass(frozen=True, slots=True)
class CommittedGlobalTuning:
    reference: ArtifactRef
    result: GlobalTuningResult


def commit_global_tuning(result: GlobalTuningResult, artifacts: LocalArtifactStore) -> CommittedGlobalTuning:
    with artifacts.recorder.phase("serialize"):
        encoded = json.dumps(to_dict(result), sort_keys=True, indent=2)
    with artifacts.begin_write("global-tuning-result") as writer:
        with artifacts.recorder.phase("write"):
            (writer.path / "global-tuning-result.json").write_text(encoded, encoding="utf-8")
        descriptor = writer.commit()
    return CommittedGlobalTuning(
        ArtifactRef("global-tuning-result", descriptor.artifact_id, descriptor.schema_version),
        result,
    )


def load_global_tuning(reference: ArtifactRef, artifacts: LocalArtifactStore) -> CommittedGlobalTuning:
    descriptor = artifacts.validate(reference.artifact_id)
    if descriptor.artifact_type != "global-tuning-result":
        raise ValueError("artifact is not a global tuning result")
    payload = json.loads(
        (artifacts.path_for(reference.artifact_id) / "global-tuning-result.json").read_text(encoding="utf-8")
    )
    return CommittedGlobalTuning(
        reference,
        from_dict(GlobalTuningResult, payload, path="global_tuning"),
    )


def activate_global_tuning(run_output: str | Path, reference: ArtifactRef) -> None:
    _write_pointer(
        Path(run_output) / "global-tuning.json",
        reference,
        temporary_prefix="global-tuning-",
    )


def activate_global_tuning_stage(
    run_output: str | Path,
    reference: ArtifactRef,
    *,
    state_namespace: str,
) -> None:
    _write_pointer(
        _stage_pointer_path(run_output, state_namespace),
        reference,
        temporary_prefix="global-tuning-stage-",
    )


def active_global_tuning(run_output: str | Path) -> ArtifactRef | None:
    path = Path(run_output) / "global-tuning.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return from_dict(ArtifactRef, payload, path="global_tuning_reference")


def active_global_tuning_stage(
    run_output: str | Path,
    *,
    state_namespace: str,
) -> ArtifactRef | None:
    path = _stage_pointer_path(run_output, state_namespace)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return from_dict(ArtifactRef, payload, path="global_tuning_stage_reference")
