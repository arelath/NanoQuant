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


@dataclass(frozen=True, slots=True)
class CommittedGlobalTuning:
    reference: ArtifactRef
    result: GlobalTuningResult


def commit_global_tuning(result: GlobalTuningResult, artifacts: LocalArtifactStore) -> CommittedGlobalTuning:
    with artifacts.begin_write("global-tuning-result") as writer:
        (writer.path / "global-tuning-result.json").write_text(
            json.dumps(to_dict(result), sort_keys=True, indent=2),
            encoding="utf-8",
        )
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
    output = Path(run_output)
    output.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="global-tuning-", suffix=".tmp", dir=output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(to_dict(reference), stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "global-tuning.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def active_global_tuning(run_output: str | Path) -> ArtifactRef | None:
    path = Path(run_output) / "global-tuning.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return from_dict(ArtifactRef, payload, path="global_tuning_reference")
