"""Portable measured rank responses and optional KL allocation weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoquant.application.kl_budget import KlBudgetProfile, load_kl_budget_profile, validate_kl_budget_profile
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.io_utils import atomic_write_json


@dataclass(frozen=True, slots=True)
class AllocationProfileDataset:
    reference: ArtifactRef
    directory: Path
    payload: dict[str, Any]
    kl_profile: KlBudgetProfile | None


def load_allocation_profile_dataset(
    root: str | Path,
    expected_key: str | None = None,
) -> AllocationProfileDataset:
    root = Path(root)
    pointer = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if pointer.get("schema_version") != 1:
        raise ValueError("unsupported allocation profile dataset schema")
    reference = from_dict(ArtifactRef, pointer["artifact"], path="allocation_dataset.artifact")
    if expected_key is not None and reference.artifact_id != expected_key:
        raise ValueError("allocation profile dataset key differs from the configured identity")
    store = LocalArtifactStore(root / "artifacts")
    descriptor = store.validate(reference.artifact_id)
    if (
        descriptor.artifact_type != "allocation-profile-dataset"
        or descriptor.schema_version != 1
        or str(reference.artifact_type) != descriptor.artifact_type
        or reference.schema_version != 1
    ):
        raise ValueError("invalid allocation profile dataset artifact")
    directory = store.path_for(reference.artifact_id)
    payload = json.loads((directory / "allocation-profile.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("units"):
        raise ValueError("allocation profile dataset is empty or unsupported")
    ids = [unit["evidence"]["unit_id"] for unit in payload["units"]]
    if len(set(ids)) != len(ids):
        raise ValueError("allocation profile dataset has duplicate units")
    kl_path = directory / "kl-budget-profile.json"
    profile = load_kl_budget_profile(kl_path) if kl_path.exists() else None
    if payload.get("kl_profile_key") != (None if profile is None else profile.profile_key):
        raise ValueError("allocation dataset KL profile identity differs")
    if profile is not None:
        validate_kl_budget_profile(
            profile,
            model_source=payload["model"]["source"],
            model_revision=payload["model"]["revision"],
        )
    return AllocationProfileDataset(reference, directory, payload, profile)


def publish_allocation_profile_dataset(
    root: Path,
    payload: dict[str, Any],
    profile: KlBudgetProfile | None,
) -> AllocationProfileDataset:
    store = LocalArtifactStore(root / "artifacts")
    with store.begin_write("allocation-profile-dataset") as writer:
        atomic_write_json(writer.path / "allocation-profile.json", payload)
        if profile is not None:
            atomic_write_json(writer.path / "kl-budget-profile.json", to_dict(profile))
        descriptor = writer.commit()
    reference = ArtifactRef(descriptor.artifact_type, descriptor.artifact_id, descriptor.schema_version)
    if (root / "dataset.json").exists():
        return load_allocation_profile_dataset(root, reference.artifact_id)
    atomic_write_json(root / "dataset.json", {"schema_version": 1, "artifact": to_dict(reference)})
    return load_allocation_profile_dataset(root, reference.artifact_id)
