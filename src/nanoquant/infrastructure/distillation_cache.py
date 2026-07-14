"""Durable per-epoch teacher-target cache for resumable top-k distillation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.application.distillation import TopKTeacherBatch, TopKTeacherCache
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef

from .artifacts import LocalArtifactStore
from .io_utils import safe_replace


@dataclass(frozen=True, slots=True)
class TeacherCacheIdentity:
    protocol_hash: str
    token_hash: str


@dataclass(frozen=True, slots=True)
class TeacherCacheJournal:
    schema_version: int
    identity: TeacherCacheIdentity
    epochs: tuple[ArtifactRef | None, ...]


@dataclass(frozen=True, slots=True)
class CommittedTeacherEpoch:
    reference: ArtifactRef
    epoch_index: int
    batches: tuple[TopKTeacherBatch, ...]
    bytes: int


def commit_teacher_epoch(
    epoch_index: int,
    batches: tuple[TopKTeacherBatch, ...],
    identity: TeacherCacheIdentity,
    artifacts: LocalArtifactStore,
) -> CommittedTeacherEpoch:
    values: dict[str, torch.Tensor] = {}
    manifest_batches = []
    cache_bytes = 0
    for batch_index, batch in enumerate(batches):
        prefix = f"batch_{batch_index}"
        values[f"{prefix}.token_indices"] = batch.token_indices.contiguous()
        values[f"{prefix}.top_values"] = batch.top_values.contiguous()
        values[f"{prefix}.top_indices"] = batch.top_indices.contiguous()
        manifest_batches.append({"prefix": prefix, "sample_indices": list(batch.sample_indices)})
        cache_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in (batch.token_indices, batch.top_values, batch.top_indices)
        )
    if not values:
        raise ValueError("cannot commit an empty teacher-target epoch")
    with artifacts.recorder.phase("serialize"):
        encoded = json.dumps(
            {
                "schema_version": 1,
                "identity": to_dict(identity),
                "epoch_index": epoch_index,
                "bytes": cache_bytes,
                "batches": manifest_batches,
            },
            sort_keys=True,
            indent=2,
        )
    with artifacts.begin_write("topk-teacher-epoch") as writer:
        with artifacts.recorder.phase("write"):
            save_file(values, writer.path / "targets.safetensors")
            (writer.path / "epoch.json").write_text(encoded, encoding="utf-8")
        descriptor = writer.commit()
    reference = ArtifactRef("topk-teacher-epoch", descriptor.artifact_id, descriptor.schema_version)
    return CommittedTeacherEpoch(reference, epoch_index, batches, cache_bytes)


def load_teacher_epoch(
    reference: ArtifactRef,
    identity: TeacherCacheIdentity,
    artifacts: LocalArtifactStore,
) -> CommittedTeacherEpoch:
    descriptor = artifacts.validate(reference.artifact_id)
    if descriptor.artifact_type != "topk-teacher-epoch":
        raise ValueError("artifact is not a top-k teacher epoch")
    root = artifacts.path_for(reference.artifact_id)
    manifest = json.loads((root / "epoch.json").read_text(encoding="utf-8"))
    observed_identity = from_dict(TeacherCacheIdentity, manifest["identity"], path="teacher_cache.identity")
    if observed_identity != identity:
        raise ValueError("teacher-cache artifact identity does not match the requested protocol")
    batches = []
    with safe_open(root / "targets.safetensors", framework="pt", device="cpu") as handle:
        for batch in manifest["batches"]:
            prefix = str(batch["prefix"])
            batches.append(
                TopKTeacherBatch(
                    tuple(int(index) for index in batch["sample_indices"]),
                    handle.get_tensor(f"{prefix}.token_indices"),
                    handle.get_tensor(f"{prefix}.top_values"),
                    handle.get_tensor(f"{prefix}.top_indices"),
                )
            )
    return CommittedTeacherEpoch(reference, int(manifest["epoch_index"]), tuple(batches), int(manifest["bytes"]))


def load_teacher_cache_journal(
    run_output: str | Path,
    identity: TeacherCacheIdentity,
    epoch_count: int,
    *,
    replace_mismatched: bool = False,
) -> TeacherCacheJournal:
    path = Path(run_output) / "global-distillation-cache.json"
    if not path.exists():
        return TeacherCacheJournal(1, identity, (None,) * epoch_count)
    journal = from_dict(
        TeacherCacheJournal,
        json.loads(path.read_text(encoding="utf-8")),
        path="teacher_cache_journal",
    )
    if journal.identity != identity or len(journal.epochs) != epoch_count:
        if replace_mismatched:
            return TeacherCacheJournal(1, identity, (None,) * epoch_count)
        raise ValueError("existing teacher-cache journal does not match the requested protocol")
    return journal


def record_teacher_epoch(
    run_output: str | Path,
    journal: TeacherCacheJournal,
    epoch_index: int,
    reference: ArtifactRef,
) -> TeacherCacheJournal:
    if epoch_index < 0 or epoch_index >= len(journal.epochs):
        raise ValueError("teacher-cache journal epoch index is out of range")
    epochs = list(journal.epochs)
    if epochs[epoch_index] is not None and epochs[epoch_index] != reference:
        raise ValueError("teacher-cache journal epoch is already committed differently")
    epochs[epoch_index] = reference
    updated = TeacherCacheJournal(journal.schema_version, journal.identity, tuple(epochs))
    output = Path(run_output)
    output.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="teacher-cache-", suffix=".tmp", dir=output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(to_dict(updated), stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        safe_replace(temporary, output / "global-distillation-cache.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return updated


def materialize_teacher_cache(
    journal: TeacherCacheJournal,
    artifacts: LocalArtifactStore,
) -> TopKTeacherCache:
    if any(reference is None for reference in journal.epochs):
        raise ValueError("teacher-cache journal is incomplete")
    committed = tuple(
        load_teacher_epoch(reference, journal.identity, artifacts)
        for reference in journal.epochs
        if reference is not None
    )
    if tuple(epoch.epoch_index for epoch in committed) != tuple(range(len(journal.epochs))):
        raise ValueError("teacher-cache epoch artifacts are not ordered and complete")
    return TopKTeacherCache(tuple(epoch.batches for epoch in committed), sum(epoch.bytes for epoch in committed))
