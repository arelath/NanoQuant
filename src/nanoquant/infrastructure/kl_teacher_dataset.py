"""Portable KL teacher inputs and targets backed by immutable local artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import canonical_device_name
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_teacher_cache import (
    CommittedKlTeacherCache,
    load_active_kl_teacher_cache,
)
from nanoquant.infrastructure.safetensors_io import SAFETENSORS


def token_hash(tokens: torch.Tensor) -> str:
    payload = tokens.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def teacher_cache_key(
    *,
    source: str,
    revision: str,
    model_hash: str,
    token_hash: str,
    model_dtype: torch.dtype | str,
    attention_implementation: str | None,
    device: str,
    batch_size: int,
) -> str:
    # Preserve the existing campaign-cache identity so it can be imported exactly.
    payload = json.dumps(
        {
            "schema_version": 1,
            "source": source,
            "revision": revision,
            "model_hash": model_hash,
            "token_hash": token_hash,
            "model_dtype": str(model_dtype),
            "teacher_cache_dtype": str(torch.float16),
            "attention_implementation": attention_implementation,
            "device": device,
            "batch_size": batch_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class KlTeacherProtocol:
    source: str
    revision: str
    model_hash: str
    tokenizer_hash: str
    model_dtype: str
    attention_implementation: str | None
    device: str
    batch_size: int
    samples: int
    sequence_length: int
    vocabulary_size: int
    dataset_fingerprint: str
    bos_token_id: int | None
    token_hash: str

    @property
    def cache_key(self) -> str:
        return teacher_cache_key(
            source=self.source,
            revision=self.revision,
            model_hash=self.model_hash,
            token_hash=self.token_hash,
            model_dtype=self.model_dtype,
            attention_implementation=self.attention_implementation,
            device=self.device,
            batch_size=self.batch_size,
        )


@dataclass(frozen=True, slots=True)
class KlTeacherDataset:
    reference: ArtifactRef
    protocol: KlTeacherProtocol
    tokens: torch.Tensor
    cache: CommittedKlTeacherCache


def validate_teacher_inputs(
    protocol: KlTeacherProtocol,
    tokens: torch.Tensor,
    cache: CommittedKlTeacherCache,
) -> None:
    if (
        protocol.samples <= 0
        or protocol.sequence_length < 2
        or protocol.batch_size <= 0
        or protocol.vocabulary_size <= 0
        or tokens.dtype != torch.int64
        or tuple(tokens.shape) != (protocol.samples, protocol.sequence_length)
        or token_hash(tokens) != protocol.token_hash
        or int(tokens.min()) < 0
        or int(tokens.max()) >= protocol.vocabulary_size
        or cache.cache_key != protocol.cache_key
    ):
        raise ValueError("KL teacher dataset inputs or identity are invalid")
    starts = range(0, protocol.samples, protocol.batch_size)
    if len(cache.batches) != len(starts):
        raise ValueError("KL teacher dataset batch count is invalid")
    for start, batch in zip(starts, cache.batches, strict=True):
        shape = (min(protocol.batch_size, protocol.samples - start), protocol.sequence_length, protocol.vocabulary_size)
        if batch.dtype != torch.float16 or tuple(batch.shape) != shape:
            raise ValueError("KL teacher dataset target shape or dtype is invalid")


def load_kl_teacher_dataset(root: str | Path) -> KlTeacherDataset:
    root = Path(root)
    pointer = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if pointer.get("schema_version") != 1:
        raise ValueError("unsupported KL teacher dataset schema")
    reference = from_dict(ArtifactRef, pointer["artifact"], path="dataset.artifact")
    store = LocalArtifactStore(root / "artifacts")
    descriptor = store.validate(reference.artifact_id)
    if (
        descriptor.artifact_type != "kl-teacher-dataset"
        or descriptor.schema_version != 1
        or str(reference.artifact_type) != descriptor.artifact_type
        or reference.schema_version != descriptor.schema_version
    ):
        raise ValueError("invalid KL teacher dataset artifact")
    artifact_root = store.path_for(reference.artifact_id)
    metadata = json.loads((artifact_root / "dataset.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported KL teacher dataset metadata schema")
    protocol = from_dict(KlTeacherProtocol, metadata["protocol"], path="dataset.protocol")
    tokens = SAFETENSORS.load(artifact_root / "tokens.safetensors")["token_ids"]
    cache = load_active_kl_teacher_cache(root, protocol.cache_key)
    if cache is None or to_dict(cache.reference) != metadata["teacher_cache"]:
        raise ValueError("KL teacher dataset target reference is missing or mismatched")
    validate_teacher_inputs(protocol, tokens, cache)
    return KlTeacherDataset(reference, protocol, tokens, cache)


def publish_kl_teacher_dataset(
    root: Path,
    protocol: KlTeacherProtocol,
    tokens: torch.Tensor,
    cache: CommittedKlTeacherCache,
) -> KlTeacherDataset:
    """Publish inputs last, after the target cache is durably committed in root."""
    validate_teacher_inputs(protocol, tokens, cache)
    if (root / "dataset.json").exists():
        existing = load_kl_teacher_dataset(root)
        if existing.protocol != protocol or existing.cache.reference != cache.reference:
            raise ValueError("output already contains a different KL teacher dataset")
        return existing
    store = LocalArtifactStore(root / "artifacts")
    with store.begin_write("kl-teacher-dataset") as writer:
        SAFETENSORS.save({"token_ids": tokens.cpu().contiguous()}, writer.path / "tokens.safetensors")
        atomic_write_json(
            writer.path / "dataset.json",
            {
                "schema_version": 1,
                "protocol": to_dict(protocol),
                "teacher_cache": to_dict(cache.reference),
            },
        )
        descriptor = writer.commit()
    reference = ArtifactRef(descriptor.artifact_type, descriptor.artifact_id, descriptor.schema_version)
    atomic_write_json(root / "dataset.json", {"schema_version": 1, "artifact": to_dict(reference)})
    return load_kl_teacher_dataset(root)


def require_compatible_teacher_dataset(
    dataset: KlTeacherDataset,
    *,
    source: str,
    revision: str,
    model_hash: str,
    tokenizer_hash: str,
    model_dtype: torch.dtype,
    attention_implementation: str | None,
    device: str,
    batch_size: int,
    samples: int,
    sequence_length: int,
    vocabulary_size: int,
) -> None:
    protocol = dataset.protocol
    expected = {
        "source": source,
        "revision": revision,
        "model_hash": model_hash,
        "tokenizer_hash": tokenizer_hash,
        "model_dtype": str(model_dtype),
        "attention_implementation": attention_implementation,
        "batch_size": batch_size,
        "samples": samples,
        "sequence_length": sequence_length,
        "vocabulary_size": vocabulary_size,
    }
    mismatched = [key for key, value in expected.items() if getattr(protocol, key) != value]
    if canonical_device_name(protocol.device) != canonical_device_name(device):
        mismatched.append("device")
    if mismatched:
        raise ValueError("KL teacher dataset differs from the requested " + ", ".join(mismatched))
