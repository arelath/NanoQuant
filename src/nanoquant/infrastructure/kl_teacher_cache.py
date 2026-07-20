"""Content-addressed persistent teacher log-prob caches for repeated KL profiles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.config.codec import to_dict
from nanoquant.domain.models import ArtifactRef, ArtifactTypes
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.io_utils import atomic_write_json


@dataclass(frozen=True, slots=True)
class CommittedKlTeacherCache:
    reference: ArtifactRef
    cache_key: str
    baseline_negative_log_likelihood: float
    batches: tuple[torch.Tensor, ...]
    tensor_bytes: int


def _store(root: Path) -> LocalArtifactStore:
    return LocalArtifactStore(root / "artifacts")


def load_active_kl_teacher_cache(
    root: str | Path,
    expected_cache_key: str,
) -> CommittedKlTeacherCache | None:
    cache_root = Path(root)
    pointer = cache_root / "active.json"
    if not pointer.exists():
        return None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("KL teacher-cache pointer is invalid")
    if payload.get("cache_key") != expected_cache_key:
        raise ValueError("KL teacher-cache identity differs from the requested teacher protocol")
    reference_payload = payload.get("artifact")
    if not isinstance(reference_payload, dict):
        raise ValueError("KL teacher-cache pointer has no artifact reference")
    reference = ArtifactRef(
        str(reference_payload["artifact_type"]),
        str(reference_payload["artifact_id"]),
        int(reference_payload["schema_version"]),
    )
    artifacts = _store(cache_root)
    descriptor = artifacts.validate(reference.artifact_id)
    if descriptor.artifact_type != ArtifactTypes.KL_TEACHER_CACHE:
        raise ValueError("active KL teacher cache has the wrong artifact type")
    artifact_root = artifacts.path_for(reference.artifact_id)
    manifest = json.loads((artifact_root / "cache.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("cache_key") != expected_cache_key
    ):
        raise ValueError("KL teacher-cache artifact identity is invalid")
    baseline_nll = float(manifest["baseline_negative_log_likelihood"])
    if not math.isfinite(baseline_nll):
        raise ValueError("KL teacher-cache baseline NLL is not finite")
    batch_count = int(manifest["batch_count"])
    with safe_open(artifact_root / "log-probs.safetensors", framework="pt", device="cpu") as handle:
        expected_keys = tuple(f"batch_{index:04d}" for index in range(batch_count))
        if tuple(sorted(handle.keys())) != expected_keys:
            raise ValueError("KL teacher-cache tensor keys are incomplete")
        batches = tuple(handle.get_tensor(key) for key in expected_keys)
    tensor_bytes = sum(batch.numel() * batch.element_size() for batch in batches)
    if tensor_bytes != int(manifest["tensor_bytes"]):
        raise ValueError("KL teacher-cache tensor byte count differs from its manifest")
    return CommittedKlTeacherCache(
        reference,
        expected_cache_key,
        baseline_nll,
        batches,
        tensor_bytes,
    )


def commit_active_kl_teacher_cache(
    root: str | Path,
    cache_key: str,
    baseline_negative_log_likelihood: float,
    batches: tuple[torch.Tensor, ...],
) -> CommittedKlTeacherCache:
    if not cache_key or not math.isfinite(baseline_negative_log_likelihood) or not batches:
        raise ValueError("KL teacher cache requires a key, finite baseline NLL, and batches")
    cache_root = Path(root)
    existing = load_active_kl_teacher_cache(cache_root, cache_key)
    if existing is not None:
        return existing
    values = {
        f"batch_{index:04d}": batch.detach().cpu().contiguous()
        for index, batch in enumerate(batches)
    }
    tensor_bytes = sum(value.numel() * value.element_size() for value in values.values())
    artifacts = _store(cache_root)
    with artifacts.begin_write(ArtifactTypes.KL_TEACHER_CACHE) as writer:
        save_file(values, writer.path / "log-probs.safetensors")
        (writer.path / "cache.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cache_key": cache_key,
                    "baseline_negative_log_likelihood": baseline_negative_log_likelihood,
                    "batch_count": len(values),
                    "tensor_bytes": tensor_bytes,
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    reference = ArtifactRef(
        ArtifactTypes.KL_TEACHER_CACHE,
        descriptor.artifact_id,
        descriptor.schema_version,
    )
    atomic_write_json(
        cache_root / "active.json",
        {
            "schema_version": 1,
            "cache_key": cache_key,
            "artifact": to_dict(reference),
        },
    )
    return CommittedKlTeacherCache(
        reference,
        cache_key,
        baseline_negative_log_likelihood,
        tuple(values.values()),
        tensor_bytes,
    )


__all__ = [
    "CommittedKlTeacherCache",
    "commit_active_kl_teacher_cache",
    "load_active_kl_teacher_cache",
]
