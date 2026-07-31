"""Durable content-keyed dense reconstruction cache for analysis probes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from nanoquant.config.codec import semantic_hash
from nanoquant.infrastructure.io_utils import (
    atomic_workspace,
    atomic_write_json,
    hash_file,
)
from nanoquant.infrastructure.safetensors_io import SAFETENSORS

PROBE_RECONSTRUCTION_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProbeReconstructionCacheEntry:
    baseline: torch.Tensor
    candidate: torch.Tensor
    baseline_metrics: dict[str, object]
    candidate_metrics: dict[str, object]
    candidate_index_metrics: dict[str, object]
    rank: int
    matrix_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if (
            self.rank <= 0
            or min(self.matrix_shape) <= 0
            or self.baseline.ndim != 2
            or self.candidate.ndim != 2
            or tuple(self.baseline.shape) != self.matrix_shape
            or tuple(self.candidate.shape) != self.matrix_shape
            or not torch.isfinite(self.baseline).all()
            or not torch.isfinite(self.candidate).all()
        ):
            raise ValueError(
                "probe reconstruction cache entry is invalid"
            )


class ProbeReconstructionCache:
    """Persist and strictly validate fitted baseline/candidate matrices."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(identity: dict[str, object]) -> str:
        return semantic_hash(identity).removeprefix("sha256:")

    def _entry_path(self, identity: dict[str, object]) -> Path:
        return self.root / self.key(identity)

    def load(
        self,
        identity: dict[str, object],
    ) -> ProbeReconstructionCacheEntry | None:
        entry_path = self._entry_path(identity)
        if not entry_path.exists():
            return None
        manifest_path = entry_path / "manifest.json"
        tensor_path = entry_path / "reconstructions.safetensors"
        if not manifest_path.is_file() or not tensor_path.is_file():
            raise ValueError(
                "probe reconstruction cache entry is incomplete"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                "probe reconstruction cache manifest must be an object"
            )
        expected_key = self.key(identity)
        if (
            payload.get("schema_version")
            != PROBE_RECONSTRUCTION_CACHE_SCHEMA_VERSION
            or payload.get("cache_key") != expected_key
            or payload.get("identity") != identity
            or payload.get("tensor_sha256") != hash_file(tensor_path)
        ):
            raise ValueError(
                "probe reconstruction cache identity or hash is invalid"
            )
        with SAFETENSORS.open(tensor_path, device="cpu") as handle:
            if set(handle.keys()) != {"baseline", "candidate"}:
                raise ValueError(
                    "probe reconstruction cache tensor inventory is invalid"
                )
            baseline = handle.get_tensor("baseline")
            candidate = handle.get_tensor("candidate")
        matrix_shape_value = payload.get("matrix_shape")
        if (
            not isinstance(matrix_shape_value, list)
            or len(matrix_shape_value) != 2
            or any(
                not isinstance(value, int) or value <= 0
                for value in matrix_shape_value
            )
        ):
            raise ValueError(
                "probe reconstruction cache matrix shape is invalid"
            )
        rank = payload.get("rank")
        metrics = payload.get("metrics")
        index_metrics = payload.get("candidate_index_metrics")
        if (
            not isinstance(rank, int)
            or rank <= 0
            or not isinstance(metrics, dict)
            or set(metrics) != {"baseline", "candidate"}
            or not all(
                isinstance(metrics[name], dict)
                for name in ("baseline", "candidate")
            )
            or not isinstance(index_metrics, dict)
        ):
            raise ValueError(
                "probe reconstruction cache metadata is invalid"
            )
        return ProbeReconstructionCacheEntry(
            baseline=baseline,
            candidate=candidate,
            baseline_metrics=cast(
                dict[str, object],
                metrics["baseline"],
            ),
            candidate_metrics=cast(
                dict[str, object],
                metrics["candidate"],
            ),
            candidate_index_metrics=cast(
                dict[str, object],
                index_metrics,
            ),
            rank=rank,
            matrix_shape=(
                matrix_shape_value[0],
                matrix_shape_value[1],
            ),
        )

    def store(
        self,
        identity: dict[str, object],
        entry: ProbeReconstructionCacheEntry,
    ) -> str:
        cache_key = self.key(identity)
        destination = self.root / cache_key
        if destination.exists():
            existing = self.load(identity)
            if existing is None:
                raise RuntimeError(
                    "existing probe reconstruction cache entry vanished"
                )
            return cache_key
        with atomic_workspace(
            destination,
            prefix=f".{cache_key}-",
        ) as temporary:
            tensor_path = temporary / "reconstructions.safetensors"
            SAFETENSORS.save(
                {
                    "baseline": entry.baseline.detach()
                    .to(device="cpu", dtype=torch.bfloat16)
                    .contiguous(),
                    "candidate": entry.candidate.detach()
                    .to(device="cpu", dtype=torch.bfloat16)
                    .contiguous(),
                },
                tensor_path,
            )
            atomic_write_json(
                temporary / "manifest.json",
                {
                    "schema_version": (
                        PROBE_RECONSTRUCTION_CACHE_SCHEMA_VERSION
                    ),
                    "cache_key": cache_key,
                    "identity": identity,
                    "tensor_sha256": hash_file(tensor_path),
                    "rank": entry.rank,
                    "matrix_shape": list(entry.matrix_shape),
                    "metrics": {
                        "baseline": entry.baseline_metrics,
                        "candidate": entry.candidate_metrics,
                    },
                    "candidate_index_metrics": (
                        entry.candidate_index_metrics
                    ),
                },
            )
        return cache_key


__all__ = [
    "PROBE_RECONSTRUCTION_CACHE_SCHEMA_VERSION",
    "ProbeReconstructionCache",
    "ProbeReconstructionCacheEntry",
]
