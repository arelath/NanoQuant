"""Hash-pinned, model-compatible foldable MLP multiplier initializer artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from nanoquant.infrastructure.io_utils import hash_file
from nanoquant.infrastructure.safetensors_io import SAFETENSORS


@dataclass(frozen=True, slots=True)
class FoldableMlpInitializer:
    root: Path
    manifest: dict[str, Any]
    tensors: dict[str, torch.Tensor]

    @property
    def tensor_sha256(self) -> str:
        return str(self.manifest["tensor_sha256"])


def load_foldable_mlp_initializer(
    root: str | Path,
    *,
    expected_sha256: str,
    model_source: str,
    model_revision: str,
) -> FoldableMlpInitializer:
    artifact = Path(root)
    manifest_path = artifact / "manifest.json"
    tensor_path = artifact / "multipliers.safetensors"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise ValueError("foldable MLP initializer artifact is incomplete")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("foldable MLP initializer manifest must be an object")
    manifest = cast(dict[str, Any], payload)
    inventory = manifest.get("tensors")
    actual_sha256 = hash_file(tensor_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("semantics") != "foldable-mlp-log-multiplier-initializer"
        or manifest.get("model") != {"source": model_source, "revision": model_revision}
        or actual_sha256 != expected_sha256
        or manifest.get("tensor_sha256") != actual_sha256
        or not isinstance(inventory, dict)
        or manifest.get("tensor_count") != len(inventory)
    ):
        raise ValueError("foldable MLP initializer identity or inventory is invalid")
    tensors = SAFETENSORS.load(tensor_path)
    if set(tensors) != set(inventory) or any(
        value.dtype != torch.float32
        or value.ndim != 1
        or not torch.isfinite(value).all()
        or list(value.shape) != inventory[name].get("shape")
        or inventory[name].get("dtype") != "float32"
        for name, value in tensors.items()
    ):
        raise ValueError("foldable MLP initializer tensors are invalid")
    return FoldableMlpInitializer(artifact, manifest, tensors)


__all__ = ["FoldableMlpInitializer", "load_foldable_mlp_initializer"]
