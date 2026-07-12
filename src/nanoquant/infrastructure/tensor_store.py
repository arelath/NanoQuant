"""Non-executable safetensors persistence over the local artifact store."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.domain.models import ArtifactRef, TensorRef, TensorSpec

from .artifacts import LocalArtifactStore


def _tensor_hash(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode() + b"\0")
    digest.update(str(tuple(contiguous.shape)).encode() + b"\0")
    if contiguous.numel():
        digest.update(memoryview(cast(Any, contiguous.view(torch.uint8).numpy())).cast("B"))
    return "sha256:" + digest.hexdigest()


class LocalTensorStore:
    def __init__(self, artifacts: LocalArtifactStore) -> None:
        self.artifacts = artifacts

    def put(self, artifact_type: str, tensors: dict[str, torch.Tensor]) -> dict[str, TensorRef]:
        if not tensors:
            raise ValueError("tensor artifact must not be empty")
        copied = {key: value.detach().cpu().clone().contiguous() for key, value in tensors.items()}
        with self.artifacts.begin_write(artifact_type) as writer:
            save_file(copied, writer.path / "tensors.safetensors")
            descriptor = writer.commit()
        artifact = ArtifactRef(artifact_type, descriptor.artifact_id, descriptor.schema_version)
        return {
            key: TensorRef(
                artifact,
                key,
                TensorSpec(tuple(value.shape), str(value.dtype).removeprefix("torch.")),
                _tensor_hash(value),
            )
            for key, value in copied.items()
        }

    @contextmanager
    def read(self, reference: TensorRef, device: str = "cpu") -> Iterator[torch.Tensor]:
        self.artifacts.validate(reference.artifact.artifact_id)
        path = self.artifacts.path_for(reference.artifact.artifact_id) / "tensors.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            if reference.key not in handle.keys():
                raise KeyError(f"tensor key not in artifact: {reference.key}")
            value = handle.get_tensor(reference.key)
            if (
                tuple(value.shape) != reference.spec.shape
                or str(value.dtype).removeprefix("torch.") != reference.spec.dtype
            ):
                raise OSError("ART001 tensor spec mismatch")
            if _tensor_hash(value) != reference.content_hash:
                raise OSError("ART001 tensor content hash mismatch")
            yield value if device == "cpu" else value.to(device)
