from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.tensor_store import LocalTensorStore


def test_tensor_content_verification_is_cached_by_immutable_file_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    tensors = LocalTensorStore(artifacts)
    from nanoquant.infrastructure.tensor_store import _tensor_hash

    calls = 0

    def counting_hash(value: torch.Tensor) -> str:
        nonlocal calls
        calls += 1
        return _tensor_hash(value)

    monkeypatch.setattr("nanoquant.infrastructure.tensor_store._tensor_hash", counting_hash)
    reference = tensors.put("fixture", {"value": torch.arange(16)})["value"]
    assert calls == 1

    for _ in range(2):
        with tensors.read(reference) as value:
            assert torch.equal(value, torch.arange(16))
    assert calls == 1

    reopened = LocalTensorStore(LocalArtifactStore(artifacts.root))
    with reopened.read(reference):
        pass
    with reopened.read(reference):
        pass
    assert calls == 2


def test_cached_tensor_verification_does_not_bypass_artifact_corruption_detection(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    tensors = LocalTensorStore(artifacts)
    reference = tensors.put("fixture", {"value": torch.arange(16)})["value"]
    path = artifacts.path_for(reference.artifact.artifact_id) / "tensors.safetensors"
    save_file({"value": torch.arange(16) + 1}, path)

    with pytest.raises(ArtifactCorruptionError, match="ART001"):
        with tensors.read(reference):
            pass
