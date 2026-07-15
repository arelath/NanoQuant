from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nanoquant.runtime import (
    FactorizedReferenceBackend,
    LogicalArtifactError,
    LogicalLayerState,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    WorkloadSpec,
    open_logical_artifact,
    plan_backends,
    prepare_plan,
    write_logical_artifact,
)


def _state(name: str, *, outliers: bool) -> LogicalLayerState:
    spec = QuantizedLinearSpec(
        name,
        "nanoquant-v1",
        4,
        3,
        2,
        "float32",
        "float32",
        outlier_count=1 if outliers else 0,
        outlier_value_dtype="int8" if outliers else None,
        has_outlier_scales=outliers,
        has_bias=True,
    )
    return LogicalLayerState(
        spec,
        torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]),
        torch.tensor([0.75, 1.0, 1.25, 1.5]),
        torch.tensor([0.5, 1.5]),
        torch.tensor([1.25, 0.75, 1.5]),
        bias=torch.tensor([0.1, -0.2, 0.3]),
        outlier_indices=torch.tensor([1], dtype=torch.int32) if outliers else None,
        outlier_values=torch.tensor([[2], [-1], [3]], dtype=torch.int8) if outliers else None,
        outlier_scales=torch.tensor([0.25]) if outliers else None,
    )


def _metadata() -> RuntimeModelMetadata:
    return RuntimeModelMetadata(
        "google/gemma-3-1b-it",
        "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "gemma3",
        "sha256:model-config",
        "sha256:tokenizer",
    )


def _write(tmp_path: Path):  # type: ignore[no-untyped-def]
    first = _state("blocks.0.self_attn.q_proj", outliers=True)
    second = _state("blocks.1.mlp.down_proj", outliers=False)
    artifact = write_logical_artifact(
        tmp_path / "logical-model",
        _metadata(),
        {0: (first,), 1: (second,)},
    )
    return artifact, first, second


def test_logical_artifact_roundtrip_is_block_sharded_and_runtime_executable(tmp_path: Path) -> None:
    artifact, first, second = _write(tmp_path)

    assert artifact.manifest.schema_version == 1
    assert artifact.manifest.logical_format == "nanoquant-v1"
    assert artifact.manifest.layer_count == 2
    assert [block.path for block in artifact.manifest.blocks] == [
        "weights/block-00000.safetensors",
        "weights/block-00001.safetensors",
    ]
    loaded_first = artifact.load_layer(first.spec.name)
    loaded_second = artifact.load_layer(second.spec.name)
    for expected, loaded in ((first, loaded_first), (second, loaded_second)):
        assert loaded.spec == expected.spec
        assert torch.equal(loaded.left_binary, expected.left_binary)
        assert torch.equal(loaded.right_binary, expected.right_binary)
        assert torch.equal(loaded.scale_pre, expected.scale_pre)
    backend = FactorizedReferenceBackend()
    workload = WorkloadSpec("decode", "cpu", "float32", 1, 1, deterministic=True)
    plan = plan_backends((loaded_first.spec, loaded_second.spec), workload, (backend,), strict=True)
    prepared = prepare_plan(
        plan,
        {loaded_first.spec.name: loaded_first, loaded_second.spec.name: loaded_second},
        (backend,),
        "cpu",
    )
    assert [item.linear(torch.ones(1, 4)).shape for item in prepared] == [(1, 3), (1, 3)]


def test_logical_artifact_open_validates_headers_without_eager_layer_loading(tmp_path: Path) -> None:
    artifact, first, _second = _write(tmp_path)
    second_shard = artifact.root / artifact.manifest.blocks[1].path
    second_shard.unlink()

    loaded = artifact.load_layer(first.spec.name)

    assert torch.equal(loaded.left_binary, first.left_binary)


def test_logical_artifact_rejects_corrupt_shard_hash(tmp_path: Path) -> None:
    artifact, _first, _second = _write(tmp_path)
    shard = artifact.root / artifact.manifest.blocks[0].path
    with shard.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 0xFF]))

    with pytest.raises(LogicalArtifactError, match="shard hash differs"):
        open_logical_artifact(artifact.root)


def test_logical_artifact_rejects_path_traversal_before_member_access(tmp_path: Path) -> None:
    artifact, _first, _second = _write(tmp_path)
    descriptor = artifact.root / "nanoquant-model.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["blocks"][0]["path"] = "../escape.safetensors"
    descriptor.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical and relative"):
        open_logical_artifact(artifact.root)


def test_logical_artifact_rejects_future_descriptor_schema(tmp_path: Path) -> None:
    artifact, _first, _second = _write(tmp_path)
    descriptor = artifact.root / "nanoquant-model.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    descriptor.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LogicalArtifactError, match="unsupported logical artifact schema"):
        open_logical_artifact(artifact.root)


def test_logical_artifact_write_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    artifact, first, _second = _write(tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        write_logical_artifact(artifact.root, _metadata(), {0: (first,)})
    assert not list(tmp_path.glob(".nanoquant-logical-*"))
