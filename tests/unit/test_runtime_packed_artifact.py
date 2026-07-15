from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from nanoquant.runtime import (
    PACKED_TENSOR_NAMESPACE,
    FactorizedReferenceBackend,
    LogicalLayerState,
    PackedArtifactError,
    PackedReferenceBackend,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    convert_logical_to_packed,
    open_packed_artifact,
    validate_packed_conversion,
    validate_packed_reference_parity,
    write_logical_artifact,
)


def _state(name: str, *, outliers: bool) -> LogicalLayerState:
    spec = QuantizedLinearSpec(
        name,
        "nanoquant-v1",
        35,
        3,
        33,
        "float32",
        "float32",
        outlier_count=2 if outliers else 0,
        outlier_value_dtype="int8" if outliers else None,
        has_outlier_scales=outliers,
        has_bias=True,
    )
    return LogicalLayerState(
        spec,
        torch.where(
            torch.arange(99).reshape(3, 33) % 2 == 0,
            torch.ones(3, 33),
            -torch.ones(3, 33),
        ),
        torch.where(
            torch.arange(33 * 35).reshape(33, 35) % 3 == 0,
            -torch.ones(33, 35),
            torch.ones(33, 35),
        ),
        (
            torch.cat((torch.tensor([0.5, 0.0]), torch.linspace(0.5, 1.5, 32), torch.zeros(1)))
            if outliers
            else torch.linspace(0.5, 1.5, 35)
        ),
        torch.linspace(0.75, 1.25, 33),
        torch.linspace(1.0, 1.5, 3),
        torch.tensor([0.1, -0.2, 0.3]),
        torch.tensor([1, 34], dtype=torch.int32) if outliers else None,
        torch.tensor([[1, -2], [3, -4], [5, -6]], dtype=torch.int8) if outliers else None,
        torch.tensor([0.25, 0.5]) if outliers else None,
    )


def _metadata() -> RuntimeModelMetadata:
    return RuntimeModelMetadata("fixture/model", "revision", "fixture", "config", "tokenizer")


def _convert(tmp_path: Path):  # type: ignore[no-untyped-def]
    first = _state("blocks.0.linear", outliers=True)
    second = _state("blocks.1.linear", outliers=False)
    logical = write_logical_artifact(
        tmp_path / "logical",
        _metadata(),
        {0: (first,), 1: (second,)},
    )
    packed = convert_logical_to_packed(logical.root, tmp_path / "packed")
    return logical, packed, first, second


def test_packed_artifact_conversion_is_sharded_bound_and_executable(tmp_path: Path) -> None:
    logical, packed, first, second = _convert(tmp_path)

    descriptor_hash = hashlib.sha256((logical.root / "nanoquant-model.json").read_bytes()).hexdigest()
    assert packed.manifest.logical_descriptor_sha256 == descriptor_hash
    assert packed.manifest.layout.version == "llama.cpp-i32-lsb-v1"
    assert packed.manifest.layout.reference.repository == "modified-llama.cpp"
    assert packed.manifest.layout.reference.commit == "5c6ae79816ee0f2b3d4bb8ec9061c294185d320b"
    assert packed.manifest.layer_count == 2
    assert packed.manifest.weight_bytes < logical.manifest.weight_bytes
    assert [block.path for block in packed.manifest.blocks] == [
        "weights/block-00000.safetensors",
        "weights/block-00001.safetensors",
    ]
    assert all(
        tensor.key.startswith(f"{PACKED_TENSOR_NAMESPACE}.")
        for block in packed.manifest.blocks
        for layer in block.layers
        for tensor in layer.tensors
    )
    loaded = packed.load_layer(first.spec.name)
    restored = loaded.to_logical()
    assert torch.equal(restored.left_binary, first.left_binary)
    assert torch.equal(restored.right_binary, first.right_binary)
    assert torch.equal(restored.outlier_values, first.outlier_values)
    value = torch.linspace(-0.5, 0.5, first.spec.in_features).reshape(1, -1)
    logical_backend = FactorizedReferenceBackend()
    packed_backend = PackedReferenceBackend()
    expected = logical_backend.linear(value, logical_backend.prepare(first, "cpu"))
    actual = packed_backend.linear(value, packed_backend.prepare(loaded, "cpu"))
    assert torch.equal(actual, expected)
    assert packed.load_layer(second.spec.name).spec == second.spec
    validation = validate_packed_conversion(logical.root, packed.root)
    assert validation.exact
    assert validation.logical_tensor_count == 15
    assert validation.packed_tensor_count == 15
    parity = validate_packed_reference_parity(logical.root, packed.root)
    assert parity.maximum_absolute_error == 0.0
    assert parity.output_elements == 6


def test_packed_artifact_open_validates_without_eager_payload_loading(tmp_path: Path) -> None:
    _logical, packed, first, _second = _convert(tmp_path)
    (packed.root / packed.manifest.blocks[1].path).unlink()

    loaded = packed.load_layer(first.spec.name)

    assert torch.equal(loaded.to_logical().left_binary, first.left_binary)


def test_packed_artifact_rejects_corrupt_shard(tmp_path: Path) -> None:
    _logical, packed, _first, _second = _convert(tmp_path)
    shard = packed.root / packed.manifest.blocks[0].path
    with shard.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 0xFF]))

    with pytest.raises(PackedArtifactError, match="shard hash differs"):
        open_packed_artifact(packed.root)


def test_packed_artifact_rejects_future_schema(tmp_path: Path) -> None:
    _logical, packed, _first, _second = _convert(tmp_path)
    descriptor = packed.root / "nanoquant-packed-model.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    descriptor.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PackedArtifactError, match="unsupported packed artifact schema"):
        open_packed_artifact(packed.root)


def test_packed_artifact_rejects_noncanonical_tensor_key(tmp_path: Path) -> None:
    _logical, packed, _first, _second = _convert(tmp_path)
    descriptor = packed.root / "nanoquant-packed-model.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["blocks"][0]["layers"][0]["tensors"][0]["key"] = "factor_left_words"
    descriptor.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PackedArtifactError, match="tensor key differs"):
        open_packed_artifact(packed.root, verify_hashes=False)


def test_packed_reference_validation_rejects_different_logical_binding(tmp_path: Path) -> None:
    logical, packed, _first, _second = _convert(tmp_path)
    descriptor = packed.root / "nanoquant-packed-model.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["logical_descriptor_sha256"] = "0" * 64
    descriptor.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bound to a different logical descriptor"):
        validate_packed_reference_parity(logical.root, packed.root)


def test_packed_artifact_conversion_refuses_overwrite(tmp_path: Path) -> None:
    logical, packed, _first, _second = _convert(tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        convert_logical_to_packed(logical.root, packed.root)
    assert not list(tmp_path.glob(".nanoquant-packed-*"))
