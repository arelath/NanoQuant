"""Versioned block-sharded artifacts for the llama.cpp NanoQuant packed layout."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.runtime.artifact import (
    MINIMUM_RUNTIME_VERSION,
    OpenLogicalArtifact,
    RuntimeModelMetadata,
    _hash_file,
    _header_dtype,
    _integer,
    _mapping,
    _resolved_member,
    _sequence,
    _spec_from_payload,
    _string,
    _validate_relative_path,
    open_logical_artifact,
)
from nanoquant.runtime.backend import DeviceLike, QuantizedLinearSpec
from nanoquant.runtime.logical import LogicalLayerState, canonical_torch_dtype
from nanoquant.runtime.packed import (
    PACKED_TENSOR_NAMESPACE,
    PackedLayerState,
    PackedLayoutMetadata,
    PackedReferenceProvenance,
    pack_logical_layer,
    packed_word_count,
)
from nanoquant.runtime.reference import FactorizedReferenceBackend, PackedReferenceBackend

PACKED_DESCRIPTOR_SCHEMA_VERSION = 1
PACKED_ARTIFACT_FORMAT = "nanoquant-packed-model"
_MAXIMUM_DESCRIPTOR_BYTES = 16 * 1024 * 1024
_PACKED_TENSOR_ROLES = (
    "factor_left_words",
    "factor_right_words",
    "scale_pre",
    "scale_mid",
    "scale_post",
    "bias",
    "outlier_indices",
    "outlier_values",
    "outlier_scales",
)


class PackedArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PackedTensorEntry:
    role: str
    key: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if self.role not in _PACKED_TENSOR_ROLES:
            raise ValueError(f"unsupported packed tensor role: {self.role}")
        if not self.key or self.key.startswith("/"):
            raise ValueError("packed tensor key must be non-empty and relative")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("packed tensor dimensions must be non-negative")
        if not self.dtype:
            raise ValueError("packed tensor dtype must be non-empty")


def _expected_tensor_metadata(spec: QuantizedLinearSpec) -> dict[str, tuple[tuple[int, ...], str]]:
    expected = {
        "factor_left_words": (
            (spec.out_features, packed_word_count(spec.rank)),
            "int32",
        ),
        "factor_right_words": (
            (spec.rank, packed_word_count(spec.in_features)),
            "int32",
        ),
        "scale_pre": ((spec.in_features,), spec.scale_dtype),
        "scale_mid": ((spec.rank,), spec.scale_dtype),
        "scale_post": ((spec.out_features,), spec.scale_dtype),
    }
    if spec.has_bias:
        expected["bias"] = ((spec.out_features,), spec.scale_dtype)
    if spec.outlier_count:
        expected["outlier_indices"] = ((spec.outlier_count,), "int32")
        expected["outlier_values"] = (
            (spec.out_features, spec.outlier_count),
            cast(str, spec.outlier_value_dtype),
        )
    if spec.has_outlier_scales:
        expected["outlier_scales"] = ((spec.outlier_count,), spec.scale_dtype)
    return expected


@dataclass(frozen=True, slots=True)
class PackedLayerEntry:
    spec: QuantizedLinearSpec
    tensors: tuple[PackedTensorEntry, ...]

    def __post_init__(self) -> None:
        by_role = {tensor.role: tensor for tensor in self.tensors}
        if len(by_role) != len(self.tensors):
            raise ValueError(f"packed layer tensor roles are duplicated: {self.spec.name}")
        expected = _expected_tensor_metadata(self.spec)
        if set(by_role) != set(expected):
            raise ValueError(f"packed layer tensor inventory differs from its specification: {self.spec.name}")
        for role, (shape, dtype) in expected.items():
            tensor = by_role[role]
            expected_key = f"{PACKED_TENSOR_NAMESPACE}.{self.spec.name}.{role}"
            if tensor.key != expected_key:
                raise ValueError(f"packed layer tensor key differs: {self.spec.name}:{role}")
            if tensor.shape != shape or tensor.dtype != dtype:
                raise ValueError(f"packed layer tensor metadata differs: {self.spec.name}:{role}")


@dataclass(frozen=True, slots=True)
class PackedBlockEntry:
    index: int
    path: str
    bytes: int
    sha256: str
    layers: tuple[PackedLayerEntry, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("packed block index must be non-negative")
        if self.bytes <= 0:
            raise ValueError("packed block shard must be non-empty")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("packed block hash must be a lowercase SHA-256 digest")
        _validate_relative_path(self.path)
        names = [layer.spec.name for layer in self.layers]
        if not names or len(names) != len(set(names)):
            raise ValueError("packed block must contain uniquely named layers")


@dataclass(frozen=True, slots=True)
class PackedModelManifest:
    schema_version: int
    artifact_format: str
    minimum_runtime_version: str
    layout: PackedLayoutMetadata
    model: RuntimeModelMetadata
    logical_descriptor_sha256: str
    blocks: tuple[PackedBlockEntry, ...]
    layer_count: int
    weight_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != PACKED_DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError(f"unsupported packed artifact schema: {self.schema_version}")
        if self.artifact_format != PACKED_ARTIFACT_FORMAT:
            raise ValueError(f"unsupported packed artifact format: {self.artifact_format}")
        if self.minimum_runtime_version != MINIMUM_RUNTIME_VERSION:
            raise ValueError(f"unsupported packed minimum runtime version: {self.minimum_runtime_version}")
        if len(self.logical_descriptor_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.logical_descriptor_sha256
        ):
            raise ValueError("packed source descriptor hash must be lowercase SHA-256")
        indexes = [block.index for block in self.blocks]
        if not indexes or indexes != list(range(len(self.blocks))):
            raise ValueError(f"packed artifact blocks are not contiguous: {indexes}")
        names = [layer.spec.name for block in self.blocks for layer in block.layers]
        if len(names) != len(set(names)) or len(names) != self.layer_count:
            raise ValueError("packed artifact layer count or uniqueness is inconsistent")
        if sum(block.bytes for block in self.blocks) != self.weight_bytes:
            raise ValueError("packed artifact weight byte count is inconsistent")


@dataclass(frozen=True, slots=True)
class OpenPackedArtifact:
    root: Path
    manifest: PackedModelManifest

    def load_layer(self, name: str, device: DeviceLike = "cpu") -> PackedLayerState:
        matches = [
            (block, layer) for block in self.manifest.blocks for layer in block.layers if layer.spec.name == name
        ]
        if len(matches) != 1:
            raise KeyError(f"packed artifact layer not found: {name}")
        block, layer = matches[0]
        by_role: dict[str, torch.Tensor] = {}
        with safe_open(
            self.root / block.path,
            framework="pt",
            device=str(torch.device(device)),
        ) as handle:
            for tensor in layer.tensors:
                by_role[tensor.role] = handle.get_tensor(tensor.key)
        return PackedLayerState(
            layer.spec,
            self.manifest.layout.version,
            by_role["factor_left_words"],
            by_role["factor_right_words"],
            by_role["scale_pre"],
            by_role["scale_mid"],
            by_role["scale_post"],
            by_role.get("bias"),
            by_role.get("outlier_indices"),
            by_role.get("outlier_values"),
            by_role.get("outlier_scales"),
        )


@dataclass(frozen=True, slots=True)
class PackedConversionValidation:
    logical_artifact: Path
    packed_artifact: Path
    block_count: int
    layer_count: int
    logical_tensor_count: int
    packed_tensor_count: int
    logical_weight_bytes: int
    packed_weight_bytes: int
    storage_ratio: float
    exact: bool


@dataclass(frozen=True, slots=True)
class PackedReferenceParityResult:
    logical_artifact: Path
    packed_artifact: Path
    layer_count: int
    output_elements: int
    maximum_absolute_error: float
    maximum_error_layer: str


def _state_tensors(state: PackedLayerState) -> tuple[tuple[str, torch.Tensor], ...]:
    values = [
        ("factor_left_words", state.left_words),
        ("factor_right_words", state.right_words),
        ("scale_pre", state.scale_pre),
        ("scale_mid", state.scale_mid),
        ("scale_post", state.scale_post),
    ]
    optional = (
        ("bias", state.bias),
        ("outlier_indices", state.outlier_indices),
        ("outlier_values", state.outlier_values),
        ("outlier_scales", state.outlier_scales),
    )
    values.extend((role, value) for role, value in optional if value is not None)
    return tuple(values)


def _write_block_shard(
    temporary: Path,
    index: int,
    states: Sequence[PackedLayerState],
) -> PackedBlockEntry:
    tensors: dict[str, torch.Tensor] = {}
    layer_entries = []
    for state in states:
        entries = []
        for role, value in _state_tensors(state):
            key = f"{PACKED_TENSOR_NAMESPACE}.{state.spec.name}.{role}"
            if key in tensors:
                raise ValueError(f"packed artifact tensor key is duplicated: {key}")
            copied = value.detach().cpu().contiguous()
            tensors[key] = copied
            entries.append(
                PackedTensorEntry(
                    role,
                    key,
                    tuple(copied.shape),
                    canonical_torch_dtype(copied.dtype),
                )
            )
        layer_entries.append(PackedLayerEntry(state.spec, tuple(entries)))
    relative = f"weights/block-{index:05d}.safetensors"
    shard = temporary / relative
    save_file(tensors, shard)
    return PackedBlockEntry(
        index,
        relative,
        shard.stat().st_size,
        _hash_file(shard),
        tuple(layer_entries),
    )


def write_packed_artifact_stream(
    output: str | Path,
    model: RuntimeModelMetadata,
    logical_descriptor_sha256: str,
    blocks: Iterable[tuple[int, list[PackedLayerState]]],
) -> OpenPackedArtifact:
    """Atomically consume one mutable packed-state list per block."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"packed artifact output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".nanoquant-packed-", dir=destination.parent))
    try:
        (temporary / "weights").mkdir()
        entries: list[PackedBlockEntry] = []
        names: set[str] = set()
        for expected_index, (index, states) in enumerate(blocks):
            if index != expected_index:
                raise ValueError(
                    f"packed artifact blocks must be contiguous: expected {expected_index}, received {index}"
                )
            if not states:
                raise ValueError(f"packed artifact block {index} contains no layers")
            for state in states:
                if state.spec.name in names:
                    raise ValueError(f"packed artifact layer name is duplicated: {state.spec.name}")
                names.add(state.spec.name)
            entries.append(_write_block_shard(temporary, index, states))
            states.clear()
            del state, states
        if not entries:
            raise ValueError("packed artifact must contain at least one block")
        block_tuple = tuple(entries)
        manifest = PackedModelManifest(
            PACKED_DESCRIPTOR_SCHEMA_VERSION,
            PACKED_ARTIFACT_FORMAT,
            MINIMUM_RUNTIME_VERSION,
            PackedLayoutMetadata(),
            model,
            logical_descriptor_sha256,
            block_tuple,
            len(names),
            sum(block.bytes for block in block_tuple),
        )
        descriptor = temporary / "nanoquant-packed-model.json"
        with descriptor.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(asdict(manifest), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return open_packed_artifact(destination)


def write_packed_artifact(
    output: str | Path,
    model: RuntimeModelMetadata,
    logical_descriptor_sha256: str,
    blocks: Mapping[int, Sequence[PackedLayerState]],
) -> OpenPackedArtifact:
    indexes = sorted(blocks)
    if indexes != list(range(len(indexes))) or not indexes:
        raise ValueError(f"packed artifact blocks must be complete and contiguous: {indexes}")
    return write_packed_artifact_stream(
        output,
        model,
        logical_descriptor_sha256,
        ((index, list(blocks[index])) for index in indexes),
    )


def _tensor_from_payload(payload: object, path: str) -> PackedTensorEntry:
    value = _mapping(payload, path)
    return PackedTensorEntry(
        _string(value.get("role"), f"{path}.role"),
        _string(value.get("key"), f"{path}.key"),
        tuple(
            _integer(item, f"{path}.shape[{index}]")
            for index, item in enumerate(_sequence(value.get("shape"), f"{path}.shape"))
        ),
        _string(value.get("dtype"), f"{path}.dtype"),
    )


def _model_from_payload(payload: object, path: str) -> RuntimeModelMetadata:
    value = _mapping(payload, path)
    return RuntimeModelMetadata(
        _string(value.get("source"), f"{path}.source"),
        _string(value.get("revision"), f"{path}.revision"),
        _string(value.get("family"), f"{path}.family"),
        _string(value.get("config_hash"), f"{path}.config_hash"),
        _string(value.get("tokenizer_hash"), f"{path}.tokenizer_hash"),
    )


def _layout_from_payload(payload: object, path: str) -> PackedLayoutMetadata:
    value = _mapping(payload, path)
    reference_path = f"{path}.reference"
    reference = _mapping(value.get("reference"), reference_path)
    return PackedLayoutMetadata(
        _string(value.get("version"), f"{path}.version"),
        _string(value.get("word_dtype"), f"{path}.word_dtype"),
        _integer(value.get("word_bits"), f"{path}.word_bits"),
        _string(value.get("bit_order"), f"{path}.bit_order"),
        _integer(value.get("positive_bit"), f"{path}.positive_bit"),
        _integer(value.get("negative_bit"), f"{path}.negative_bit"),
        _integer(value.get("padding_bit"), f"{path}.padding_bit"),
        _integer(value.get("minimum_alignment_bytes"), f"{path}.minimum_alignment_bytes"),
        _integer(value.get("vector_alignment_bytes"), f"{path}.vector_alignment_bytes"),
        _string(value.get("tensor_namespace"), f"{path}.tensor_namespace"),
        _string(value.get("left_sidecar_name"), f"{path}.left_sidecar_name"),
        _string(value.get("right_sidecar_name"), f"{path}.right_sidecar_name"),
        _string(value.get("scale_pre_sidecar_name"), f"{path}.scale_pre_sidecar_name"),
        _string(value.get("scale_mid_sidecar_name"), f"{path}.scale_mid_sidecar_name"),
        _string(value.get("scale_post_sidecar_name"), f"{path}.scale_post_sidecar_name"),
        _string(value.get("outlier_index_sidecar_name"), f"{path}.outlier_index_sidecar_name"),
        _string(value.get("outlier_value_sidecar_name"), f"{path}.outlier_value_sidecar_name"),
        _string(value.get("outlier_scale_sidecar_name"), f"{path}.outlier_scale_sidecar_name"),
        _string(value.get("bias_storage"), f"{path}.bias_storage"),
        PackedReferenceProvenance(
            _string(reference.get("repository"), f"{reference_path}.repository"),
            _string(reference.get("commit"), f"{reference_path}.commit"),
            _string(
                reference.get("dirty_diff_git_object"),
                f"{reference_path}.dirty_diff_git_object",
            ),
            _string(reference.get("cuda_sha256"), f"{reference_path}.cuda_sha256"),
            _string(
                reference.get("converter_sha256"),
                f"{reference_path}.converter_sha256",
            ),
            _string(
                reference.get("documentation_sha256"),
                f"{reference_path}.documentation_sha256",
            ),
            _string(
                reference.get("model_loader_sha256"),
                f"{reference_path}.model_loader_sha256",
            ),
            _string(reference.get("cpu_sha256"), f"{reference_path}.cpu_sha256"),
        ),
    )


def _manifest_from_payload(payload: object) -> PackedModelManifest:
    value = _mapping(payload, "manifest")
    blocks = []
    for block_index, block_payload in enumerate(_sequence(value.get("blocks"), "manifest.blocks")):
        block_path = f"manifest.blocks[{block_index}]"
        block_value = _mapping(block_payload, block_path)
        layers = []
        for layer_index, layer_payload in enumerate(_sequence(block_value.get("layers"), f"{block_path}.layers")):
            layer_path = f"{block_path}.layers[{layer_index}]"
            layer_value = _mapping(layer_payload, layer_path)
            layers.append(
                PackedLayerEntry(
                    _spec_from_payload(layer_value.get("spec"), f"{layer_path}.spec"),
                    tuple(
                        _tensor_from_payload(item, f"{layer_path}.tensors[{tensor_index}]")
                        for tensor_index, item in enumerate(
                            _sequence(layer_value.get("tensors"), f"{layer_path}.tensors")
                        )
                    ),
                )
            )
        blocks.append(
            PackedBlockEntry(
                _integer(block_value.get("index"), f"{block_path}.index"),
                _string(block_value.get("path"), f"{block_path}.path"),
                _integer(block_value.get("bytes"), f"{block_path}.bytes"),
                _string(block_value.get("sha256"), f"{block_path}.sha256"),
                tuple(layers),
            )
        )
    return PackedModelManifest(
        _integer(value.get("schema_version"), "manifest.schema_version"),
        _string(value.get("artifact_format"), "manifest.artifact_format"),
        _string(value.get("minimum_runtime_version"), "manifest.minimum_runtime_version"),
        _layout_from_payload(value.get("layout"), "manifest.layout"),
        _model_from_payload(value.get("model"), "manifest.model"),
        _string(value.get("logical_descriptor_sha256"), "manifest.logical_descriptor_sha256"),
        tuple(blocks),
        _integer(value.get("layer_count"), "manifest.layer_count"),
        _integer(value.get("weight_bytes"), "manifest.weight_bytes"),
    )


def open_packed_artifact(
    root: str | Path,
    *,
    verify_hashes: bool = True,
) -> OpenPackedArtifact:
    """Inspect hashes and headers without loading packed tensor payloads."""

    artifact_root = Path(root)
    descriptor = artifact_root / "nanoquant-packed-model.json"
    if not descriptor.is_file():
        raise PackedArtifactError("packed artifact descriptor is missing")
    if descriptor.stat().st_size > _MAXIMUM_DESCRIPTOR_BYTES:
        raise PackedArtifactError("packed artifact descriptor exceeds the size limit")
    try:
        payload = cast(Any, json.loads(descriptor.read_text(encoding="utf-8")))
        manifest = _manifest_from_payload(payload)
    except PackedArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise PackedArtifactError(f"packed artifact descriptor is invalid: {error}") from error
    for block in manifest.blocks:
        shard = _resolved_member(artifact_root, block.path)
        if not shard.is_file() or shard.stat().st_size != block.bytes:
            raise PackedArtifactError(f"packed artifact shard size or presence differs: {block.path}")
        if verify_hashes and _hash_file(shard) != block.sha256:
            raise PackedArtifactError(f"packed artifact shard hash differs: {block.path}")
        declared = {tensor.key: tensor for layer in block.layers for tensor in layer.tensors}
        if len(declared) != sum(len(layer.tensors) for layer in block.layers):
            raise PackedArtifactError(f"packed artifact tensor key is duplicated: {block.path}")
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != set(declared):
                    raise PackedArtifactError(f"packed artifact shard tensor inventory differs: {block.path}")
                for key, tensor in declared.items():
                    view = handle.get_slice(key)
                    if tuple(view.get_shape()) != tensor.shape or _header_dtype(view.get_dtype()) != tensor.dtype:
                        raise PackedArtifactError(f"packed artifact tensor header differs: {block.path}:{key}")
        except PackedArtifactError:
            raise
        except Exception as error:
            raise PackedArtifactError(f"packed artifact shard header is invalid: {block.path}") from error
    return OpenPackedArtifact(artifact_root.resolve(), manifest)


def convert_logical_to_packed(
    logical_root: str | Path,
    output: str | Path,
) -> OpenPackedArtifact:
    """Stream a validated logical artifact into the concrete llama.cpp sign-word layout."""

    logical: OpenLogicalArtifact = open_logical_artifact(logical_root, verify_hashes=True)
    descriptor_hash = _hash_file(logical.root / "nanoquant-model.json")

    def blocks() -> Iterable[tuple[int, list[PackedLayerState]]]:
        for block in logical.manifest.blocks:
            states = [pack_logical_layer(logical.load_layer(layer.spec.name)) for layer in block.layers]
            yield block.index, states

    return write_packed_artifact_stream(
        output,
        logical.manifest.model,
        descriptor_hash,
        blocks(),
    )


def _logical_values(state: LogicalLayerState) -> tuple[torch.Tensor | None, ...]:
    return (
        state.left_binary,
        state.right_binary,
        state.scale_pre,
        state.scale_mid,
        state.scale_post,
        state.bias,
        state.outlier_indices,
        state.outlier_values,
        state.outlier_scales,
    )


def _validate_artifact_pair(
    logical: OpenLogicalArtifact,
    packed: OpenPackedArtifact,
) -> list[str]:
    descriptor_hash = _hash_file(logical.root / "nanoquant-model.json")
    if packed.manifest.logical_descriptor_sha256 != descriptor_hash:
        raise ValueError("packed artifact is bound to a different logical descriptor")
    if packed.manifest.model != logical.manifest.model:
        raise ValueError("packed and logical artifact model metadata differs")
    logical_names = [layer.spec.name for block in logical.manifest.blocks for layer in block.layers]
    packed_names = [layer.spec.name for block in packed.manifest.blocks for layer in block.layers]
    if packed_names != logical_names:
        raise ValueError("packed and logical layer inventory or ordering differs")
    return logical_names


def validate_packed_conversion(
    logical_root: str | Path,
    packed_root: str | Path,
) -> PackedConversionValidation:
    """Prove packed words and sidecars exactly reconstruct every source logical tensor."""

    logical = open_logical_artifact(logical_root, verify_hashes=True)
    packed = open_packed_artifact(packed_root, verify_hashes=True)
    logical_names = _validate_artifact_pair(logical, packed)
    logical_tensor_count = 0
    packed_tensor_count = 0
    for name in logical_names:
        source = logical.load_layer(name)
        packed_state = packed.load_layer(name)
        restored = packed_state.to_logical()
        if restored.spec != source.spec:
            raise ValueError(f"packed layer specification differs: {name}")
        for expected, actual in zip(_logical_values(source), _logical_values(restored), strict=True):
            if (expected is None) != (actual is None):
                raise ValueError(f"packed layer tensor presence differs: {name}")
            if expected is not None and actual is not None:
                if not torch.equal(expected, actual):
                    raise ValueError(f"packed layer tensor differs: {name}")
                logical_tensor_count += 1
        packed_tensor_count += len(_state_tensors(packed_state))
    return PackedConversionValidation(
        logical.root,
        packed.root,
        len(packed.manifest.blocks),
        packed.manifest.layer_count,
        logical_tensor_count,
        packed_tensor_count,
        logical.manifest.weight_bytes,
        packed.manifest.weight_bytes,
        packed.manifest.weight_bytes / logical.manifest.weight_bytes,
        True,
    )


def _input_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError as error:
        raise ValueError(f"packed reference validation input dtype is unsupported: {name}") from error


def validate_packed_reference_parity(
    logical_root: str | Path,
    packed_root: str | Path,
    *,
    absolute_tolerance: float = 0.0,
) -> PackedReferenceParityResult:
    """Execute logical and packed reference backends across every artifact layer."""

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("packed parity absolute tolerance must be finite and non-negative")
    logical = open_logical_artifact(logical_root, verify_hashes=True)
    packed = open_packed_artifact(packed_root, verify_hashes=True)
    _validate_artifact_pair(logical, packed)
    logical_backend = FactorizedReferenceBackend()
    packed_backend = PackedReferenceBackend()
    maximum_error = 0.0
    maximum_layer = ""
    output_elements = 0
    with torch.no_grad():
        for block in logical.manifest.blocks:
            for entry in block.layers:
                source = logical.load_layer(entry.spec.name)
                packed_state = packed.load_layer(entry.spec.name)
                value = (
                    torch.linspace(
                        -0.5,
                        0.5,
                        source.spec.in_features,
                        dtype=torch.float32,
                    )
                    .to(_input_dtype(source.spec.scale_dtype))
                    .reshape(1, -1)
                )
                expected = logical_backend.linear(
                    value,
                    logical_backend.prepare(source, "cpu"),
                ).float()
                actual = packed_backend.linear(
                    value,
                    packed_backend.prepare(packed_state, "cpu"),
                ).float()
                if not bool(torch.all(torch.isfinite(actual))):
                    raise ValueError(f"packed reference produced a non-finite output: {entry.spec.name}")
                error = float(torch.max(torch.abs(expected - actual)).item())
                if error > maximum_error or not maximum_layer:
                    maximum_error = error
                    maximum_layer = entry.spec.name
                output_elements += actual.numel()
    if maximum_error > absolute_tolerance:
        raise ValueError(
            f"packed reference differs beyond tolerance: {maximum_error} > {absolute_tolerance} at {maximum_layer}"
        )
    return PackedReferenceParityResult(
        logical.root,
        packed.root,
        packed.manifest.layer_count,
        output_elements,
        maximum_error,
        maximum_layer,
    )
