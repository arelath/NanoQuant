"""Versioned, block-sharded logical NanoQuant artifacts for deployment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.runtime.backend import DeviceLike, ProjectionMemberSpec, QuantizedLinearSpec
from nanoquant.runtime.logical import LogicalLayerState, canonical_torch_dtype

DESCRIPTOR_SCHEMA_VERSION = 1
LOGICAL_FORMAT_VERSION = "nanoquant-v1"
MINIMUM_RUNTIME_VERSION = "0.1.0"
_MAXIMUM_DESCRIPTOR_BYTES = 16 * 1024 * 1024
_TENSOR_ROLES = (
    "factor_left",
    "factor_right",
    "scale_pre",
    "scale_mid",
    "scale_post",
    "bias",
    "outlier_indices",
    "outlier_values",
    "outlier_scales",
)
_SAFE_DTYPE_NAMES = {
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
}


class LogicalArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeModelMetadata:
    source: str
    revision: str
    family: str
    config_hash: str
    tokenizer_hash: str

    def __post_init__(self) -> None:
        if not all((self.source, self.revision, self.family, self.config_hash, self.tokenizer_hash)):
            raise ValueError("runtime model metadata fields must be non-empty")


@dataclass(frozen=True, slots=True)
class LogicalTensorEntry:
    role: str
    key: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if self.role not in _TENSOR_ROLES:
            raise ValueError(f"unsupported logical tensor role: {self.role}")
        if not self.key or self.key.startswith("/"):
            raise ValueError("logical tensor key must be non-empty and relative")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("logical tensor dimensions must be non-negative")
        if not self.dtype:
            raise ValueError("logical tensor dtype must be non-empty")


@dataclass(frozen=True, slots=True)
class LogicalLayerEntry:
    spec: QuantizedLinearSpec
    tensors: tuple[LogicalTensorEntry, ...]

    def __post_init__(self) -> None:
        roles = [tensor.role for tensor in self.tensors]
        if len(roles) != len(set(roles)):
            raise ValueError(f"logical layer tensor roles are duplicated: {self.spec.name}")
        expected = {
            "factor_left",
            "factor_right",
            "scale_pre",
            "scale_mid",
            "scale_post",
        }
        if self.spec.has_bias:
            expected.add("bias")
        if self.spec.outlier_count:
            expected.update(("outlier_indices", "outlier_values"))
        if self.spec.has_outlier_scales:
            expected.add("outlier_scales")
        if set(roles) != expected:
            raise ValueError(f"logical layer tensor inventory differs from its specification: {self.spec.name}")


@dataclass(frozen=True, slots=True)
class LogicalBlockEntry:
    index: int
    path: str
    bytes: int
    sha256: str
    layers: tuple[LogicalLayerEntry, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("logical block index must be non-negative")
        if self.bytes <= 0:
            raise ValueError("logical block shard must be non-empty")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("logical block hash must be a lowercase SHA-256 digest")
        _validate_relative_path(self.path)
        names = [layer.spec.name for layer in self.layers]
        if not names or len(names) != len(set(names)):
            raise ValueError("logical block must contain uniquely named layers")


@dataclass(frozen=True, slots=True)
class LogicalModelManifest:
    schema_version: int
    artifact_format: str
    minimum_runtime_version: str
    logical_format: str
    model: RuntimeModelMetadata
    blocks: tuple[LogicalBlockEntry, ...]
    layer_count: int
    weight_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError(f"unsupported logical artifact schema: {self.schema_version}")
        if self.artifact_format != "nanoquant-logical-model":
            raise ValueError(f"unsupported logical artifact format: {self.artifact_format}")
        if self.minimum_runtime_version != MINIMUM_RUNTIME_VERSION:
            raise ValueError(f"unsupported minimum runtime version: {self.minimum_runtime_version}")
        if self.logical_format != LOGICAL_FORMAT_VERSION:
            raise ValueError(f"unsupported logical format: {self.logical_format}")
        if not self.blocks:
            raise ValueError("logical artifact must contain at least one block")
        indexes = [block.index for block in self.blocks]
        if indexes != list(range(len(self.blocks))):
            raise ValueError(f"logical artifact blocks are not contiguous: {indexes}")
        names = [layer.spec.name for block in self.blocks for layer in block.layers]
        if len(names) != len(set(names)) or len(names) != self.layer_count:
            raise ValueError("logical artifact layer count or uniqueness is inconsistent")
        if sum(block.bytes for block in self.blocks) != self.weight_bytes:
            raise ValueError("logical artifact weight byte count is inconsistent")
        if any(layer.spec.logical_format != self.logical_format for block in self.blocks for layer in block.layers):
            raise ValueError("logical artifact contains a layer with a different format")


@dataclass(frozen=True, slots=True)
class OpenLogicalArtifact:
    root: Path
    manifest: LogicalModelManifest

    def load_layer(self, name: str, device: DeviceLike = "cpu") -> LogicalLayerState:
        matches = [
            (block, layer) for block in self.manifest.blocks for layer in block.layers if layer.spec.name == name
        ]
        if len(matches) != 1:
            raise KeyError(f"logical artifact layer not found: {name}")
        block, layer = matches[0]
        by_role: dict[str, torch.Tensor] = {}
        with safe_open(self.root / block.path, framework="pt", device=str(torch.device(device))) as handle:
            for tensor in layer.tensors:
                by_role[tensor.role] = handle.get_tensor(tensor.key)
        return LogicalLayerState(
            layer.spec,
            by_role["factor_left"],
            by_role["factor_right"],
            by_role["scale_pre"],
            by_role["scale_mid"],
            by_role["scale_post"],
            by_role.get("bias"),
            by_role.get("outlier_indices"),
            by_role.get("outlier_values"),
            by_role.get("outlier_scales"),
        )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"logical artifact path is not canonical and relative: {value!r}")


def _resolved_member(root: Path, relative: str) -> Path:
    _validate_relative_path(relative)
    resolved_root = root.resolve()
    candidate = (root / relative).resolve()
    if resolved_root not in candidate.parents:
        raise LogicalArtifactError(f"logical artifact member escapes its root: {relative}")
    return candidate


def _state_tensors(state: LogicalLayerState) -> tuple[tuple[str, torch.Tensor], ...]:
    values = [
        ("factor_left", state.left_binary),
        ("factor_right", state.right_binary),
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
    states: Sequence[LogicalLayerState],
) -> LogicalBlockEntry:
    tensors: dict[str, torch.Tensor] = {}
    layer_entries: list[LogicalLayerEntry] = []
    for state in states:
        entries: list[LogicalTensorEntry] = []
        for role, value in _state_tensors(state):
            key = f"{state.spec.name}.{role}"
            if key in tensors:
                raise ValueError(f"logical artifact tensor key is duplicated: {key}")
            copied = value.detach().cpu().contiguous()
            tensors[key] = copied
            entries.append(LogicalTensorEntry(role, key, tuple(copied.shape), canonical_torch_dtype(copied.dtype)))
        layer_entries.append(LogicalLayerEntry(state.spec, tuple(entries)))
    relative = f"weights/block-{index:05d}.safetensors"
    shard = temporary / relative
    save_file(tensors, shard)
    return LogicalBlockEntry(
        index,
        relative,
        shard.stat().st_size,
        _hash_file(shard),
        tuple(layer_entries),
    )


def write_logical_artifact(
    output: str | Path,
    model: RuntimeModelMetadata,
    blocks: Mapping[int, Sequence[LogicalLayerState]],
) -> OpenLogicalArtifact:
    """Atomically write a complete in-memory block mapping."""

    indexes = sorted(blocks)
    if indexes != list(range(len(indexes))) or not indexes:
        raise ValueError(f"logical artifact blocks must be complete and contiguous: {indexes}")
    return write_logical_artifact_stream(
        output,
        model,
        ((index, list(blocks[index])) for index in indexes),
    )


def write_logical_artifact_stream(
    output: str | Path,
    model: RuntimeModelMetadata,
    blocks: Iterable[tuple[int, Sequence[LogicalLayerState]]],
) -> OpenLogicalArtifact:
    """Atomically consume state sequences while retaining at most one logical block."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"logical artifact output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".nanoquant-logical-", dir=destination.parent))
    try:
        weights = temporary / "weights"
        weights.mkdir()
        block_entries: list[LogicalBlockEntry] = []
        names: set[str] = set()
        iterator = iter(blocks)
        expected_index = 0
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                break
            index, states = item
            if index != expected_index:
                raise ValueError(
                    "logical artifact blocks must be complete and contiguous: "
                    f"expected {expected_index}, received {index}"
                )
            if not states:
                raise ValueError(f"logical artifact block {index} contains no layers")
            for state in states:
                if state.spec.name in names:
                    raise ValueError(f"logical artifact layer name is duplicated: {state.spec.name}")
                names.add(state.spec.name)
            block_entries.append(_write_block_shard(temporary, index, states))
            del item, state, states
            expected_index += 1
        if not block_entries:
            raise ValueError("logical artifact must contain at least one block")
        block_tuple = tuple(block_entries)
        manifest = LogicalModelManifest(
            DESCRIPTOR_SCHEMA_VERSION,
            "nanoquant-logical-model",
            MINIMUM_RUNTIME_VERSION,
            LOGICAL_FORMAT_VERSION,
            model,
            block_tuple,
            len(names),
            sum(block.bytes for block in block_tuple),
        )
        descriptor = temporary / "nanoquant-model.json"
        with descriptor.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(asdict(manifest), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return open_logical_artifact(destination)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise LogicalArtifactError(f"{path} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise LogicalArtifactError(f"{path} must be an array")
    return cast(Sequence[object], value)


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise LogicalArtifactError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LogicalArtifactError(f"{path} must be an integer")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise LogicalArtifactError(f"{path} must be a boolean")
    return value


def _optional_string(value: object, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _spec_from_payload(payload: object, path: str) -> QuantizedLinearSpec:
    value = _mapping(payload, path)
    members_payload = value.get("members", [])
    members = tuple(
        ProjectionMemberSpec(
            _string(member.get("name"), f"{path}.members[{index}].name"),
            _integer(member.get("row_start"), f"{path}.members[{index}].row_start"),
            _integer(member.get("row_end"), f"{path}.members[{index}].row_end"),
        )
        for index, raw_member in enumerate(_sequence(members_payload, f"{path}.members"))
        for member in (_mapping(raw_member, f"{path}.members[{index}]"),)
    )
    return QuantizedLinearSpec(
        _string(value.get("name"), f"{path}.name"),
        _string(value.get("logical_format"), f"{path}.logical_format"),
        _integer(value.get("in_features"), f"{path}.in_features"),
        _integer(value.get("out_features"), f"{path}.out_features"),
        _integer(value.get("rank"), f"{path}.rank"),
        _string(value.get("factor_dtype"), f"{path}.factor_dtype"),
        _string(value.get("scale_dtype"), f"{path}.scale_dtype"),
        _integer(value.get("outlier_count"), f"{path}.outlier_count"),
        _optional_string(value.get("outlier_value_dtype"), f"{path}.outlier_value_dtype"),
        _boolean(value.get("has_outlier_scales"), f"{path}.has_outlier_scales"),
        _boolean(value.get("has_bias"), f"{path}.has_bias"),
        members,
    )


def _tensor_from_payload(payload: object, path: str) -> LogicalTensorEntry:
    value = _mapping(payload, path)
    shape = tuple(
        _integer(item, f"{path}.shape[{index}]")
        for index, item in enumerate(_sequence(value.get("shape"), f"{path}.shape"))
    )
    return LogicalTensorEntry(
        _string(value.get("role"), f"{path}.role"),
        _string(value.get("key"), f"{path}.key"),
        shape,
        _string(value.get("dtype"), f"{path}.dtype"),
    )


def _manifest_from_payload(payload: object) -> LogicalModelManifest:
    value = _mapping(payload, "manifest")
    model_value = _mapping(value.get("model"), "manifest.model")
    model = RuntimeModelMetadata(
        _string(model_value.get("source"), "manifest.model.source"),
        _string(model_value.get("revision"), "manifest.model.revision"),
        _string(model_value.get("family"), "manifest.model.family"),
        _string(model_value.get("config_hash"), "manifest.model.config_hash"),
        _string(model_value.get("tokenizer_hash"), "manifest.model.tokenizer_hash"),
    )
    blocks = []
    for block_index, block_payload in enumerate(_sequence(value.get("blocks"), "manifest.blocks")):
        block_value = _mapping(block_payload, f"manifest.blocks[{block_index}]")
        layers = []
        for layer_index, layer_payload in enumerate(
            _sequence(block_value.get("layers"), f"manifest.blocks[{block_index}].layers")
        ):
            layer_value = _mapping(
                layer_payload,
                f"manifest.blocks[{block_index}].layers[{layer_index}]",
            )
            tensor_path = f"manifest.blocks[{block_index}].layers[{layer_index}].tensors"
            layers.append(
                LogicalLayerEntry(
                    _spec_from_payload(
                        layer_value.get("spec"),
                        f"manifest.blocks[{block_index}].layers[{layer_index}].spec",
                    ),
                    tuple(
                        _tensor_from_payload(item, f"{tensor_path}[{tensor_index}]")
                        for tensor_index, item in enumerate(_sequence(layer_value.get("tensors"), tensor_path))
                    ),
                )
            )
        blocks.append(
            LogicalBlockEntry(
                _integer(block_value.get("index"), f"manifest.blocks[{block_index}].index"),
                _string(block_value.get("path"), f"manifest.blocks[{block_index}].path"),
                _integer(block_value.get("bytes"), f"manifest.blocks[{block_index}].bytes"),
                _string(block_value.get("sha256"), f"manifest.blocks[{block_index}].sha256"),
                tuple(layers),
            )
        )
    return LogicalModelManifest(
        _integer(value.get("schema_version"), "manifest.schema_version"),
        _string(value.get("artifact_format"), "manifest.artifact_format"),
        _string(value.get("minimum_runtime_version"), "manifest.minimum_runtime_version"),
        _string(value.get("logical_format"), "manifest.logical_format"),
        model,
        tuple(blocks),
        _integer(value.get("layer_count"), "manifest.layer_count"),
        _integer(value.get("weight_bytes"), "manifest.weight_bytes"),
    )


def _header_dtype(value: object) -> str:
    name = str(value)
    try:
        return _SAFE_DTYPE_NAMES[name]
    except KeyError as error:
        raise LogicalArtifactError(f"unsupported safetensors dtype in logical artifact: {name}") from error


def open_logical_artifact(
    root: str | Path,
    *,
    verify_hashes: bool = True,
) -> OpenLogicalArtifact:
    """Validate descriptor, members, and safetensors headers without loading tensor payloads."""

    artifact_root = Path(root)
    descriptor = artifact_root / "nanoquant-model.json"
    if not descriptor.is_file():
        raise LogicalArtifactError("logical artifact descriptor is missing")
    if descriptor.stat().st_size > _MAXIMUM_DESCRIPTOR_BYTES:
        raise LogicalArtifactError("logical artifact descriptor exceeds the size limit")
    try:
        payload = cast(Any, json.loads(descriptor.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LogicalArtifactError("logical artifact descriptor is not valid UTF-8 JSON") from error
    try:
        manifest = _manifest_from_payload(payload)
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, LogicalArtifactError):
            raise
        raise LogicalArtifactError(f"logical artifact descriptor is invalid: {error}") from error
    for block in manifest.blocks:
        shard = _resolved_member(artifact_root, block.path)
        if not shard.is_file() or shard.stat().st_size != block.bytes:
            raise LogicalArtifactError(f"logical artifact shard size or presence differs: {block.path}")
        if verify_hashes and _hash_file(shard) != block.sha256:
            raise LogicalArtifactError(f"logical artifact shard hash differs: {block.path}")
        declared = {tensor.key: tensor for layer in block.layers for tensor in layer.tensors}
        if len(declared) != sum(len(layer.tensors) for layer in block.layers):
            raise LogicalArtifactError(f"logical artifact tensor key is duplicated: {block.path}")
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if keys != set(declared):
                    raise LogicalArtifactError(f"logical artifact shard tensor inventory differs: {block.path}")
                for key, tensor in declared.items():
                    view = handle.get_slice(key)
                    if tuple(view.get_shape()) != tensor.shape or _header_dtype(view.get_dtype()) != tensor.dtype:
                        raise LogicalArtifactError(f"logical artifact tensor header differs: {block.path}:{key}")
        except LogicalArtifactError:
            raise
        except Exception as error:
            raise LogicalArtifactError(f"logical artifact shard header is invalid: {block.path}") from error
    return OpenLogicalArtifact(artifact_root.resolve(), manifest)
