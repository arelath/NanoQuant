"""Base-bound block shards for optional compact mixed-V layer replacements."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch

from nanoquant.runtime.artifact import (
    MINIMUM_RUNTIME_VERSION,
    _hash_file,
    _header_dtype,
    _resolved_member,
    _validate_relative_path,
)
from nanoquant.runtime.backend import DeviceLike, QuantizedLinearSpec, RuntimeLayerState
from nanoquant.runtime.codec import decode_dataclass
from nanoquant.runtime.io_utils import atomic_output_directory
from nanoquant.runtime.logical import canonical_torch_dtype
from nanoquant.runtime.mixed_v import (
    MIXED_V_CODEBOOK_SIZE,
    MIXED_V_CORRECTION_BITS,
    MIXED_V_CORRECTION_PAIR_COUNT,
    MIXED_V_FORMAT_VERSION,
    MIXED_V_INDEX_BITS,
    MIXED_V_RECORD_BITS,
    MixedVLayerState,
    mixed_v_payload_word_count,
)
from nanoquant.runtime.packed import PACKED_WORD_BITS, PackedLayerState, packed_word_count
from nanoquant.runtime.packed_artifact import OpenPackedArtifact, open_packed_artifact
from nanoquant.runtime.safetensors_io import SAFETENSORS, load_tensors

MIXED_V_ARTIFACT_SCHEMA_VERSION = 1
MIXED_V_ARTIFACT_FORMAT = "nanoquant-mixed-v-overlay"
MIXED_V_DESCRIPTOR = "nanoquant-mixed-v-overlay.json"
MIXED_V_TENSOR_NAMESPACE = f"layouts.{MIXED_V_FORMAT_VERSION}"
_MAXIMUM_DESCRIPTOR_BYTES = 16 * 1024 * 1024
_TENSOR_ROLES = (
    "factor_left_words",
    "factor_right_free_words",
    "factor_right_coded_payload",
    "factor_right_codebook_words",
    "scale_pre",
    "scale_mid",
    "scale_post",
    "bias",
    "outlier_indices",
    "outlier_values",
    "outlier_scales",
    "patch_left",
    "patch_right",
)


class MixedVArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MixedVLayoutMetadata:
    version: str = MIXED_V_FORMAT_VERSION
    word_bits: int = PACKED_WORD_BITS
    index_bits: int = MIXED_V_INDEX_BITS
    correction_bits: int = MIXED_V_CORRECTION_BITS
    record_bits: int = MIXED_V_RECORD_BITS
    codebook_size: int = MIXED_V_CODEBOOK_SIZE
    correction_pair_count: int = MIXED_V_CORRECTION_PAIR_COUNT
    correction_mapping: str = "lexicographic-unordered-distinct-bit-pairs"
    tensor_namespace: str = MIXED_V_TENSOR_NAMESPACE
    runtime_preparation: str = "expand-right-factor-to-llama.cpp-i32-lsb-v1"

    def __post_init__(self) -> None:
        if (
            self.version,
            self.word_bits,
            self.index_bits,
            self.correction_bits,
            self.record_bits,
            self.codebook_size,
            self.correction_pair_count,
            self.correction_mapping,
            self.tensor_namespace,
            self.runtime_preparation,
        ) != (
            MIXED_V_FORMAT_VERSION,
            PACKED_WORD_BITS,
            MIXED_V_INDEX_BITS,
            MIXED_V_CORRECTION_BITS,
            MIXED_V_RECORD_BITS,
            MIXED_V_CODEBOOK_SIZE,
            MIXED_V_CORRECTION_PAIR_COUNT,
            "lexicographic-unordered-distinct-bit-pairs",
            MIXED_V_TENSOR_NAMESPACE,
            "expand-right-factor-to-llama.cpp-i32-lsb-v1",
        ):
            raise ValueError("mixed-V layout metadata differs from schema 1")


@dataclass(frozen=True, slots=True)
class MixedVTensorEntry:
    role: str
    key: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if self.role not in _TENSOR_ROLES:
            raise ValueError(f"unsupported mixed-V tensor role: {self.role}")
        if not self.key or self.key.startswith("/"):
            raise ValueError("mixed-V tensor key must be non-empty and relative")
        if any(dimension < 0 for dimension in self.shape) or not self.dtype:
            raise ValueError("mixed-V tensor metadata is invalid")


def _optional_metadata(
    spec: QuantizedLinearSpec,
) -> dict[str, tuple[tuple[int, ...], str]]:
    expected: dict[str, tuple[tuple[int, ...], str]] = {}
    if spec.has_bias:
        expected["bias"] = ((spec.out_features,), spec.bias_dtype or spec.scale_dtype)
    if spec.outlier_count:
        expected["outlier_indices"] = ((spec.outlier_count,), "int32")
        expected["outlier_values"] = (
            (spec.out_features, spec.outlier_count),
            cast(str, spec.outlier_value_dtype),
        )
    if spec.has_outlier_scales:
        expected["outlier_scales"] = ((spec.outlier_count,), spec.scale_dtype)
    if spec.patch_rank:
        expected["patch_left"] = (
            (spec.out_features, spec.patch_rank),
            cast(str, spec.patch_value_dtype),
        )
        expected["patch_right"] = (
            (spec.patch_rank, spec.in_features),
            cast(str, spec.patch_value_dtype),
        )
    return expected


def _expected_metadata(
    spec: QuantizedLinearSpec,
    free_rows: int,
) -> dict[str, tuple[tuple[int, ...], str]]:
    words_per_row = packed_word_count(spec.in_features)
    record_count = (spec.rank - free_rows) * words_per_row
    expected = {
        "factor_left_words": (
            (spec.out_features, packed_word_count(spec.rank)),
            "int32",
        ),
        "factor_right_free_words": ((free_rows, words_per_row), "int32"),
        "factor_right_coded_payload": (
            (mixed_v_payload_word_count(record_count),),
            "int32",
        ),
        "factor_right_codebook_words": ((MIXED_V_CODEBOOK_SIZE,), "int32"),
        "scale_pre": ((spec.in_features,), spec.scale_dtype),
        "scale_mid": ((spec.rank,), spec.scale_dtype),
        "scale_post": ((spec.out_features,), spec.scale_dtype),
    }
    expected.update(_optional_metadata(spec))
    return expected


@dataclass(frozen=True, slots=True)
class MixedVLayerEntry:
    spec: QuantizedLinearSpec
    free_rows: int
    tensors: tuple[MixedVTensorEntry, ...]

    def __post_init__(self) -> None:
        by_role = {tensor.role: tensor for tensor in self.tensors}
        if len(by_role) != len(self.tensors):
            raise ValueError(f"mixed-V tensor roles are duplicated: {self.spec.name}")
        expected = _expected_metadata(self.spec, self.free_rows)
        if set(by_role) != set(expected):
            raise ValueError(f"mixed-V tensor inventory differs: {self.spec.name}")
        for role, (shape, dtype) in expected.items():
            tensor = by_role[role]
            if tensor.key != f"{MIXED_V_TENSOR_NAMESPACE}.{self.spec.name}.{role}":
                raise ValueError(f"mixed-V tensor key differs: {self.spec.name}:{role}")
            if tensor.shape != shape or tensor.dtype != dtype:
                raise ValueError(f"mixed-V tensor metadata differs: {self.spec.name}:{role}")


@dataclass(frozen=True, slots=True)
class MixedVBlockEntry:
    index: int
    path: str
    bytes: int
    sha256: str
    layers: tuple[MixedVLayerEntry, ...]

    def __post_init__(self) -> None:
        if self.index < 0 or self.bytes <= 0:
            raise ValueError("mixed-V block index or byte count is invalid")
        _validate_relative_path(self.path)
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("mixed-V block hash must be a lowercase SHA-256 digest")
        names = tuple(layer.spec.name for layer in self.layers)
        if not names or len(names) != len(set(names)):
            raise ValueError("mixed-V block must contain uniquely named layers")


@dataclass(frozen=True, slots=True)
class MixedVArtifactManifest:
    schema_version: int
    artifact_format: str
    minimum_runtime_version: str
    layout: MixedVLayoutMetadata
    base_packed_descriptor_sha256: str
    blocks: tuple[MixedVBlockEntry, ...]
    layer_count: int
    artifact_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != MIXED_V_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported mixed-V artifact schema: {self.schema_version}")
        if self.artifact_format != MIXED_V_ARTIFACT_FORMAT:
            raise ValueError(f"unsupported mixed-V artifact format: {self.artifact_format}")
        if self.minimum_runtime_version != MINIMUM_RUNTIME_VERSION:
            raise ValueError(
                f"unsupported mixed-V minimum runtime version: {self.minimum_runtime_version}"
            )
        if len(self.base_packed_descriptor_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.base_packed_descriptor_sha256
        ):
            raise ValueError("mixed-V base descriptor hash must be lowercase SHA-256")
        indexes = tuple(block.index for block in self.blocks)
        if not indexes or tuple(sorted(indexes)) != indexes or len(indexes) != len(set(indexes)):
            raise ValueError("mixed-V replacement block indexes must be sorted and unique")
        names = tuple(layer.spec.name for block in self.blocks for layer in block.layers)
        if len(names) != len(set(names)) or len(names) != self.layer_count:
            raise ValueError("mixed-V replacement layer count or uniqueness is inconsistent")
        if sum(block.bytes for block in self.blocks) != self.artifact_bytes:
            raise ValueError("mixed-V artifact byte count is inconsistent")


@dataclass(frozen=True, slots=True)
class OpenMixedVArtifact:
    root: Path
    base: OpenPackedArtifact
    manifest: MixedVArtifactManifest

    @property
    def replacement_names(self) -> frozenset[str]:
        return frozenset(
            layer.spec.name for block in self.manifest.blocks for layer in block.layers
        )

    def layer_specs(self) -> tuple[QuantizedLinearSpec, ...]:
        replacements = {
            layer.spec.name: layer.spec
            for block in self.manifest.blocks
            for layer in block.layers
        }
        return tuple(
            replacements.get(layer.spec.name, layer.spec)
            for block in self.base.manifest.blocks
            for layer in block.layers
        )

    def load_compact_layer(
        self,
        name: str,
        device: DeviceLike = "cpu",
    ) -> MixedVLayerState:
        matches = [
            (block, layer)
            for block in self.manifest.blocks
            for layer in block.layers
            if layer.spec.name == name
        ]
        if len(matches) != 1:
            raise KeyError(f"mixed-V replacement layer not found: {name}")
        block, layer = matches[0]
        loaded = load_tensors(
            self.root / block.path,
            (tensor.key for tensor in layer.tensors),
            device=device,
        )
        by_role = {tensor.role: loaded[tensor.key] for tensor in layer.tensors}
        return MixedVLayerState(
            layer.spec,
            self.manifest.layout.version,
            layer.free_rows,
            by_role["factor_left_words"],
            by_role["factor_right_free_words"],
            by_role["factor_right_coded_payload"],
            by_role["factor_right_codebook_words"],
            by_role["scale_pre"],
            by_role["scale_mid"],
            by_role["scale_post"],
            by_role.get("bias"),
            by_role.get("outlier_indices"),
            by_role.get("outlier_values"),
            by_role.get("outlier_scales"),
            by_role.get("patch_left"),
            by_role.get("patch_right"),
        )

    def load_runtime_layer(
        self,
        name: str,
        device: DeviceLike = "cpu",
    ) -> RuntimeLayerState:
        if name in self.replacement_names:
            return self.load_compact_layer(name, device)
        return self.base.load_layer(name, device)

    def load_packed_layer(
        self,
        name: str,
        device: DeviceLike = "cpu",
    ) -> PackedLayerState:
        state = self.load_runtime_layer(name, device)
        if isinstance(state, MixedVLayerState):
            return state.to_packed()
        if not isinstance(state, PackedLayerState):
            raise TypeError("mixed-V artifact base returned an unsupported runtime state")
        return state


def _state_tensors(state: MixedVLayerState) -> tuple[tuple[str, torch.Tensor], ...]:
    values = [
        ("factor_left_words", state.left_words),
        ("factor_right_free_words", state.free_right_words),
        ("factor_right_coded_payload", state.coded_payload),
        ("factor_right_codebook_words", state.codebook_words),
        ("scale_pre", state.scale_pre),
        ("scale_mid", state.scale_mid),
        ("scale_post", state.scale_post),
    ]
    optional = (
        ("bias", state.bias),
        ("outlier_indices", state.outlier_indices),
        ("outlier_values", state.outlier_values),
        ("outlier_scales", state.outlier_scales),
        ("patch_left", state.patch_left),
        ("patch_right", state.patch_right),
    )
    values.extend((role, value) for role, value in optional if value is not None)
    return tuple(values)


def _base_entries(
    base: OpenPackedArtifact,
) -> tuple[dict[str, tuple[int, QuantizedLinearSpec]], dict[int, set[str]]]:
    by_name: dict[str, tuple[int, QuantizedLinearSpec]] = {}
    by_block: dict[int, set[str]] = {}
    for block in base.manifest.blocks:
        by_block[block.index] = set()
        for layer in block.layers:
            by_name[layer.spec.name] = (block.index, layer.spec)
            by_block[block.index].add(layer.spec.name)
    return by_name, by_block


def _validate_replacement_spec(
    spec: QuantizedLinearSpec,
    block_index: int,
    base_by_name: Mapping[str, tuple[int, QuantizedLinearSpec]],
) -> None:
    if spec.name not in base_by_name:
        raise ValueError(f"mixed-V replacement is absent from the base artifact: {spec.name}")
    expected_block, base_spec = base_by_name[spec.name]
    if expected_block != block_index:
        raise ValueError(f"mixed-V replacement uses the wrong source block: {spec.name}")
    if replace(spec, rank=base_spec.rank) != base_spec:
        raise ValueError(
            f"mixed-V replacement may change only rank and rank-shaped tensors: {spec.name}"
        )


def write_mixed_v_artifact(
    output: str | Path,
    base: str | Path | OpenPackedArtifact,
    replacements: Mapping[int, Sequence[MixedVLayerState]],
) -> OpenMixedVArtifact:
    """Write selected compact replacements without copying ordinary packed layers."""

    opened_base = (
        base if isinstance(base, OpenPackedArtifact) else open_packed_artifact(base, verify_hashes=True)
    )
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"mixed-V artifact output already exists: {destination}")
    indexes = tuple(sorted(replacements))
    if not indexes or any(index < 0 for index in indexes):
        raise ValueError("mixed-V artifact requires at least one non-negative replacement block")
    base_by_name, _ = _base_entries(opened_base)
    names: set[str] = set()
    with atomic_output_directory(destination, prefix=".nanoquant-mixed-v-") as temporary:
        (temporary / "weights").mkdir()
        blocks: list[MixedVBlockEntry] = []
        for index in indexes:
            states = tuple(replacements[index])
            if not states:
                raise ValueError(f"mixed-V replacement block {index} contains no layers")
            tensors: dict[str, torch.Tensor] = {}
            layer_entries: list[MixedVLayerEntry] = []
            for state in states:
                _validate_replacement_spec(state.spec, index, base_by_name)
                if state.spec.name in names:
                    raise ValueError(f"mixed-V replacement layer is duplicated: {state.spec.name}")
                names.add(state.spec.name)
                tensor_entries: list[MixedVTensorEntry] = []
                for role, value in _state_tensors(state):
                    key = f"{MIXED_V_TENSOR_NAMESPACE}.{state.spec.name}.{role}"
                    copied = value.detach().cpu().contiguous()
                    tensors[key] = copied
                    tensor_entries.append(
                        MixedVTensorEntry(
                            role,
                            key,
                            tuple(copied.shape),
                            canonical_torch_dtype(copied.dtype),
                        )
                    )
                layer_entries.append(
                    MixedVLayerEntry(state.spec, state.free_rows, tuple(tensor_entries))
                )
            relative = f"weights/block-{index:05d}.safetensors"
            shard = temporary / relative
            SAFETENSORS.save(tensors, shard)
            blocks.append(
                MixedVBlockEntry(
                    index,
                    relative,
                    shard.stat().st_size,
                    _hash_file(shard),
                    tuple(layer_entries),
                )
            )
        block_tuple = tuple(blocks)
        manifest = MixedVArtifactManifest(
            MIXED_V_ARTIFACT_SCHEMA_VERSION,
            MIXED_V_ARTIFACT_FORMAT,
            MINIMUM_RUNTIME_VERSION,
            MixedVLayoutMetadata(),
            _hash_file(opened_base.root / "nanoquant-packed-model.json"),
            block_tuple,
            len(names),
            sum(block.bytes for block in block_tuple),
        )
        descriptor = temporary / MIXED_V_DESCRIPTOR
        with descriptor.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(asdict(manifest), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return open_mixed_v_artifact(destination, opened_base)


def open_mixed_v_artifact(
    root: str | Path,
    base: str | Path | OpenPackedArtifact,
    *,
    verify_hashes: bool = True,
) -> OpenMixedVArtifact:
    """Validate an overlay and its exact base binding without eager tensor reads."""

    artifact_root = Path(root)
    descriptor = artifact_root / MIXED_V_DESCRIPTOR
    if not descriptor.is_file():
        raise MixedVArtifactError("mixed-V artifact descriptor is missing")
    if descriptor.stat().st_size > _MAXIMUM_DESCRIPTOR_BYTES:
        raise MixedVArtifactError("mixed-V artifact descriptor exceeds the size limit")
    opened_base = (
        base
        if isinstance(base, OpenPackedArtifact)
        else open_packed_artifact(base, verify_hashes=verify_hashes)
    )
    try:
        payload = cast(Any, json.loads(descriptor.read_text(encoding="utf-8")))
        manifest = decode_dataclass(MixedVArtifactManifest, payload)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise MixedVArtifactError(f"mixed-V artifact descriptor is invalid: {error}") from error
    if manifest.base_packed_descriptor_sha256 != _hash_file(
        opened_base.root / "nanoquant-packed-model.json"
    ):
        raise MixedVArtifactError("mixed-V artifact is bound to a different packed base")
    base_by_name, _ = _base_entries(opened_base)
    for block in manifest.blocks:
        shard = _resolved_member(artifact_root, block.path)
        if not shard.is_file() or shard.stat().st_size != block.bytes:
            raise MixedVArtifactError(f"mixed-V shard size or presence differs: {block.path}")
        if verify_hashes and _hash_file(shard) != block.sha256:
            raise MixedVArtifactError(f"mixed-V shard hash differs: {block.path}")
        for layer in block.layers:
            try:
                _validate_replacement_spec(
                    layer.spec,
                    block.index,
                    base_by_name,
                )
            except ValueError as error:
                raise MixedVArtifactError(str(error)) from error
        declared = {tensor.key: tensor for layer in block.layers for tensor in layer.tensors}
        if len(declared) != sum(len(layer.tensors) for layer in block.layers):
            raise MixedVArtifactError(f"mixed-V tensor key is duplicated: {block.path}")
        try:
            with SAFETENSORS.open(shard) as handle:
                if set(handle.keys()) != set(declared):
                    raise MixedVArtifactError(
                        f"mixed-V shard tensor inventory differs: {block.path}"
                    )
                for key, tensor in declared.items():
                    view = handle.get_slice(key)
                    if (
                        tuple(view.get_shape()) != tensor.shape
                        or _header_dtype(view.get_dtype()) != tensor.dtype
                    ):
                        raise MixedVArtifactError(
                            f"mixed-V tensor header differs: {block.path}:{key}"
                        )
        except MixedVArtifactError:
            raise
        except Exception as error:
            raise MixedVArtifactError(
                f"mixed-V shard header is invalid: {block.path}"
            ) from error
    return OpenMixedVArtifact(artifact_root.resolve(), opened_base, manifest)
__all__ = [
    "MIXED_V_ARTIFACT_FORMAT",
    "MIXED_V_ARTIFACT_SCHEMA_VERSION",
    "MIXED_V_DESCRIPTOR",
    "MIXED_V_TENSOR_NAMESPACE",
    "MixedVArtifactError",
    "MixedVArtifactManifest",
    "MixedVBlockEntry",
    "MixedVLayerEntry",
    "MixedVLayoutMetadata",
    "MixedVTensorEntry",
    "OpenMixedVArtifact",
    "open_mixed_v_artifact",
    "write_mixed_v_artifact",
]
