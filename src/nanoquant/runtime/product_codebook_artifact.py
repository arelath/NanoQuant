"""Base-bound compact product-codebook replacements for packed evaluation."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch

from nanoquant.runtime.artifact import _hash_file
from nanoquant.runtime.backend import DeviceLike, QuantizedLinearSpec
from nanoquant.runtime.codec import decode_dataclass
from nanoquant.runtime.io_utils import atomic_output_directory
from nanoquant.runtime.logical import canonical_torch_dtype
from nanoquant.runtime.packed import PackedLayerState
from nanoquant.runtime.packed_artifact import OpenPackedArtifact, open_packed_artifact
from nanoquant.runtime.product_codebook import (
    PRODUCT_CODEBOOK_FORMAT_VERSION,
    ProductCodebookLayerState,
)
from nanoquant.runtime.safetensors_io import SAFETENSORS

PRODUCT_CODEBOOK_ARTIFACT_SCHEMA_VERSION = 1
PRODUCT_CODEBOOK_ARTIFACT_FORMAT = "nanoquant-product-codebook-overlay"
PRODUCT_CODEBOOK_DESCRIPTOR = "nanoquant-product-codebook-overlay.json"
PRODUCT_CODEBOOK_TENSORS = "components.safetensors"
PRODUCT_CODEBOOK_TENSOR_NAMESPACE = f"layouts.{PRODUCT_CODEBOOK_FORMAT_VERSION}"
_MAXIMUM_DESCRIPTOR_BYTES = 16 * 1024 * 1024
_REQUIRED_ROLES = (
    "factor_left_words",
    "factor_right_free_words",
    "factor_right_coded_payload",
    "factor_right_first_half_words",
    "factor_right_second_half_words",
    "factor_scale_pre",
    "factor_scale_mid",
    "factor_scale_post",
)


class ProductCodebookArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProductCodebookLayerEntry:
    block: int
    spec: QuantizedLinearSpec
    factorization_transposed: bool
    free_rows: int
    compact_logical_bits: int
    tensors: tuple[tuple[str, str, tuple[int, ...], str], ...]

    def __post_init__(self) -> None:
        if self.block < 0 or self.free_rows <= 0 or self.compact_logical_bits <= 0:
            raise ValueError("product-codebook layer metadata is invalid")
        by_role = {role: (key, shape, dtype) for role, key, shape, dtype in self.tensors}
        if len(by_role) != len(self.tensors):
            raise ValueError("product-codebook tensor roles are duplicated")
        required = set(_REQUIRED_ROLES)
        if self.spec.outlier_count:
            required.update(("outlier_indices", "outlier_values"))
        if set(by_role) != required:
            raise ValueError("product-codebook tensor inventory differs")
        prefix = f"{PRODUCT_CODEBOOK_TENSOR_NAMESPACE}.{self.spec.name}."
        if any(key != prefix + role for role, (key, _shape, _dtype) in by_role.items()):
            raise ValueError("product-codebook tensor key differs")


@dataclass(frozen=True, slots=True)
class ProductCodebookArtifactManifest:
    schema_version: int
    artifact_format: str
    layout_version: str
    base_packed_descriptor_sha256: str
    allocation_sha256: str
    allocation_total_bits: int
    effective_bpw: float
    correction_source_sha256: str
    layers: tuple[ProductCodebookLayerEntry, ...]
    layer_count: int
    compact_mlp_bits: int
    tensor_bytes: int
    tensor_sha256: str
    replay: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != PRODUCT_CODEBOOK_ARTIFACT_SCHEMA_VERSION
            or self.artifact_format != PRODUCT_CODEBOOK_ARTIFACT_FORMAT
            or self.layout_version != PRODUCT_CODEBOOK_FORMAT_VERSION
        ):
            raise ValueError("unsupported product-codebook artifact schema")
        if (
            len(self.base_packed_descriptor_sha256) != 64
            or len(self.allocation_sha256) != 64
            or len(self.correction_source_sha256) != 64
            or len(self.tensor_sha256) != 64
        ):
            raise ValueError("product-codebook artifact hash is invalid")
        names = tuple(layer.spec.name for layer in self.layers)
        if (
            not names
            or len(names) != len(set(names))
            or self.layer_count != len(names)
            or self.compact_mlp_bits != sum(layer.compact_logical_bits for layer in self.layers)
            or self.allocation_total_bits <= 0
            or self.effective_bpw <= 0
            or self.tensor_bytes <= 0
            or len({key for key, _value in self.replay}) != len(self.replay)
        ):
            raise ValueError("product-codebook artifact totals are invalid")


@dataclass(frozen=True, slots=True)
class OpenProductCodebookArtifact:
    root: Path
    base: OpenPackedArtifact
    manifest: ProductCodebookArtifactManifest
    tensors: dict[str, torch.Tensor]

    @property
    def replacement_names(self) -> frozenset[str]:
        return frozenset(layer.spec.name for layer in self.manifest.layers)

    def _entry(self, name: str) -> ProductCodebookLayerEntry:
        try:
            return next(layer for layer in self.manifest.layers if layer.spec.name == name)
        except StopIteration as error:
            raise KeyError(f"product-codebook replacement is absent: {name}") from error

    def load_compact_layer(
        self,
        name: str,
        device: DeviceLike = "cpu",
    ) -> ProductCodebookLayerState:
        entry = self._entry(name)
        prefix = f"{PRODUCT_CODEBOOK_TENSOR_NAMESPACE}.{name}."

        def required(role: str) -> torch.Tensor:
            return self.tensors[prefix + role].to(device).contiguous()

        def optional(role: str) -> torch.Tensor | None:
            value = self.tensors.get(prefix + role)
            return None if value is None else value.to(device).contiguous()

        return ProductCodebookLayerState(
            entry.spec,
            PRODUCT_CODEBOOK_FORMAT_VERSION,
            entry.factorization_transposed,
            entry.free_rows,
            required("factor_left_words"),
            required("factor_right_free_words"),
            required("factor_right_coded_payload"),
            required("factor_right_first_half_words"),
            required("factor_right_second_half_words"),
            required("factor_scale_pre"),
            required("factor_scale_mid"),
            required("factor_scale_post"),
            optional("outlier_indices"),
            optional("outlier_values"),
        )

    def load_packed_layer(
        self,
        name: str,
        device: DeviceLike = "cpu",
    ) -> PackedLayerState:
        if name in self.replacement_names:
            return self.load_compact_layer(name, device).to_packed()
        return self.base.load_layer(name, device)


def _state_tensors(state: ProductCodebookLayerState) -> tuple[tuple[str, torch.Tensor], ...]:
    values = [
        ("factor_left_words", state.factor_left_words),
        ("factor_right_free_words", state.factor_right_free_words),
        ("factor_right_coded_payload", state.factor_right_coded_payload),
        ("factor_right_first_half_words", state.first_half_words),
        ("factor_right_second_half_words", state.second_half_words),
        ("factor_scale_pre", state.factor_scale_pre),
        ("factor_scale_mid", state.factor_scale_mid),
        ("factor_scale_post", state.factor_scale_post),
    ]
    if state.outlier_indices is not None:
        assert state.outlier_values is not None
        values.extend(
            (
                ("outlier_indices", state.outlier_indices),
                ("outlier_values", state.outlier_values),
            )
        )
    return tuple(values)


def _base_entries(base: OpenPackedArtifact) -> dict[str, tuple[int, QuantizedLinearSpec]]:
    return {
        layer.spec.name: (block.index, layer.spec)
        for block in base.manifest.blocks
        for layer in block.layers
    }


def _validate_replacement(
    state: ProductCodebookLayerState,
    block: int,
    base_entries: Mapping[str, tuple[int, QuantizedLinearSpec]],
) -> None:
    expected = base_entries.get(state.spec.name)
    if expected is None:
        raise ValueError(f"product-codebook replacement is absent from base: {state.spec.name}")
    expected_block, base_spec = expected
    if expected_block != block or replace(state.spec, rank=base_spec.rank) != base_spec:
        raise ValueError(f"product-codebook replacement spec differs from base: {state.spec.name}")


def write_product_codebook_artifact(
    output: str | Path,
    base: str | Path | OpenPackedArtifact,
    replacements: Mapping[int, Sequence[ProductCodebookLayerState]],
    *,
    allocation_sha256: str,
    allocation_total_bits: int,
    effective_bpw: float,
    correction_source_sha256: str,
    replay: Mapping[str, Any],
) -> OpenProductCodebookArtifact:
    opened_base = base if isinstance(base, OpenPackedArtifact) else open_packed_artifact(base, verify_hashes=True)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"product-codebook output already exists: {destination}")
    base_entries = _base_entries(opened_base)
    tensors: dict[str, torch.Tensor] = {}
    entries = []
    names: set[str] = set()
    for block in sorted(replacements):
        states = tuple(replacements[block])
        if not states:
            raise ValueError(f"product-codebook block {block} is empty")
        for state in states:
            _validate_replacement(state, block, base_entries)
            if state.spec.name in names:
                raise ValueError(f"product-codebook layer is duplicated: {state.spec.name}")
            names.add(state.spec.name)
            metadata = []
            for role, value in _state_tensors(state):
                key = f"{PRODUCT_CODEBOOK_TENSOR_NAMESPACE}.{state.spec.name}.{role}"
                copied = value.detach().cpu().contiguous()
                tensors[key] = copied
                metadata.append((role, key, tuple(copied.shape), canonical_torch_dtype(copied.dtype)))
            entries.append(
                ProductCodebookLayerEntry(
                    block,
                    state.spec,
                    state.factorization_transposed,
                    state.free_rows,
                    state.compact_logical_bits(),
                    tuple(metadata),
                )
            )
    if not entries:
        raise ValueError("product-codebook artifact requires replacements")
    with atomic_output_directory(destination, prefix=".nanoquant-product-codebook-") as temporary:
        tensor_path = temporary / PRODUCT_CODEBOOK_TENSORS
        SAFETENSORS.save(tensors, tensor_path)
        manifest = ProductCodebookArtifactManifest(
            PRODUCT_CODEBOOK_ARTIFACT_SCHEMA_VERSION,
            PRODUCT_CODEBOOK_ARTIFACT_FORMAT,
            PRODUCT_CODEBOOK_FORMAT_VERSION,
            _hash_file(opened_base.root / "nanoquant-packed-model.json"),
            allocation_sha256,
            allocation_total_bits,
            effective_bpw,
            correction_source_sha256,
            tuple(entries),
            len(entries),
            sum(entry.compact_logical_bits for entry in entries),
            tensor_path.stat().st_size,
            _hash_file(tensor_path),
            tuple(
                (key, json.dumps(value, sort_keys=True, separators=(",", ":")))
                for key, value in sorted(replay.items())
            ),
        )
        descriptor = temporary / PRODUCT_CODEBOOK_DESCRIPTOR
        with descriptor.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(asdict(manifest), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return open_product_codebook_artifact(destination, opened_base)


def open_product_codebook_artifact(
    root: str | Path,
    base: str | Path | OpenPackedArtifact,
    *,
    verify_hashes: bool = True,
) -> OpenProductCodebookArtifact:
    artifact_root = Path(root)
    descriptor = artifact_root / PRODUCT_CODEBOOK_DESCRIPTOR
    tensor_path = artifact_root / PRODUCT_CODEBOOK_TENSORS
    if (
        not descriptor.is_file()
        or not tensor_path.is_file()
        or descriptor.stat().st_size > _MAXIMUM_DESCRIPTOR_BYTES
    ):
        raise ProductCodebookArtifactError("product-codebook artifact is incomplete")
    opened_base = (
        base
        if isinstance(base, OpenPackedArtifact)
        else open_packed_artifact(base, verify_hashes=verify_hashes)
    )
    try:
        payload = cast(Any, json.loads(descriptor.read_text(encoding="utf-8")))
        manifest = decode_dataclass(ProductCodebookArtifactManifest, payload)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ProductCodebookArtifactError(f"product-codebook descriptor is invalid: {error}") from error
    if manifest.base_packed_descriptor_sha256 != _hash_file(opened_base.root / "nanoquant-packed-model.json"):
        raise ProductCodebookArtifactError("product-codebook artifact is bound to another base")
    if tensor_path.stat().st_size != manifest.tensor_bytes or (
        verify_hashes and _hash_file(tensor_path) != manifest.tensor_sha256
    ):
        raise ProductCodebookArtifactError("product-codebook tensor size or hash differs")
    declared = {
        key: (shape, dtype)
        for layer in manifest.layers
        for _role, key, shape, dtype in layer.tensors
    }
    if len(declared) != sum(len(layer.tensors) for layer in manifest.layers):
        raise ProductCodebookArtifactError("product-codebook tensor key is duplicated")
    tensors = SAFETENSORS.load(tensor_path, device="cpu")
    if set(tensors) != set(declared) or any(
        tuple(tensors[key].shape) != shape
        or canonical_torch_dtype(tensors[key].dtype) != dtype
        for key, (shape, dtype) in declared.items()
    ):
        raise ProductCodebookArtifactError("product-codebook tensor inventory differs")
    base_entries = _base_entries(opened_base)
    artifact = OpenProductCodebookArtifact(artifact_root.resolve(), opened_base, manifest, tensors)
    for entry in manifest.layers:
        try:
            _validate_replacement(artifact.load_compact_layer(entry.spec.name), entry.block, base_entries)
        except (KeyError, TypeError, ValueError) as error:
            raise ProductCodebookArtifactError(str(error)) from error
    return artifact


__all__ = [
    "PRODUCT_CODEBOOK_ARTIFACT_FORMAT",
    "PRODUCT_CODEBOOK_ARTIFACT_SCHEMA_VERSION",
    "PRODUCT_CODEBOOK_DESCRIPTOR",
    "OpenProductCodebookArtifact",
    "ProductCodebookArtifactError",
    "ProductCodebookArtifactManifest",
    "open_product_codebook_artifact",
    "write_product_codebook_artifact",
]
