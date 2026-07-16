"""Explicit bridge from packed runtime artifacts to the modified llama.cpp converter."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file

from nanoquant.runtime.artifact import RuntimeModelMetadata, _hash_file
from nanoquant.runtime.packed import PackedLayerState, PackedReferenceProvenance
from nanoquant.runtime.packed_artifact import open_packed_artifact

LLAMACPP_CHECKPOINT_SCHEMA_VERSION = 1
LLAMACPP_CHECKPOINT_FORMAT = "nanoquant-llamacpp-checkpoint"
_GEMMA_LAYER = re.compile(
    r"blocks\.(?P<block>[0-9]+)\."
    r"(?P<path>self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\Z"
)
_GEMMA_GGUF_BASES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


@dataclass(frozen=True, slots=True)
class LlamaCppCheckpointShard:
    index: int
    path: str
    bytes: int
    sha256: str
    layer_count: int
    tensor_count: int


@dataclass(frozen=True, slots=True)
class LlamaCppCheckpointManifest:
    schema_version: int
    artifact_format: str
    model: RuntimeModelMetadata
    source_packed_descriptor_sha256: str
    reference: PackedReferenceProvenance
    shards: tuple[LlamaCppCheckpointShard, ...]
    layer_count: int
    tensor_count: int
    weight_bytes: int


def open_llamacpp_checkpoint(
    root: str | Path,
    *,
    verify_hashes: bool = True,
) -> LlamaCppCheckpointManifest:
    """Validate a converter checkpoint descriptor and all of its block shards."""

    checkpoint_root = Path(root)
    descriptor = checkpoint_root / "nanoquant-llamacpp-checkpoint.json"
    if not descriptor.is_file():
        raise ValueError("llama.cpp checkpoint descriptor is missing")
    try:
        payload = cast(Any, json.loads(descriptor.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("llama.cpp checkpoint descriptor is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("llama.cpp checkpoint descriptor must be an object")
    expected_fields = {
        "schema_version",
        "artifact_format",
        "model",
        "source_packed_descriptor_sha256",
        "reference",
        "shards",
        "layer_count",
        "tensor_count",
        "weight_bytes",
    }
    if set(payload) != expected_fields:
        raise ValueError("llama.cpp checkpoint descriptor fields differ from schema")
    try:
        model_payload = cast(dict[str, Any], payload["model"])
        reference_payload = cast(dict[str, Any], payload["reference"])
        shard_payloads = cast(list[dict[str, Any]], payload["shards"])
        manifest = LlamaCppCheckpointManifest(
            int(payload["schema_version"]),
            str(payload["artifact_format"]),
            RuntimeModelMetadata(**model_payload),
            str(payload["source_packed_descriptor_sha256"]),
            PackedReferenceProvenance(**reference_payload),
            tuple(LlamaCppCheckpointShard(**shard) for shard in shard_payloads),
            int(payload["layer_count"]),
            int(payload["tensor_count"]),
            int(payload["weight_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("llama.cpp checkpoint descriptor fields are invalid") from exc
    if manifest.schema_version != LLAMACPP_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("llama.cpp checkpoint schema version is unsupported")
    if manifest.artifact_format != LLAMACPP_CHECKPOINT_FORMAT:
        raise ValueError("llama.cpp checkpoint artifact format is unsupported")
    if len({shard.index for shard in manifest.shards}) != len(manifest.shards):
        raise ValueError("llama.cpp checkpoint shard indices are duplicated")
    if tuple(shard.index for shard in manifest.shards) != tuple(range(len(manifest.shards))):
        raise ValueError("llama.cpp checkpoint shard indices are not contiguous")
    for shard in manifest.shards:
        relative = Path(shard.path)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
            raise ValueError(f"llama.cpp checkpoint shard path is unsafe: {shard.path}")
        path = checkpoint_root / relative
        if not path.is_file() or path.stat().st_size != shard.bytes:
            raise ValueError(f"llama.cpp checkpoint shard size or presence differs: {shard.path}")
        if verify_hashes and _hash_file(path) != shard.sha256:
            raise ValueError(f"llama.cpp checkpoint shard hash differs: {shard.path}")
    if sum(shard.layer_count for shard in manifest.shards) != manifest.layer_count:
        raise ValueError("llama.cpp checkpoint layer count differs from its shards")
    if sum(shard.tensor_count for shard in manifest.shards) != manifest.tensor_count:
        raise ValueError("llama.cpp checkpoint tensor count differs from its shards")
    if sum(shard.bytes for shard in manifest.shards) != manifest.weight_bytes:
        raise ValueError("llama.cpp checkpoint byte count differs from its shards")
    return manifest


def gemma_hf_checkpoint_prefix(canonical_name: str) -> str:
    match = _GEMMA_LAYER.fullmatch(canonical_name)
    if match is None:
        raise ValueError(f"unsupported Gemma runtime layer name: {canonical_name}")
    return f"model.layers.{int(match.group('block'))}.{match.group('path')}"


def gemma_gguf_tensor_prefix(canonical_name: str) -> str:
    """Map one canonical runtime layer to its modified llama.cpp GGUF tensor base."""

    match = _GEMMA_LAYER.fullmatch(canonical_name)
    if match is None:
        raise ValueError(f"unsupported Gemma runtime layer name: {canonical_name}")
    return f"blk.{int(match.group('block'))}.{_GEMMA_GGUF_BASES[match.group('path')]}"


def llamacpp_checkpoint_tensors(
    state: PackedLayerState,
    checkpoint_prefix: str,
) -> dict[str, torch.Tensor]:
    """Return the exact packed checkpoint group accepted by the pinned converter."""

    if not checkpoint_prefix or checkpoint_prefix.startswith(".") or checkpoint_prefix.endswith("."):
        raise ValueError("llama.cpp checkpoint prefix must be a canonical dotted name")
    if state.bias is not None:
        raise ValueError(
            "llama.cpp NanoQuant sidecars do not carry bias; model-shell bias export is required"
        )
    tensors = {
        f"{checkpoint_prefix}.V_packed": state.right_words.detach().cpu().contiguous(),
        f"{checkpoint_prefix}.V_shape": torch.tensor(
            (state.spec.rank, state.spec.in_features), dtype=torch.int64
        ),
        f"{checkpoint_prefix}.U_packed": state.left_words.detach().cpu().contiguous(),
        f"{checkpoint_prefix}.U_shape": torch.tensor(
            (state.spec.out_features, state.spec.rank), dtype=torch.int64
        ),
        f"{checkpoint_prefix}.scale_pre": state.scale_pre.detach().cpu().contiguous(),
        f"{checkpoint_prefix}.scale_mid": state.scale_mid.detach().cpu().contiguous(),
        f"{checkpoint_prefix}.scale_post": state.scale_post.detach().cpu().contiguous(),
    }
    optional = (
        ("salient_idx", state.outlier_indices),
        ("salient_weight", state.outlier_values),
        ("salient_scale", state.outlier_scales),
    )
    for suffix, value in optional:
        if value is not None:
            tensors[f"{checkpoint_prefix}.{suffix}"] = value.detach().cpu().contiguous()
    return tensors


def export_llamacpp_checkpoint(
    packed_root: str | Path,
    output: str | Path,
) -> LlamaCppCheckpointManifest:
    """Write block shards consumable by the pinned modified llama.cpp converter."""

    packed = open_packed_artifact(packed_root, verify_hashes=True)
    if packed.manifest.model.family != "gemma3":
        raise ValueError(
            f"llama.cpp checkpoint export does not support model family: "
            f"{packed.manifest.model.family}"
        )
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"llama.cpp checkpoint output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".nanoquant-llamacpp-", dir=destination.parent))
    try:
        shards: list[LlamaCppCheckpointShard] = []
        total_layers = 0
        total_tensors = 0
        for block in packed.manifest.blocks:
            tensors: dict[str, torch.Tensor] = {}
            for layer in block.layers:
                state = packed.load_layer(layer.spec.name)
                prefix = gemma_hf_checkpoint_prefix(layer.spec.name)
                group = llamacpp_checkpoint_tensors(state, prefix)
                duplicates = set(tensors).intersection(group)
                if duplicates:
                    raise ValueError(f"llama.cpp checkpoint tensor is duplicated: {sorted(duplicates)}")
                tensors.update(group)
            relative = f"block-{block.index:05d}.safetensors"
            shard = temporary / relative
            save_file(tensors, shard)
            shards.append(
                LlamaCppCheckpointShard(
                    block.index,
                    relative,
                    shard.stat().st_size,
                    _hash_file(shard),
                    len(block.layers),
                    len(tensors),
                )
            )
            total_layers += len(block.layers)
            total_tensors += len(tensors)
            tensors.clear()
        source_descriptor = packed.root / "nanoquant-packed-model.json"
        manifest = LlamaCppCheckpointManifest(
            LLAMACPP_CHECKPOINT_SCHEMA_VERSION,
            LLAMACPP_CHECKPOINT_FORMAT,
            packed.manifest.model,
            _hash_file(source_descriptor),
            packed.manifest.layout.reference,
            tuple(shards),
            total_layers,
            total_tensors,
            sum(shard.bytes for shard in shards),
        )
        descriptor = temporary / "nanoquant-llamacpp-checkpoint.json"
        with descriptor.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(asdict(manifest), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
