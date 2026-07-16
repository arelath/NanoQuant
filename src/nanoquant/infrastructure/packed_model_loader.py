"""Load a packed derivative into the same PyTorch quality-evaluation shell."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from nanoquant.application.layers import (
    BlockEditor,
    DenseWeightReferenceLinear,
    FactorizedReferenceLinear,
    FrozenReferenceLinear,
)
from nanoquant.domain.linear_math import functional_dense_reconstruction
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.commits import CommitIdentity
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.runtime_export import load_frozen_run_auxiliary
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.runtime import OpenPackedArtifact, open_packed_artifact


@dataclass(frozen=True, slots=True)
class LoadedPackedModel:
    model: nn.Module
    packed: OpenPackedArtifact
    identity: CommitIdentity
    global_tuning: ArtifactRef | None
    packed_descriptor_sha256: str


def _dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    return (
        {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(value, torch.float32)
        if isinstance(value, str)
        else torch.float32
    )


def _decoder_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("model does not expose a supported decoder layer stack")
    return tuple(layers)


@torch.inference_mode()
def load_packed_model(
    packed_root: str | Path,
    auxiliary_run: str | Path,
    snapshot: str | Path,
    *,
    source_name: str,
    revision: str,
    expected_blocks: int | None = None,
    device: str = "cuda",
    backend: str = "factorized",
    use_global_tuning: bool = True,
) -> LoadedPackedModel:
    """Install packed linears while retaining the parent's tuned non-linear state."""

    if backend not in {"dense", "factorized"}:
        raise ValueError(f"unsupported packed quality backend: {backend}")
    packed = open_packed_artifact(packed_root, verify_hashes=True)
    if packed.manifest.model.source != source_name or packed.manifest.model.revision != revision:
        raise ValueError("packed quality model identity differs from the requested source")
    block_count = len(packed.manifest.blocks)
    if expected_blocks is not None and block_count != expected_blocks:
        raise ValueError("packed quality model block count differs from the requested protocol")
    source = SafetensorsModelSource(
        snapshot,
        source=source_name,
        revision=revision,
        verify_hashes=False,
    )
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    model = load_causal_language_model(
        snapshot,
        torch_dtype=_dtype(checkpoint.config),
        attention_implementation=adapter.attention_implementation,
    ).to(device)
    model.eval()
    blocks = _decoder_layers(model)
    editor = BlockEditor()
    for packed_block, block in zip(packed.manifest.blocks, blocks, strict=True):
        block_dtype = next(block.parameters()).dtype
        for entry in packed_block.layers:
            state = packed.load_layer(entry.spec.name).to_logical()
            if backend == "dense":
                weight = functional_dense_reconstruction(
                    state.left_binary.to(device),
                    state.right_binary.to(device),
                    state.scale_pre.to(device),
                    state.scale_mid.to(device),
                    state.scale_post.to(device),
                    None if state.outlier_indices is None else state.outlier_indices.to(device),
                    None if state.outlier_values is None else state.outlier_values.to(device),
                    None if state.outlier_scales is None else state.outlier_scales.to(device),
                ).to(block_dtype)
                bias = None if state.bias is None else state.bias.to(device=device, dtype=block_dtype)
                module: FrozenReferenceLinear = DenseWeightReferenceLinear(weight, bias)
            else:
                module = FactorizedReferenceLinear(
                    state.left_binary.to(device=device, dtype=block_dtype),
                    state.right_binary.to(device=device, dtype=block_dtype),
                    state.scale_pre.to(device=device, dtype=block_dtype),
                    state.scale_mid.to(device=device, dtype=block_dtype),
                    state.scale_post.to(device=device, dtype=block_dtype),
                    None if state.bias is None else state.bias.to(device=device, dtype=block_dtype),
                    None if state.outlier_indices is None else state.outlier_indices.to(device),
                    None if state.outlier_values is None else state.outlier_values.to(device=device, dtype=block_dtype),
                    None if state.outlier_scales is None else state.outlier_scales.to(device),
                )
            editor.install_frozen_layer(block, entry.spec.name.split(f"blocks.{packed_block.index}.", 1)[1], module)
            del state, module
        gc.collect()
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()

    auxiliary = load_frozen_run_auxiliary(
        auxiliary_run,
        block_count,
        use_global_tuning=use_global_tuning,
        fresh_validation=True,
    )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in auxiliary.parameters:
            if name not in parameters:
                raise ValueError(f"packed quality auxiliary parameter is absent from the shell: {name}")
            target = parameters[name]
            if target.shape != value.shape:
                raise ValueError(f"packed quality auxiliary parameter shape differs: {name}")
            target.copy_(value.to(device=device, dtype=target.dtype))
    del parameters
    cast(Any, model).config.use_cache = False
    return LoadedPackedModel(
        model,
        packed,
        auxiliary.identity,
        auxiliary.global_tuning,
        hash_file(packed.root / "nanoquant-packed-model.json"),
    )


__all__ = ["LoadedPackedModel", "load_packed_model"]
