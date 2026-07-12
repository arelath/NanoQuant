"""Load a committed logical frozen model into the dense PyTorch reference backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.layers import BlockEditor, LayerFreezer
from nanoquant.config.codec import from_dict
from nanoquant.domain.models import ArtifactRef, BlockResult
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, load_committed_block
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True, slots=True)
class LoadedFrozenModel:
    model: nn.Module
    blocks: tuple[BlockResult, ...]
    identity: CommitIdentity


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
    values = getattr(base, "layers", None)
    if not isinstance(values, nn.ModuleList):
        raise TypeError("model does not expose a supported decoder layer stack")
    return tuple(values)


def load_frozen_run(
    run_output: str | Path,
    snapshot: str | Path,
    *,
    source_name: str,
    revision: str,
    device: str = "cuda",
    verify_hashes: bool = True,
    backend: str = "factorized",
) -> LoadedFrozenModel:
    run_output = Path(run_output)
    artifacts = LocalArtifactStore(run_output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    source = SafetensorsModelSource(
        snapshot,
        source=source_name,
        revision=revision,
        verify_hashes=verify_hashes,
    )
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    records = [json.loads(line) for line in (run_output / "state" / "journal.jsonl").read_text().splitlines()]
    if not records:
        raise ValueError("frozen run journal is empty")
    identity = from_dict(CommitIdentity, records[0]["identity"], path="identity")
    block_records = {int(record["block"]): record for record in records if record.get("kind") == "block"}
    expected_blocks = adapter.decoder_block_count(source)
    if sorted(block_records) != list(range(expected_blocks)):
        raise ValueError("frozen run does not contain complete contiguous block commits")
    committed = tuple(
        load_committed_block(
            ArtifactRef("block-result", str(block_records[index]["artifact_id"]), 1),
            artifacts,
            identity,
        ).result
        for index in range(expected_blocks)
    )
    model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=_dtype(checkpoint.config),
        ),
    ).to(device)
    model.eval()
    decoder_layers = _decoder_layers(model)
    freezer = LayerFreezer()
    editor = BlockEditor()
    for block_result, block in zip(committed, decoder_layers, strict=True):
        block_dtype = next(block.parameters()).dtype
        for state in block_result.frozen_state.quantized_layers:
            frozen = freezer.load(state, tensors, device=device, dtype=block_dtype, backend=backend)
            editor.install_frozen_layer(block, state.layer.path, frozen.module)
    cast(Any, model).config.use_cache = False
    return LoadedFrozenModel(model, committed, identity)
