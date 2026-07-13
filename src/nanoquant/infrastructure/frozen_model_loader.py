"""Load a committed logical frozen model into the dense PyTorch reference backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.layers import BlockEditor, LayerFreezer, restore_block_auxiliary_parameters
from nanoquant.config.codec import from_dict
from nanoquant.domain.models import ArtifactRef, BlockResult
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, load_committed_block
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True, slots=True)
class LoadedFrozenModel:
    model: nn.Module
    blocks: tuple[BlockResult, ...]
    identity: CommitIdentity
    global_tuning: ArtifactRef | None


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


def _latest_complete_identity(
    records: list[dict[str, Any]], expected_blocks: int
) -> tuple[CommitIdentity, dict[int, dict[str, Any]]]:
    """Select the newest journal identity with one complete contiguous block set."""
    seen: set[str] = set()
    for candidate in reversed(records):
        payload = candidate.get("identity")
        if not isinstance(payload, dict):
            continue
        key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        block_records = {
            int(record["block"]): record
            for record in records
            if record.get("kind") == "block" and record.get("identity") == payload
        }
        if sorted(block_records) == list(range(expected_blocks)):
            return from_dict(CommitIdentity, payload, path="identity"), block_records
    raise ValueError("frozen run does not contain a complete contiguous block identity")


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
    expected_blocks = adapter.decoder_block_count(source)
    identity, block_records = _latest_complete_identity(records, expected_blocks)
    committed = tuple(
        load_committed_block(
            ArtifactRef("block-result", str(block_records[index]["artifact_id"]), 1),
            artifacts,
            identity,
        ).result
        for index in range(expected_blocks)
    )
    global_tuning_ref = active_global_tuning(run_output)
    global_tuning = None if global_tuning_ref is None else load_global_tuning(global_tuning_ref, artifacts).result
    source_blocks = tuple(block.teacher_outputs.artifact for block in committed)
    if global_tuning is not None:
        if global_tuning.source_blocks != source_blocks:
            raise ValueError("global tuning result does not match the run's committed blocks")
        if tuple(state.block.index for state in global_tuning.tuned_blocks) != tuple(range(expected_blocks)):
            raise ValueError("global tuning result does not contain complete contiguous block states")
    model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=_dtype(checkpoint.config),
            attn_implementation=adapter.attention_implementation,
        ),
    ).to(device)
    model.eval()
    decoder_layers = _decoder_layers(model)
    freezer = LayerFreezer()
    editor = BlockEditor()
    block_states = (
        tuple(block.frozen_state for block in committed)
        if global_tuning is None
        else global_tuning.tuned_blocks
    )
    for block_state, block in zip(block_states, decoder_layers, strict=True):
        block_dtype = next(block.parameters()).dtype
        for state in block_state.quantized_layers:
            frozen = freezer.load(state, tensors, device=device, dtype=block_dtype, backend=backend)
            editor.install_frozen_layer(block, state.layer.path, frozen.module)
        restore_block_auxiliary_parameters(
            block,
            block_state.auxiliary_parameters,
            tensors,
            device=device,
        )
    if global_tuning is not None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, reference in global_tuning.auxiliary_parameters:
                if name not in parameters:
                    raise ValueError(f"global tuning parameter is absent from the model: {name}")
                parameter = parameters[name]
                with tensors.read(reference, device) as value:
                    if value.shape != parameter.shape:
                        raise ValueError(f"global tuning parameter shape differs: {name}")
                    parameter.copy_(value.to(dtype=parameter.dtype))
    cast(Any, model).config.use_cache = False
    return LoadedFrozenModel(model, committed, identity, global_tuning_ref)
