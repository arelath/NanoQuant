"""Load a committed logical frozen model into the dense PyTorch reference backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from nanoquant.application.layers import (
    BlockEditor,
    LayerFreezer,
    SharedInputGroupFreezer,
    restore_block_auxiliary_parameters,
)
from nanoquant.domain.models import ArtifactRef, ArtifactTypes, BlockResult
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, latest_complete_identity, load_committed_block
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
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


def load_frozen_run(
    run_output: str | Path,
    snapshot: str | Path,
    *,
    source_name: str,
    revision: str,
    device: str = "cuda",
    verify_hashes: bool = True,
    backend: str = "factorized",
    use_global_tuning: bool = True,
    recorder: PhaseRecorder = NULL_RECORDER,
) -> LoadedFrozenModel:
    run_output = Path(run_output)
    artifacts = LocalArtifactStore(run_output / "artifacts", recorder=recorder)
    tensors = LocalTensorStore(artifacts)
    with recorder.phase("inventory"):
        source = SafetensorsModelSource(
            snapshot,
            source=source_name,
            revision=revision,
            verify_hashes=verify_hashes,
        )
        checkpoint = source.inventory()
        adapter = adapter_for_config(checkpoint.config)
    with recorder.phase("journal"):
        records = [json.loads(line) for line in (run_output / "state" / "journal.jsonl").read_text().splitlines()]
    if not records:
        raise ValueError("frozen run journal is empty")
    expected_blocks = adapter.decoder_block_count(source)
    with recorder.phase("commits"):
        identity, block_records = latest_complete_identity(records, expected_blocks)
        committed = tuple(
            load_committed_block(
                ArtifactRef(ArtifactTypes.BLOCK_RESULT, str(block_records[index]["artifact_id"]), 1),
                artifacts,
                identity,
            ).result
            for index in range(expected_blocks)
        )
    with recorder.phase("global_tuning"):
        global_tuning_ref = active_global_tuning(run_output) if use_global_tuning else None
        global_tuning = None if global_tuning_ref is None else load_global_tuning(global_tuning_ref, artifacts).result
    source_blocks = tuple(block.teacher_outputs.artifact for block in committed)
    if global_tuning is not None:
        if global_tuning.source_blocks != source_blocks:
            raise ValueError("global tuning result does not match the run's committed blocks")
        if tuple(state.block.index for state in global_tuning.tuned_blocks) != tuple(range(expected_blocks)):
            raise ValueError("global tuning result does not contain complete contiguous block states")
    with recorder.phase("model_load"):
        model = load_causal_language_model(
            snapshot,
            torch_dtype=_dtype(checkpoint.config),
            attention_implementation=adapter.attention_implementation,
        ).to(device)
    model.eval()
    decoder_layers = _decoder_layers(model)
    freezer = LayerFreezer()
    editor = BlockEditor()
    block_states = (
        tuple(block.frozen_state for block in committed) if global_tuning is None else global_tuning.tuned_blocks
    )
    for block_state, block in zip(block_states, decoder_layers, strict=True):
        with recorder.phase("install_block", block=block_state.block.index):
            block_dtype = next(block.parameters()).dtype
            for state in block_state.quantized_layers:
                with recorder.phase("install_layer", layer=state.layer.path):
                    frozen = freezer.load(
                        state,
                        tensors,
                        device=device,
                        dtype=block_dtype,
                        backend=backend,
                        compact_dense=backend == "dense",
                    )
                    editor.install_frozen_layer(block, state.layer.path, frozen.module)
            for group_state in block_state.shared_input_groups:
                with recorder.phase("install_group", layer=group_state.name):
                    frozen_group = SharedInputGroupFreezer().load(
                        group_state,
                        tensors,
                        device=device,
                        dtype=block_dtype,
                        backend="factorized",
                    )
                    editor.install_frozen_group(block, frozen_group)
                recorder.add("replay.layers", 1)
            with recorder.phase("auxiliary"):
                restore_block_auxiliary_parameters(
                    block,
                    block_state.auxiliary_parameters,
                    tensors,
                    device=device,
                )
            recorder.add("replay.blocks", 1)
    if global_tuning is not None:
        with recorder.phase("install_global_parameters"):
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
