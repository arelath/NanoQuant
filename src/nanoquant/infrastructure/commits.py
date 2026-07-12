"""Immutable layer and atomic block commit envelopes with failure injection."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import torch
from safetensors.torch import save_file

from nanoquant.config.codec import to_dict
from nanoquant.domain.models import (
    ActivationStreamRef,
    ArtifactRef,
    BlockId,
    BlockLossMetrics,
    BlockResult,
    FrozenBlockState,
    LayerResult,
)

from .artifacts import LocalArtifactStore

FailureInjector = Callable[[str], None]


def _no_failure(point: str) -> None:
    return None


@dataclass(frozen=True, slots=True)
class CommittedLayer:
    reference: ArtifactRef
    result: LayerResult


@dataclass(frozen=True, slots=True)
class CommittedBlock:
    reference: ArtifactRef
    result: BlockResult


@dataclass(frozen=True, slots=True)
class CommitIdentity:
    config_hash: str
    model_hash: str
    plan_hash: str


def commit_layer(
    result: LayerResult, artifacts: LocalArtifactStore, identity: CommitIdentity, inject: FailureInjector = _no_failure
) -> CommittedLayer:
    inject("before_layer_commit")
    with artifacts.begin_write("layer-result") as writer:
        payload = {"identity": to_dict(identity), "result": to_dict(result)}
        (writer.path / "layer-result.json").write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        descriptor = writer.commit()
    reference = ArtifactRef("layer-result", descriptor.artifact_id, 1)
    inject("after_layer_commit")
    return CommittedLayer(reference, result)


def commit_block(
    block: BlockId,
    layers: tuple[LayerResult, ...],
    frozen_state: FrozenBlockState,
    losses: BlockLossMetrics,
    teacher_outputs: torch.Tensor,
    compressed_outputs: torch.Tensor,
    extra_bits_used: int,
    artifacts: LocalArtifactStore,
    identity: CommitIdentity,
    *,
    inject: FailureInjector = _no_failure,
    wall_seconds: float = 0.0,
    peak_gpu_bytes: int = 0,
    peak_host_bytes: int = 0,
    warnings: tuple[str, ...] = (),
) -> CommittedBlock:
    if teacher_outputs.shape != compressed_outputs.shape:
        raise ValueError("teacher/compressed activation shapes differ")
    inject("before_block_commit")
    core = {
        "schema_version": 1,
        "identity": to_dict(identity),
        "block": to_dict(block),
        "layers": to_dict(layers),
        "frozen_state": to_dict(frozen_state),
        "losses": to_dict(losses),
        "extra_bits_used": extra_bits_used,
        "wall_seconds": wall_seconds,
        "peak_gpu_bytes": peak_gpu_bytes,
        "peak_host_bytes": peak_host_bytes,
        "warnings": list(warnings),
        "teacher_shape": list(teacher_outputs.shape),
        "compressed_shape": list(compressed_outputs.shape),
        "teacher_dtype": str(teacher_outputs.dtype).removeprefix("torch."),
        "compressed_dtype": str(compressed_outputs.dtype).removeprefix("torch."),
    }
    with artifacts.begin_write("block-result") as writer:
        save_file(
            {
                "teacher_outputs": teacher_outputs.detach().cpu().contiguous(),
                "compressed_outputs": compressed_outputs.detach().cpu().contiguous(),
            },
            writer.path / "activations.safetensors",
        )
        (writer.path / "block-result.json").write_text(json.dumps(core, sort_keys=True, indent=2), encoding="utf-8")
        descriptor = writer.commit()
    reference = ArtifactRef("block-result", descriptor.artifact_id, 1)
    teacher_ref = ActivationStreamRef(
        reference,
        tuple(teacher_outputs.shape),
        core["teacher_dtype"],
        teacher_outputs.shape[0],
        teacher_outputs.shape[-2],
    )
    compressed_ref = ActivationStreamRef(
        reference,
        tuple(compressed_outputs.shape),
        core["compressed_dtype"],
        compressed_outputs.shape[0],
        compressed_outputs.shape[-2],
    )
    result = BlockResult(
        1,
        block,
        layers,
        frozen_state,
        losses,
        teacher_ref,
        compressed_ref,
        extra_bits_used,
        wall_seconds,
        peak_gpu_bytes,
        peak_host_bytes,
        warnings,
    )
    inject("after_block_commit")
    return CommittedBlock(reference, result)
