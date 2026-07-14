"""Immutable layer and atomic block commit envelopes with failure injection."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import (
    ActivationStreamRef,
    ArtifactRef,
    ArtifactTypes,
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


def _payload(reference: ArtifactRef, artifacts: LocalArtifactStore, filename: str) -> dict[str, Any]:
    artifacts.validate(reference.artifact_id)
    try:
        value = json.loads((artifacts.path_for(reference.artifact_id) / filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid committed result payload: {reference.artifact_id}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"committed result payload is not an object: {reference.artifact_id}")
    return cast(dict[str, Any], value)


def load_committed_layer(
    reference: ArtifactRef, artifacts: LocalArtifactStore, identity: CommitIdentity
) -> CommittedLayer:
    payload = _payload(reference, artifacts, "layer-result.json")
    if from_dict(CommitIdentity, payload["identity"], path="identity") != identity:
        raise ValueError("committed layer identity does not match the active run")
    result = from_dict(LayerResult, payload["result"], path="layer_result")
    return CommittedLayer(reference, result)


def load_committed_block(
    reference: ArtifactRef, artifacts: LocalArtifactStore, identity: CommitIdentity
) -> CommittedBlock:
    payload = _payload(reference, artifacts, "block-result.json")
    if from_dict(CommitIdentity, payload["identity"], path="identity") != identity:
        raise ValueError("committed block identity does not match the active run")
    block = from_dict(BlockId, payload["block"], path="block")
    layers = tuple(
        from_dict(LayerResult, item, path=f"layers[{index}]") for index, item in enumerate(payload["layers"])
    )
    frozen_state = from_dict(FrozenBlockState, payload["frozen_state"], path="frozen_state")
    losses = from_dict(BlockLossMetrics, payload["losses"], path="losses")
    teacher_shape = tuple(int(value) for value in payload["teacher_shape"])
    compressed_shape = tuple(int(value) for value in payload["compressed_shape"])
    schema_version = int(payload["schema_version"])
    activation_reference = (
        reference
        if schema_version == 1
        else from_dict(ArtifactRef, payload["activation_generation"], path="activation_generation")
    )
    teacher = ActivationStreamRef(
        activation_reference,
        teacher_shape,
        str(payload["teacher_dtype"]),
        teacher_shape[0],
        teacher_shape[-2],
    )
    compressed = ActivationStreamRef(
        activation_reference,
        compressed_shape,
        str(payload["compressed_dtype"]),
        compressed_shape[0],
        compressed_shape[-2],
    )
    result = BlockResult(
        schema_version,
        block,
        layers,
        frozen_state,
        losses,
        teacher,
        compressed,
        int(payload["extra_bits_used"]),
        float(payload["wall_seconds"]),
        int(payload["peak_gpu_bytes"]),
        int(payload["peak_host_bytes"]),
        tuple(str(value) for value in payload["warnings"]),
    )
    return CommittedBlock(reference, result)


def load_block_activations(
    reference: ArtifactRef, artifacts: LocalArtifactStore, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    artifacts.validate(reference.artifact_id)
    root = artifacts.path_for(reference.artifact_id)
    legacy = root / "activations.safetensors"
    if legacy.exists():
        with safe_open(legacy, framework="pt", device="cpu") as handle:
            teacher = handle.get_tensor("teacher_outputs")
            compressed = handle.get_tensor("compressed_outputs")
    elif (root / "teacher-activations.safetensors").exists():
        with safe_open(root / "teacher-activations.safetensors", framework="pt", device="cpu") as handle:
            teacher = handle.get_tensor("teacher_outputs")
        with safe_open(root / "compressed-activations.safetensors", framework="pt", device="cpu") as handle:
            compressed = handle.get_tensor("compressed_outputs")
    else:
        payload = _payload(reference, artifacts, "block-result.json")
        generation = from_dict(ArtifactRef, payload["activation_generation"], path="activation_generation")
        artifacts.validate(generation.artifact_id)
        generation_root = artifacts.path_for(generation.artifact_id)
        with safe_open(generation_root / "teacher-activations.safetensors", framework="pt", device="cpu") as handle:
            teacher = handle.get_tensor("teacher_outputs")
        with safe_open(
            generation_root / "compressed-activations.safetensors", framework="pt", device="cpu"
        ) as handle:
            compressed = handle.get_tensor("compressed_outputs")
    if device != "cpu":
        teacher = teacher.to(device)
        compressed = compressed.to(device)
    return teacher, compressed


@dataclass(frozen=True, slots=True)
class CommitIdentity:
    config_hash: str
    model_hash: str
    plan_hash: str


def commit_layer(
    result: LayerResult, artifacts: LocalArtifactStore, identity: CommitIdentity, inject: FailureInjector = _no_failure
) -> CommittedLayer:
    inject("before_layer_commit")
    with artifacts.begin_write(ArtifactTypes.LAYER_RESULT) as writer:
        with artifacts.recorder.phase("serialize"):
            payload = {"identity": to_dict(identity), "result": to_dict(result)}
            encoded = json.dumps(payload, sort_keys=True, indent=2)
        with artifacts.recorder.phase("write"):
            (writer.path / "layer-result.json").write_text(encoded, encoding="utf-8")
        descriptor = writer.commit()
    reference = ArtifactRef(ArtifactTypes.LAYER_RESULT, descriptor.artifact_id, 1)
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
    with artifacts.recorder.phase("serialize"):
        activation_core = {
            "schema_version": 1,
            "identity": to_dict(identity),
            "boundary_after_block": block.index,
            "teacher_shape": list(teacher_outputs.shape),
            "compressed_shape": list(compressed_outputs.shape),
            "teacher_dtype": str(teacher_outputs.dtype).removeprefix("torch."),
            "compressed_dtype": str(compressed_outputs.dtype).removeprefix("torch."),
        }
        encoded_activation_core = json.dumps(activation_core, sort_keys=True, indent=2)
    with artifacts.begin_write(ArtifactTypes.ACTIVATION_GENERATION) as writer:
        with artifacts.recorder.phase("write"):
            save_file(
                {"teacher_outputs": teacher_outputs.detach().cpu().contiguous()},
                writer.path / "teacher-activations.safetensors",
            )
            save_file(
                {"compressed_outputs": compressed_outputs.detach().cpu().contiguous()},
                writer.path / "compressed-activations.safetensors",
            )
            (writer.path / "activation-generation.json").write_text(encoded_activation_core, encoding="utf-8")
        activation_descriptor = writer.commit()
    artifacts.recorder.add(
        "io.activation_bytes_written",
        sum(item.bytes for item in activation_descriptor.files if item.path.endswith(".safetensors")),
    )
    activation_reference = ArtifactRef(ArtifactTypes.ACTIVATION_GENERATION, activation_descriptor.artifact_id, 1)
    inject("after_activation_commit")
    with artifacts.recorder.phase("serialize"):
        core = {
            "schema_version": 2,
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
            "activation_generation": to_dict(activation_reference),
        }
        encoded_core = json.dumps(core, sort_keys=True, indent=2)
    with artifacts.begin_write(ArtifactTypes.BLOCK_RESULT) as writer:
        with artifacts.recorder.phase("write"):
            (writer.path / "block-result.json").write_text(encoded_core, encoding="utf-8")
        descriptor = writer.commit()
    reference = ArtifactRef(ArtifactTypes.BLOCK_RESULT, descriptor.artifact_id, 1)
    teacher_ref = ActivationStreamRef(
        activation_reference,
        tuple(teacher_outputs.shape),
        core["teacher_dtype"],
        teacher_outputs.shape[0],
        teacher_outputs.shape[-2],
    )
    compressed_ref = ActivationStreamRef(
        activation_reference,
        tuple(compressed_outputs.shape),
        core["compressed_dtype"],
        compressed_outputs.shape[0],
        compressed_outputs.shape[-2],
    )
    result = BlockResult(
        2,
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


def retire_block_activations(result: BlockResult, artifacts: LocalArtifactStore) -> int:
    """Retire external resume generations while leaving durable block evidence intact."""
    references = {result.teacher_outputs.artifact, result.compressed_outputs.artifact}
    retired = 0
    for reference in references:
        if (
            reference.artifact_type == ArtifactTypes.ACTIVATION_GENERATION
            and artifacts.path_for(reference.artifact_id).exists()
        ):
            retired += artifacts.remove_artifact(
                reference.artifact_id, expected_type=ArtifactTypes.ACTIVATION_GENERATION
            )
    return retired
