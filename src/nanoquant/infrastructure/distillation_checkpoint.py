"""Durable optimizer/parameter checkpoints for resumable model distillation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.application.distillation import (
    DistillationOptimizerState,
    DistillationResumeState,
)
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef

from .artifacts import LocalArtifactStore


@dataclass(frozen=True, slots=True)
class DistillationCheckpointIdentity:
    source_blocks: tuple[ArtifactRef, ...]
    protocol_hash: str
    token_hash: str


@dataclass(frozen=True, slots=True)
class CommittedDistillationCheckpoint:
    reference: ArtifactRef
    identity: DistillationCheckpointIdentity
    state: DistillationResumeState


def commit_distillation_checkpoint(
    state: DistillationResumeState,
    identity: DistillationCheckpointIdentity,
    artifacts: LocalArtifactStore,
) -> CommittedDistillationCheckpoint:
    optimizer_by_name = {item.parameter_name: item for item in state.optimizer_states}
    if set(optimizer_by_name) != set(dict(state.parameter_values)):
        raise ValueError("distillation checkpoint parameters and optimizer states differ")
    tensors: dict[str, torch.Tensor] = {}
    parameters = []
    for index, (name, value) in enumerate(state.parameter_values):
        optimizer = optimizer_by_name[name]
        prefix = f"parameter_{index}"
        tensors[f"{prefix}.value"] = value.detach().cpu().contiguous()
        tensors[f"{prefix}.step"] = optimizer.step.detach().cpu().contiguous()
        tensors[f"{prefix}.exp_avg"] = optimizer.exponential_average.detach().cpu().contiguous()
        tensors[f"{prefix}.exp_avg_sq"] = optimizer.exponential_average_squared.detach().cpu().contiguous()
        parameters.append({"name": name, "prefix": prefix})
    with artifacts.begin_write("distillation-checkpoint") as writer:
        save_file(tensors, writer.path / "state.safetensors")
        (writer.path / "checkpoint.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "identity": to_dict(identity),
                    "completed_epochs": state.completed_epochs,
                    "epoch_losses": list(state.epoch_losses),
                    "steps_completed": state.steps_completed,
                    "parameters": parameters,
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    reference = ArtifactRef("distillation-checkpoint", descriptor.artifact_id, descriptor.schema_version)
    return CommittedDistillationCheckpoint(reference, identity, state)


def load_distillation_checkpoint(
    reference: ArtifactRef,
    identity: DistillationCheckpointIdentity,
    artifacts: LocalArtifactStore,
) -> CommittedDistillationCheckpoint:
    descriptor = artifacts.validate(reference.artifact_id)
    if descriptor.artifact_type != "distillation-checkpoint":
        raise ValueError("artifact is not a distillation checkpoint")
    root = artifacts.path_for(reference.artifact_id)
    manifest = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    observed_identity = from_dict(
        DistillationCheckpointIdentity,
        manifest["identity"],
        path="distillation_checkpoint.identity",
    )
    if observed_identity != identity:
        raise ValueError("distillation checkpoint does not match the requested run/protocol")
    parameter_values = []
    optimizer_states = []
    with safe_open(root / "state.safetensors", framework="pt", device="cpu") as handle:
        for item in manifest["parameters"]:
            name = str(item["name"])
            prefix = str(item["prefix"])
            parameter_values.append((name, handle.get_tensor(f"{prefix}.value")))
            optimizer_states.append(
                DistillationOptimizerState(
                    name,
                    handle.get_tensor(f"{prefix}.step"),
                    handle.get_tensor(f"{prefix}.exp_avg"),
                    handle.get_tensor(f"{prefix}.exp_avg_sq"),
                )
            )
    state = DistillationResumeState(
        int(manifest["completed_epochs"]),
        tuple(float(value) for value in manifest["epoch_losses"]),
        int(manifest["steps_completed"]),
        tuple(parameter_values),
        tuple(optimizer_states),
    )
    return CommittedDistillationCheckpoint(reference, identity, state)


def activate_distillation_checkpoint(run_output: str | Path, reference: ArtifactRef) -> None:
    output = Path(run_output)
    output.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="distillation-checkpoint-", suffix=".tmp", dir=output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(to_dict(reference), stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "global-distillation-training.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def active_distillation_checkpoint(
    run_output: str | Path,
    identity: DistillationCheckpointIdentity,
    artifacts: LocalArtifactStore,
) -> CommittedDistillationCheckpoint | None:
    path = Path(run_output) / "global-distillation-training.json"
    if not path.exists():
        return None
    reference = from_dict(
        ArtifactRef,
        json.loads(path.read_text(encoding="utf-8")),
        path="distillation_checkpoint_reference",
    )
    return load_distillation_checkpoint(reference, identity, artifacts)
