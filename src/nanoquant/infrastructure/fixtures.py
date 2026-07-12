"""Canonical non-executable layer/block capture and replay artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from nanoquant.domain.factorization import factorize_admm
from nanoquant.domain.metrics import reconstruction_metrics
from nanoquant.domain.models import ArtifactRef, LayerId, ReconstructionMetrics

from .artifacts import LocalArtifactStore


@dataclass(frozen=True, slots=True)
class LayerReplayResult:
    metrics: ReconstructionMetrics
    expected_close: bool | None
    maximum_absolute_difference: float | None


@dataclass(frozen=True, slots=True)
class BlockReplayResult:
    loss: float
    expected_loss: float
    absolute_difference: float


def capture_layer(
    layer: LayerId,
    weight: torch.Tensor,
    residual_weight: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    rank: int,
    logical_seed: int,
    artifacts: LocalArtifactStore,
    *,
    accepted_reconstruction: torch.Tensor | None = None,
    outer_iterations: int = 400,
    inner_iterations: int = 5,
    device: str = "cpu",
) -> ArtifactRef:
    tensors = {
        "weight": weight.detach().cpu().clone().contiguous(),
        "residual_weight": residual_weight.detach().cpu().clone().contiguous(),
        "input_importance": input_importance.detach().cpu().clone().contiguous(),
        "output_importance": output_importance.detach().cpu().clone().contiguous(),
    }
    if accepted_reconstruction is not None:
        tensors["accepted_reconstruction"] = accepted_reconstruction.detach().cpu().clone().contiguous()
    metadata = {
        "schema_version": 1,
        "layer": {"block": layer.block.index, "path": layer.path},
        "rank": rank,
        "logical_seed": logical_seed,
        "outer_iterations": outer_iterations,
        "inner_iterations": inner_iterations,
        "device": device,
    }
    with artifacts.begin_write("layer-fixture") as writer:
        save_file(tensors, writer.path / "tensors.safetensors")
        (writer.path / "fixture.json").write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
        descriptor = writer.commit()
    return ArtifactRef("layer-fixture", descriptor.artifact_id, 1)


def replay_layer(
    reference: ArtifactRef, artifacts: LocalArtifactStore, *, tolerance: float = 1e-5
) -> LayerReplayResult:
    artifacts.validate(reference.artifact_id)
    root = artifacts.path_for(reference.artifact_id)
    metadata = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    device = str(metadata.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("captured layer replay requires CUDA")
    with safe_open(root / "tensors.safetensors", framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("weight").to(device)
        residual = tensors.get_tensor("residual_weight").to(device)
        input_importance = tensors.get_tensor("input_importance").to(device)
        output_importance = tensors.get_tensor("output_importance").to(device)
        result = factorize_admm(
            residual,
            input_importance,
            output_importance,
            int(metadata["rank"]),
            torch.Generator(device=device).manual_seed(int(metadata["logical_seed"])),
            outer_iterations=int(metadata["outer_iterations"]),
            inner_iterations=int(metadata["inner_iterations"]),
        )
        metrics = reconstruction_metrics(weight, result.reconstruction, input_importance, output_importance)
        if "accepted_reconstruction" in tensors.keys():
            expected = tensors.get_tensor("accepted_reconstruction").to(device)
            difference = float((result.reconstruction - expected).abs().max())
            close = difference <= tolerance
        else:
            difference = None
            close = None
    return LayerReplayResult(metrics, close, difference)


BlockRunner = Callable[[nn.Module, torch.Tensor], torch.Tensor]
BlockFactory = Callable[[], nn.Module]


def capture_block(
    block_index: int,
    block: nn.Module,
    inputs: torch.Tensor,
    teacher_targets: torch.Tensor,
    artifacts: LocalArtifactStore,
    runner: BlockRunner,
) -> ArtifactRef:
    with torch.no_grad():
        accepted_outputs = runner(block, inputs).detach().cpu()
    state = {f"state.{key}": value.detach().cpu().contiguous() for key, value in block.state_dict().items()}
    tensors = {
        **state,
        "inputs": inputs.detach().cpu().contiguous(),
        "teacher_targets": teacher_targets.detach().cpu().contiguous(),
        "accepted_outputs": accepted_outputs.contiguous(),
    }
    metadata = {
        "schema_version": 1,
        "block": block_index,
        "state_keys": sorted(block.state_dict()),
        "accepted_loss": float((accepted_outputs.float() - teacher_targets.float()).square().mean()),
    }
    with artifacts.begin_write("block-fixture") as writer:
        save_file(tensors, writer.path / "tensors.safetensors")
        (writer.path / "fixture.json").write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
        descriptor = writer.commit()
    return ArtifactRef("block-fixture", descriptor.artifact_id, 1)


def replay_block(
    reference: ArtifactRef, artifacts: LocalArtifactStore, factory: BlockFactory, runner: BlockRunner
) -> BlockReplayResult:
    artifacts.validate(reference.artifact_id)
    root = artifacts.path_for(reference.artifact_id)
    metadata = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    block = factory()
    with safe_open(root / "tensors.safetensors", framework="pt", device="cpu") as tensors:
        state = {key: tensors.get_tensor(f"state.{key}") for key in metadata["state_keys"]}
        block.load_state_dict(state, strict=True)
        inputs = tensors.get_tensor("inputs")
        targets = tensors.get_tensor("teacher_targets")
        with torch.no_grad():
            outputs = runner(block, inputs)
    loss = float((outputs.float() - targets.float()).square().mean())
    expected = float(metadata["accepted_loss"])
    return BlockReplayResult(loss, expected, abs(loss - expected))
