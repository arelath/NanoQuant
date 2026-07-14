"""Bounded atomic checkpoints for resumable resident tuning epochs."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.application.tuning import (
    TuningOptimizerState,
    TuningResumeState,
)
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.infrastructure.io_utils import safe_replace


@dataclass(frozen=True, slots=True)
class TuningCheckpointIdentity:
    config_hash: str
    model_hash: str
    plan_hash: str
    block: int
    layer: str
    phase: str


@dataclass(frozen=True, slots=True)
class StoredTuningCheckpoint:
    identity: TuningCheckpointIdentity
    state: TuningResumeState
    generation: str


def _checkpoint_root(run_output: str | Path) -> Path:
    return Path(run_output) / "state" / "tuning-checkpoint"


def _validate_state(state: TuningResumeState) -> None:
    parameter_names = [name for name, _value in state.parameter_values]
    best_names = [name for name, _value in state.best_parameter_values]
    optimizer_names = [value.parameter_name for value in state.optimizer_states]
    if (
        state.completed_epochs < 0
        or len(state.epoch_losses) != state.completed_epochs + 1
        or not all(math.isfinite(value) for value in state.epoch_losses)
        or state.steps_completed < 0
        or state.best_epoch < -1
        or state.best_epoch >= state.completed_epochs
        or len(parameter_names) != len(set(parameter_names))
        or len(best_names) != len(set(best_names))
        or len(optimizer_names) != len(set(optimizer_names))
        or set(parameter_names) != set(best_names)
        or set(parameter_names) != set(optimizer_names)
    ):
        raise ValueError("tuning checkpoint state is inconsistent")
    for name, value in state.parameter_values:
        best = dict(state.best_parameter_values)[name]
        optimizer = {item.parameter_name: item for item in state.optimizer_states}[name]
        if (
            not name
            or value.shape != best.shape
            or value.shape != optimizer.exponential_average.shape
            or value.shape != optimizer.exponential_average_squared.shape
            or (optimizer.kahan_compensation is not None and value.shape != optimizer.kahan_compensation.shape)
            or optimizer.step.numel() != 1
            or int(optimizer.step.item()) != state.steps_completed
        ):
            raise ValueError("tuning checkpoint tensor inventory is inconsistent")


def _write_pointer(root: Path, identity: TuningCheckpointIdentity, generation: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="active-", suffix=".tmp", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {"schema_version": 1, "identity": to_dict(identity), "generation": generation},
                stream,
                sort_keys=True,
                indent=2,
            )
            stream.flush()
            os.fsync(stream.fileno())
        safe_replace(temporary, root / "active.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _remove_inactive_generations(root: Path, active: str | None) -> None:
    for path in root.iterdir():
        if path.is_dir() and (
            path.name.startswith("generation-") or path.name.startswith(".tuning-checkpoint-")
        ):
            if path.name != active:
                shutil.rmtree(path)


def save_tuning_checkpoint(
    run_output: str | Path,
    state: TuningResumeState,
    identity: TuningCheckpointIdentity,
) -> StoredTuningCheckpoint:
    _validate_state(state)
    root = _checkpoint_root(run_output)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tuning-checkpoint-", dir=root))
    generation = f"generation-{uuid.uuid4().hex}"
    final = root / generation
    try:
        optimizer_by_name = {item.parameter_name: item for item in state.optimizer_states}
        best_by_name = dict(state.best_parameter_values)
        tensors: dict[str, torch.Tensor] = {}
        parameters = []
        for index, (name, value) in enumerate(state.parameter_values):
            prefix = f"parameter_{index}"
            optimizer = optimizer_by_name[name]
            tensors[f"{prefix}.value"] = value.detach().cpu().contiguous()
            tensors[f"{prefix}.best"] = best_by_name[name].detach().cpu().contiguous()
            tensors[f"{prefix}.step"] = optimizer.step.detach().cpu().contiguous()
            tensors[f"{prefix}.exp_avg"] = optimizer.exponential_average.detach().cpu().contiguous()
            tensors[f"{prefix}.exp_avg_sq"] = optimizer.exponential_average_squared.detach().cpu().contiguous()
            if optimizer.kahan_compensation is not None:
                tensors[f"{prefix}.kahan_comp"] = optimizer.kahan_compensation.detach().cpu().contiguous()
            parameters.append(
                {
                    "name": name,
                    "prefix": prefix,
                    "has_kahan_compensation": optimizer.kahan_compensation is not None,
                }
            )
        save_file(tensors, temporary / "state.safetensors")
        manifest = {
            "schema_version": 1,
            "identity": to_dict(identity),
            "completed_epochs": state.completed_epochs,
            "epoch_losses": list(state.epoch_losses),
            "steps_completed": state.steps_completed,
            "best_epoch": state.best_epoch,
            "stopped_early": state.stopped_early,
            "parameters": parameters,
        }
        manifest_path = temporary / "checkpoint.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        safe_replace(temporary, final)
        _write_pointer(root, identity, generation)
        _remove_inactive_generations(root, generation)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return StoredTuningCheckpoint(identity, state, generation)


def active_tuning_checkpoint(
    run_output: str | Path,
    identity: TuningCheckpointIdentity,
) -> StoredTuningCheckpoint | None:
    root = _checkpoint_root(run_output)
    pointer = root / "active.json"
    if not pointer.exists():
        return None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported tuning checkpoint pointer schema")
    observed_identity = from_dict(
        TuningCheckpointIdentity,
        payload["identity"],
        path="tuning_checkpoint.identity",
    )
    if observed_identity != identity:
        return None
    generation = str(payload["generation"])
    if not generation.startswith("generation-") or Path(generation).name != generation:
        raise ValueError("tuning checkpoint generation is invalid")
    generation_root = root / generation
    manifest = json.loads((generation_root / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported tuning checkpoint schema")
    manifest_identity = from_dict(
        TuningCheckpointIdentity,
        manifest["identity"],
        path="tuning_checkpoint.identity",
    )
    if manifest_identity != identity:
        raise ValueError("tuning checkpoint manifest identity differs from its pointer")
    parameter_values = []
    best_parameter_values = []
    optimizer_states = []
    with safe_open(generation_root / "state.safetensors", framework="pt", device="cpu") as handle:
        for item in manifest["parameters"]:
            name = str(item["name"])
            prefix = str(item["prefix"])
            parameter_values.append((name, handle.get_tensor(f"{prefix}.value")))
            best_parameter_values.append((name, handle.get_tensor(f"{prefix}.best")))
            optimizer_states.append(
                TuningOptimizerState(
                    name,
                    handle.get_tensor(f"{prefix}.step"),
                    handle.get_tensor(f"{prefix}.exp_avg"),
                    handle.get_tensor(f"{prefix}.exp_avg_sq"),
                    handle.get_tensor(f"{prefix}.kahan_comp")
                    if item.get("has_kahan_compensation", False)
                    else None,
                )
            )
    state = TuningResumeState(
        int(manifest["completed_epochs"]),
        tuple(float(value) for value in manifest["epoch_losses"]),
        int(manifest["steps_completed"]),
        tuple(parameter_values),
        tuple(best_parameter_values),
        tuple(optimizer_states),
        int(manifest["best_epoch"]),
        bool(manifest["stopped_early"]),
    )
    _validate_state(state)
    return StoredTuningCheckpoint(identity, state, generation)


def clear_tuning_checkpoint(run_output: str | Path) -> None:
    root = _checkpoint_root(run_output)
    if not root.exists():
        return
    pointer = root / "active.json"
    if pointer.exists():
        pointer.unlink()
    _remove_inactive_generations(root, None)
    for path in root.iterdir():
        if path.name.startswith("active-") and path.suffix == ".tmp":
            path.unlink()
    try:
        root.rmdir()
    except OSError:
        pass
