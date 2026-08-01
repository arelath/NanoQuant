"""Resumable post-KD tuning of zero-byte foldable MLP multipliers."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from nanoquant.application.distillation import TopKTeacherBatch, topk_distillation_loss
from nanoquant.application.foldable_mlp_multipliers import (
    InstalledMultipliers,
    family_identity_penalty,
    fold_global_mlp_multipliers,
    gradient_summary,
    install_global_mlp_multipliers,
    multiplier_summary,
    seed_global_mlp_multipliers,
)
from nanoquant.config.codec import semantic_hash, to_dict
from nanoquant.config.schema import FoldableMlpMultiplierTuningConfig, RunConfig
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import latest_complete_identity
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.distillation_cache import TeacherCacheJournal, load_teacher_epoch
from nanoquant.infrastructure.factorized_component_overlay import load_factorized_component_overlay
from nanoquant.infrastructure.foldable_mlp_initializer import (
    FoldableMlpInitializer,
    load_foldable_mlp_initializer,
)
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.global_tuning import active_global_tuning
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json, hash_file
from nanoquant.infrastructure.safetensors_io import SAFETENSORS

_OVERLAY_DIRECTORY = "foldable-mlp-overlay"
_CHECKPOINT_DIRECTORY = "foldable-mlp-training"
_REPORT_NAME = "foldable-mlp-tuning-report.json"
_ACTIVE_NAME = "foldable-mlp-tuning.json"


@dataclass(frozen=True, slots=True)
class FoldableMlpTuningResult:
    overlay: Path
    report: Path
    protocol_hash: str
    tensor_sha256: str
    steps_completed: int
    reused: bool


def _hidden_states(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    text_stack = getattr(model, "model", None)
    if not isinstance(text_stack, nn.Module):
        language_model = getattr(model, "language_model", None)
        text_stack = getattr(language_model, "model", None)
    if not isinstance(text_stack, nn.Module):
        raise TypeError("model does not expose a supported text stack")
    outputs = cast(Any, text_stack)(input_ids=token_ids, use_cache=False)
    value = outputs[0] if isinstance(outputs, tuple) else getattr(outputs, "last_hidden_state", None)
    if not isinstance(value, torch.Tensor):
        raise TypeError("model text stack did not return hidden states")
    return value


def _lm_head(model: nn.Module) -> nn.Module:
    value = getattr(model, "lm_head", None)
    if not isinstance(value, nn.Module):
        raise TypeError("model does not expose an LM head")
    return value


def _hidden_width(model: nn.Module) -> int:
    width = getattr(getattr(model, "config", None), "hidden_size", None)
    if not isinstance(width, int):
        raise TypeError("model config does not expose hidden size")
    return width


def _load_calibration(run_output: Path) -> torch.Tensor:
    payload = json.loads((run_output / "calibration-input.json").read_text(encoding="utf-8"))
    reference = ArtifactRef("calibration-dataset-manifest", str(payload["artifact_id"]), 1)
    return load_pinned_calibration(run_output, reference).input_ids


def _token_hash(tokens: torch.Tensor) -> str:
    value = tokens.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _training_batches(run_output: Path, steps: int) -> tuple[tuple[TopKTeacherBatch, ...], int]:
    payload = json.loads((run_output / "global-distillation-cache.json").read_text(encoding="utf-8"))
    from nanoquant.config.codec import from_dict

    journal = from_dict(TeacherCacheJournal, payload, path="teacher_cache_journal")
    artifacts = LocalArtifactStore(run_output / "artifacts")
    batches: list[TopKTeacherBatch] = []
    loaded_bytes = 0
    for reference in journal.epochs:
        if reference is None:
            raise ValueError("retained global-distillation teacher cache is incomplete")
        epoch = load_teacher_epoch(reference, journal.identity, artifacts)
        loaded_bytes += epoch.bytes
        batches.extend(epoch.batches)
        if len(batches) >= steps:
            break
    if len(batches) < steps:
        raise ValueError(
            f"foldable MLP tuning requests {steps} steps but the retained cache has {len(batches)}"
        )
    return tuple(batches[:steps]), loaded_bytes


def _commit_identity(run_output: Path) -> dict[str, str]:
    records = [
        json.loads(line)
        for line in (run_output / "state" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    block_indices = {
        int(record["block"])
        for record in records
        if record.get("kind") == "block" and "block" in record
    }
    if not block_indices:
        raise ValueError("foldable MLP tuning requires complete committed blocks")
    identity, _ = latest_complete_identity(records, max(block_indices) + 1)
    return {
        "model_hash": identity.model_hash,
        "config_hash": identity.config_hash,
        "plan_hash": identity.plan_hash,
    }


def _protocol_hash(
    config: FoldableMlpMultiplierTuningConfig,
    frozen_identity: dict[str, str],
    global_tuning: ArtifactRef,
    token_hash: str,
    initializer: FoldableMlpInitializer | None,
) -> str:
    return semantic_hash(
        {
            "producer": "foldable-mlp-multiplier-tuning-v1",
            "config": config,
            "frozen_identity": frozen_identity,
            "global_tuning": global_tuning,
            "token_hash": token_hash,
            "initializer": (
                None
                if initializer is None
                else {
                    "tensor_sha256": initializer.tensor_sha256,
                    "manifest_sha256": hash_file(initializer.root / "manifest.json"),
                }
            ),
        }
    )


def _load_initializer(
    config: RunConfig,
) -> FoldableMlpInitializer | None:
    tuning = config.distillation.foldable_mlp_multipliers
    if tuning.initializer_artifact is None:
        return None
    if tuning.initializer_sha256 is None:
        raise ValueError("foldable MLP initializer SHA-256 is missing")
    return load_foldable_mlp_initializer(
        tuning.initializer_artifact,
        expected_sha256=tuning.initializer_sha256,
        model_source=config.model.source,
        model_revision=str(config.model.revision),
    )


def _tensor_payload_hash(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _checkpoint_state(
    installed: InstalledMultipliers,
    optimizer: torch.optim.AdamW,
    *,
    protocol_hash: str,
    completed_steps: int,
    losses: list[float],
    destination: Path,
) -> None:
    tensors: dict[str, torch.Tensor] = {}
    inventory = []
    for index, (name, parameter) in enumerate(installed.named_parameters):
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"optimizer state is missing for foldable multiplier {name}")
        prefix = f"parameter_{index}"
        tensors[f"{prefix}.value"] = parameter.detach().cpu().contiguous()
        tensors[f"{prefix}.step"] = cast(torch.Tensor, state["step"]).detach().cpu().contiguous()
        tensors[f"{prefix}.exp_avg"] = cast(torch.Tensor, state["exp_avg"]).detach().cpu().contiguous()
        tensors[f"{prefix}.exp_avg_sq"] = cast(torch.Tensor, state["exp_avg_sq"]).detach().cpu().contiguous()
        inventory.append({"name": name, "prefix": prefix})
    with atomic_workspace(destination, replace_existing=True) as temporary:
        SAFETENSORS.save(tensors, temporary / "state.safetensors")
        atomic_write_json(
            temporary / "checkpoint.json",
            {
                "schema_version": 1,
                "protocol_hash": protocol_hash,
                "completed_steps": completed_steps,
                "losses": losses,
                "parameters": inventory,
            },
        )


def _restore_checkpoint(
    installed: InstalledMultipliers,
    optimizer: torch.optim.AdamW,
    *,
    protocol_hash: str,
    source: Path,
) -> tuple[int, list[float]]:
    manifest_path = source / "checkpoint.json"
    tensor_path = source / "state.safetensors"
    if not manifest_path.is_file() or not tensor_path.is_file():
        return 0, []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("protocol_hash") != protocol_hash:
        return 0, []
    expected = dict(installed.named_parameters)
    inventory = payload.get("parameters")
    if not isinstance(inventory, list) or {str(item["name"]) for item in inventory} != set(expected):
        raise ValueError("foldable MLP checkpoint parameter inventory differs")
    with SAFETENSORS.open(tensor_path) as handle, torch.no_grad():
        for item in inventory:
            name = str(item["name"])
            prefix = str(item["prefix"])
            parameter = expected[name]
            value = handle.get_tensor(f"{prefix}.value")
            if value.shape != parameter.shape:
                raise ValueError(f"foldable MLP checkpoint parameter shape differs: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            optimizer.state[parameter] = {
                "step": handle.get_tensor(f"{prefix}.step").cpu(),
                "exp_avg": handle.get_tensor(f"{prefix}.exp_avg").to(parameter.device),
                "exp_avg_sq": handle.get_tensor(f"{prefix}.exp_avg_sq").to(parameter.device),
            }
    completed = int(payload["completed_steps"])
    losses = [float(value) for value in payload["losses"]]
    if completed < 0 or len(losses) != completed:
        raise ValueError("foldable MLP checkpoint progress is invalid")
    return completed, losses


def _learning_rate(base: float, completed_steps: int, total_steps: int) -> float:
    return base * (1.0 + math.cos(math.pi * completed_steps / total_steps)) / 2.0


def _record_gradient_step(index: int, starting_step: int, total_steps: int) -> bool:
    return index == starting_step or index + 1 == total_steps


def _export_overlay(
    destination: Path,
    tensors: dict[str, torch.Tensor],
    *,
    source_component_sha256: str,
    frozen_identity: dict[str, str],
    global_tuning: ArtifactRef,
    protocol_hash: str,
    replaced_bytes: int,
) -> dict[str, object]:
    replacement_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    if replacement_bytes != replaced_bytes:
        raise ValueError("foldable MLP tuning changed represented payload bytes")
    with atomic_workspace(destination, replace_existing=True) as temporary:
        tensor_path = temporary / "components.safetensors"
        SAFETENSORS.save(tensors, tensor_path)
        manifest = {
            "schema_version": 2,
            "semantics": "replace-existing-factorized-components",
            "source_dense_tensor_sha256": source_component_sha256,
            "frozen_identity": frozen_identity,
            "global_tuning": to_dict(global_tuning),
            "policy": {
                str(index): "global-foldable-multipliers"
                for index in sorted({int(name.split(".")[2]) for name in tensors})
            },
            "protocol_hash": protocol_hash,
            "tensor_sha256": hash_file(tensor_path),
            "tensor_count": len(tensors),
            "replaced_payload_bytes": replaced_bytes,
            "replacement_payload_bytes": replacement_bytes,
            "payload_byte_delta": 0,
            "tensors": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
                for name, value in sorted(tensors.items())
            },
        }
        atomic_write_json(temporary / "manifest.json", manifest)
    return manifest


def active_foldable_mlp_tuning(
    run_output: str | Path,
    *,
    expected_protocol_hash: str | None = None,
) -> FoldableMlpTuningResult | None:
    run = Path(run_output)
    path = run / _ACTIVE_NAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("overlay") != _OVERLAY_DIRECTORY
        or payload.get("report") != _REPORT_NAME
    ):
        raise ValueError("active foldable MLP tuning receipt is invalid")
    protocol = str(payload["protocol_hash"])
    if expected_protocol_hash is not None and protocol != expected_protocol_hash:
        return None
    overlay = run / _OVERLAY_DIRECTORY
    report = run / _REPORT_NAME
    if not report.is_file():
        raise ValueError("active foldable MLP tuning report is missing")
    return FoldableMlpTuningResult(
        overlay,
        report,
        protocol,
        str(payload["tensor_sha256"]),
        int(payload["steps_completed"]),
        True,
    )


def _run(config: RunConfig, run_output: Path, snapshot: Path) -> FoldableMlpTuningResult:
    tuning = config.distillation.foldable_mlp_multipliers
    if not tuning.enabled or not config.distillation.enabled:
        raise ValueError("foldable MLP tuning requires both tuning and global distillation")
    global_tuning = active_global_tuning(run_output)
    if global_tuning is None:
        raise ValueError("foldable MLP tuning requires an active global tuning result")
    tokens = _load_calibration(run_output)
    frozen_identity = _commit_identity(run_output)
    token_hash = _token_hash(tokens)
    initializer = _load_initializer(config)
    protocol_hash = _protocol_hash(
        tuning,
        frozen_identity,
        global_tuning,
        token_hash,
        initializer,
    )
    active = active_foldable_mlp_tuning(run_output, expected_protocol_hash=protocol_hash)
    if active is not None:
        loaded_overlay = load_factorized_component_overlay(
            active.overlay,
            frozen_identity=frozen_identity,
            global_tuning=global_tuning,
        )
        if loaded_overlay.manifest["tensor_sha256"] != active.tensor_sha256:
            raise ValueError("active foldable MLP receipt differs from its component overlay")
        return active

    started = time.perf_counter()
    batches, teacher_cache_bytes = _training_batches(run_output, tuning.steps)
    loaded = load_frozen_run(
        run_output,
        snapshot,
        source_name=config.model.source,
        revision=str(config.model.revision),
        device=config.runtime.compute_device,
        backend="factorized",
        use_global_tuning=True,
    )
    student = loaded.model
    cast(Any, student).config.use_cache = False
    installed = install_global_mlp_multipliers(student)
    initialization: dict[str, object] = {"kind": "identity"}
    if initializer is not None:
        initializer_log_limit = math.log(tuning.initializer_multiplier_limit)
        consumed = seed_global_mlp_multipliers(
            installed,
            initializer.tensors,
            log_limit=initializer_log_limit,
        )
        seed_summary = multiplier_summary(installed, initializer_log_limit)
        seeded_tensors, seeded_replaced_bytes = fold_global_mlp_multipliers(
            student,
            installed,
        )
        initialization = {
            "kind": "artifact",
            "artifact": str(initializer.root.resolve()),
            "tensor_sha256": initializer.tensor_sha256,
            "manifest_sha256": hash_file(initializer.root / "manifest.json"),
            "seeded_axis_count": len(consumed),
            "seeded_axes": list(consumed),
            "seed_multiplier_summary": seed_summary,
            "seeded_component_sha256": _tensor_payload_hash(seeded_tensors),
            "seeded_replaced_payload_bytes": seeded_replaced_bytes,
        }
        installed = install_global_mlp_multipliers(student)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in installed.parameters:
        parameter.requires_grad_(True)
    if tuning.gradient_checkpointing:
        enable = getattr(student, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()
        enable_inputs = getattr(student, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
    parameters = list(installed.parameters)
    optimizer = torch.optim.AdamW(parameters, lr=tuning.learning_rate, weight_decay=0.0)
    checkpoint = run_output / _CHECKPOINT_DIRECTORY
    completed, losses = _restore_checkpoint(
        installed,
        optimizer,
        protocol_hash=protocol_hash,
        source=checkpoint,
    )
    if completed > tuning.steps:
        raise ValueError("foldable MLP checkpoint exceeds the configured step count")
    optimizer.param_groups[0]["lr"] = _learning_rate(tuning.learning_rate, completed, tuning.steps)
    cpu_tokens = tokens.detach().cpu()
    log_limit = math.log(tuning.multiplier_limit)
    gradient_checks: list[dict[str, object]] = []
    starting_step = completed
    student.train()
    for index in range(completed, tuning.steps):
        target = batches[index]
        indices = torch.tensor(target.sample_indices, dtype=torch.long)
        batch = cpu_tokens.index_select(0, indices).to(config.runtime.compute_device)
        selected = target.token_indices.to(device=config.runtime.compute_device, dtype=torch.long)
        hidden = _hidden_states(student, batch).reshape(-1, _hidden_width(student)).index_select(0, selected)
        kd_loss = topk_distillation_loss(
            hidden,
            target.top_values.to(config.runtime.compute_device),
            target.top_indices.to(device=config.runtime.compute_device, dtype=torch.long),
            _lm_head(student),
            temperature=config.distillation.temperature,
            token_chunk_size=config.distillation.token_chunk_size,
            token_weights=(
                None
                if target.token_weights is None
                else target.token_weights.to(config.runtime.compute_device)
            ),
        )
        penalty = family_identity_penalty(installed.families)
        loss = kd_loss + tuning.identity_penalty * penalty
        optimizer.zero_grad(set_to_none=True)
        cast(Any, loss).backward()
        coverage = gradient_summary(installed)
        if any(
            int(cast(Any, cast(dict[str, object], family)["missing_gradient_tensors"])) > 0
            or int(cast(Any, cast(dict[str, object], family)["nonfinite_gradient_tensors"])) > 0
            for family in coverage.values()
        ):
            raise FloatingPointError(f"foldable MLP multiplier gradient coverage is invalid: {coverage}")
        if _record_gradient_step(index, starting_step, tuning.steps):
            gradient_checks.append({"step": index + 1, "families": coverage})
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, tuning.gradient_clip)
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError("foldable MLP tuning produced a non-finite loss or gradient")
        optimizer.step()
        with torch.no_grad():
            for parameter in parameters:
                parameter.clamp_(min=-log_limit, max=log_limit)
        completed = index + 1
        losses.append(float(kd_loss.detach()))
        optimizer.param_groups[0]["lr"] = _learning_rate(
            tuning.learning_rate,
            completed,
            tuning.steps,
        )
        if completed % tuning.checkpoint_interval_steps == 0 or completed == tuning.steps:
            _checkpoint_state(
                installed,
                optimizer,
                protocol_hash=protocol_hash,
                completed_steps=completed,
                losses=losses,
                destination=checkpoint,
            )
        del batch, hidden, kd_loss, penalty, loss

    student.eval()
    replay_tokens = cpu_tokens[:1].to(config.runtime.compute_device)
    with torch.no_grad():
        unfolded = _hidden_states(student, replay_tokens).detach().cpu()
    source_components: dict[str, torch.Tensor] = {}
    for (block_index, path), wrapper in installed.wrappers.items():
        prefix = f"model.layers.{block_index}.{path}"
        source_components[f"{prefix}.scale_pre"] = wrapper.base.scale_pre
        source_components[f"{prefix}.scale_post"] = wrapper.base.scale_post
        for component in ("outlier_values", "patch_left", "patch_right"):
            value = getattr(wrapper.base, component)
            if isinstance(value, torch.Tensor):
                source_components[f"{prefix}.{component}"] = value
    source_component_sha256 = _tensor_payload_hash(source_components)
    tensors, replaced_bytes = fold_global_mlp_multipliers(student, installed)
    with torch.no_grad():
        folded = _hidden_states(student, replay_tokens).detach().cpu()
    replay_maximum_error = float((unfolded.float() - folded.float()).abs().max())
    if replay_maximum_error != 0.0:
        raise ValueError(
            "folded MLP components do not exactly replay the deployment-faithful tuning forward: "
            f"{replay_maximum_error}"
        )
    overlay = run_output / _OVERLAY_DIRECTORY
    manifest = _export_overlay(
        overlay,
        tensors,
        source_component_sha256=source_component_sha256,
        frozen_identity=frozen_identity,
        global_tuning=global_tuning,
        protocol_hash=protocol_hash,
        replaced_bytes=replaced_bytes,
    )
    report = run_output / _REPORT_NAME
    selected_parameter_count = sum(parameter.numel() for parameter in parameters)
    atomic_write_json(
        report,
        {
            "schema_version": 1,
            "role": "post-distillation foldable MLP multiplier tuning",
            "protocol_hash": protocol_hash,
            "config": to_dict(tuning),
            "source": {
                "frozen_identity": frozen_identity,
                "global_tuning": to_dict(global_tuning),
                "token_hash": token_hash,
                "teacher_cache_bytes_loaded": teacher_cache_bytes,
            },
            "initialization": initialization,
            "training": {
                "steps_completed": completed,
                "selected_parameter_count": selected_parameter_count,
                "kd_loss_first": losses[0],
                "kd_loss_final": losses[-1],
                "gradient_checks": gradient_checks,
                "multiplier_summary": multiplier_summary(installed, log_limit),
            },
            "folding": {
                "replay_maximum_absolute_error": replay_maximum_error,
                "replaced_payload_bytes": replaced_bytes,
                "replacement_payload_bytes": manifest["replacement_payload_bytes"],
                "payload_byte_delta": manifest["payload_byte_delta"],
                "tensor_sha256": manifest["tensor_sha256"],
            },
            "wall_seconds": time.perf_counter() - started,
        },
    )
    atomic_write_json(
        run_output / _ACTIVE_NAME,
        {
            "schema_version": 1,
            "protocol_hash": protocol_hash,
            "overlay": _OVERLAY_DIRECTORY,
            "report": _REPORT_NAME,
            "tensor_sha256": manifest["tensor_sha256"],
            "steps_completed": completed,
        },
    )
    del student, loaded, replay_tokens, unfolded, folded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return FoldableMlpTuningResult(
        overlay,
        report,
        protocol_hash,
        str(manifest["tensor_sha256"]),
        completed,
        False,
    )


def run_foldable_mlp_tuning(
    config: RunConfig,
    run_output: str | Path,
    snapshot: str | Path,
) -> FoldableMlpTuningResult:
    """Run or reuse the configured post-KD stage and return its active overlay."""

    device = config.runtime.compute_device
    if device.startswith("cuda"):
        with acquire_device_lease(device):
            return _run(config, Path(run_output), Path(snapshot))
    return _run(config, Path(run_output), Path(snapshot))


__all__ = [
    "FoldableMlpTuningResult",
    "active_foldable_mlp_tuning",
    "run_foldable_mlp_tuning",
]
