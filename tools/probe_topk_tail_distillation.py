"""Resume-safe top-k-plus-tail KD ablation on a retained pre-KD frozen run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_composed_context_mlp_refit import _capture_mlp_context
from probe_global_foldable_mlp_multipliers import (
    _hidden_states,
    _lm_head,
    _load_calibration,
    _load_training_cache,
)
from probe_mlp_overlays_kl import _split_tokens
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION
from probe_topk_tail_mass import TailMassSums, _means, _tail_mass_sums
from torch import nn

from nanoquant.application.distillation import (
    DistillationResumeState,
    TopKDistillationConfig,
    TopKTeacherBatch,
    topk_distillation_loss,
    topk_tail_distillation_loss,
    vocabulary_logsumexp,
)
from nanoquant.application.parity_adamw import (
    ParityAdamW,
    capture_optimizer_state,
    restore_cosine_annealing_state,
    restore_optimizer_state,
)
from nanoquant.config.codec import semantic_hash, to_dict
from nanoquant.global_distillation import _selected_parameters, _thaw_frozen_layers
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.distillation_checkpoint import (
    DistillationCheckpointIdentity,
    activate_distillation_checkpoint,
    active_distillation_checkpoint,
    commit_distillation_checkpoint,
)
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.safetensors_io import SAFETENSORS
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.kl_budget_workflow import _token_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--maximum-batches-per-epoch", type=int)
    parser.add_argument("--mass-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--objective",
        choices=("topk_tail", "conditional_topk"),
        default="topk_tail",
    )
    parser.add_argument("--monitor-split", choices=("test", "validation"), default="validation")
    parser.add_argument("--monitor-offset", type=int, default=104)
    parser.add_argument("--monitor-samples", type=int, default=4)
    parser.add_argument("--monitor-sequence-length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--interrupt-after-epoch", type=int)
    return parser


def _distillation_config(run_output: Path, epochs: int) -> TopKDistillationConfig:
    manifest = json.loads((run_output / "manifest.json").read_text(encoding="utf-8"))
    payload = manifest["resolved_config"]["canonical_run_config"]["distillation"]
    names = {field.name for field in fields(TopKDistillationConfig)}
    values = {name: payload[name] for name in names if name in payload}
    values["epochs"] = epochs
    return TopKDistillationConfig(**values)


def _checkpoint_dtype(config: dict[str, object]) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(str(config.get("torch_dtype")), torch.float32)


def _normalizer_protocol(
    tokens: torch.Tensor,
    config: TopKDistillationConfig,
    revision: str,
    batches_by_epoch: tuple[tuple[TopKTeacherBatch, ...], ...] = (),
) -> dict[str, object]:
    target_digest = hashlib.sha256()
    for epoch_index, epoch in enumerate(batches_by_epoch):
        for batch_index, target in enumerate(epoch):
            target_digest.update(f"{epoch_index}:{batch_index}:".encode("ascii"))
            target_digest.update(json.dumps(target.sample_indices).encode("ascii"))
            target_digest.update(
                target.token_indices.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
    return {
        "schema_version": 1,
        "semantics": "teacher-selected-token-full-vocabulary-log-normalizer-v1",
        "model_revision": revision,
        "token_hash": _token_hash(tokens),
        "sample_count": tokens.shape[0],
        "sequence_length": tokens.shape[1],
        "temperature": config.temperature,
        "vocabulary_chunk_size": config.vocabulary_chunk_size,
        "token_chunk_size": config.token_chunk_size,
        "epoch_count": len(batches_by_epoch),
        "batch_count": sum(len(epoch) for epoch in batches_by_epoch),
        "target_selection_hash": "sha256:" + target_digest.hexdigest(),
    }


def _load_or_create_teacher_normalizers(
    output: Path,
    snapshot: Path,
    tokens: torch.Tensor,
    batches_by_epoch: tuple[tuple[TopKTeacherBatch, ...], ...],
    config: TopKDistillationConfig,
    model_revision: str,
    *,
    device: str,
    local_files_only: bool,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    cache_root = output / "teacher-normalizers"
    receipt_path = cache_root / "manifest.json"
    tensor_path = cache_root / "normalizers.safetensors"
    protocol = _normalizer_protocol(tokens, config, model_revision, batches_by_epoch)
    expected_keys = tuple(
        tuple(f"epoch-{epoch_index:04d}-batch-{batch_index:04d}" for batch_index in range(len(epoch)))
        for epoch_index, epoch in enumerate(batches_by_epoch)
    )
    if receipt_path.is_file() and tensor_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("protocol") != protocol or receipt.get("tensor_sha256") != hash_file(tensor_path):
            raise ValueError("retained teacher normalizers use a different protocol")
        tensors = SAFETENSORS.load(tensor_path)
        if set(tensors) != {key for epoch in expected_keys for key in epoch}:
            raise ValueError("retained teacher normalizer tensor inventory is invalid")
        values = tuple(tuple(tensors[key] for key in epoch) for epoch in expected_keys)
        for epoch, targets in zip(values, batches_by_epoch, strict=True):
            for value, target in zip(epoch, targets, strict=True):
                if (
                    value.shape != (target.top_values.shape[0],)
                    or value.dtype is not torch.float32
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("retained teacher normalizer tensor is invalid")
        return values
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(model_config)
    teacher = load_causal_language_model(
        snapshot,
        torch_dtype=_checkpoint_dtype(model_config),
        attention_implementation=adapter.attention_implementation,
        local_files_only=local_files_only,
    ).to(device)
    teacher.eval()
    values: list[tuple[torch.Tensor, ...]] = []
    with torch.no_grad():
        completed = 0
        total_batches = sum(len(epoch) for epoch in batches_by_epoch)
        for epoch in batches_by_epoch:
            rows = []
            for target in epoch:
                sample_indices = torch.tensor(target.sample_indices, dtype=torch.long)
                batch = tokens.index_select(0, sample_indices).to(device)
                selected_tokens = target.token_indices.to(device=device, dtype=torch.long)
                hidden = _hidden_states(teacher, batch).reshape(-1, _hidden_width(teacher))
                hidden = hidden.index_select(0, selected_tokens)
                rows.append(
                    vocabulary_logsumexp(
                        hidden,
                        _lm_head(teacher),
                        vocabulary_chunk_size=config.vocabulary_chunk_size,
                        token_chunk_size=config.token_chunk_size,
                        temperature=config.temperature,
                    ).cpu()
                )
                completed += 1
                print(f"teacher normalizers: {completed}/{total_batches} batches", flush=True)
            values.append(tuple(rows))
    teacher.cpu()
    del teacher
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    frozen_values = tuple(
        tuple(value.float().contiguous() for value in epoch) for epoch in values
    )
    output.mkdir(parents=True, exist_ok=True)
    with atomic_workspace(cache_root) as temporary:
        temporary_tensor = temporary / "normalizers.safetensors"
        SAFETENSORS.save(
            {
                key: value
                for keys, epoch in zip(expected_keys, frozen_values, strict=True)
                for key, value in zip(keys, epoch, strict=True)
            },
            temporary_tensor,
        )
        atomic_write_json(
            temporary / "manifest.json",
            {
                "schema_version": 1,
                "protocol": protocol,
                "tensor_sha256": hash_file(temporary_tensor),
                "bytes": temporary_tensor.stat().st_size,
            },
        )
    return frozen_values


def _hidden_width(model: nn.Module) -> int:
    width = getattr(getattr(model, "config", None), "hidden_size", None)
    if not isinstance(width, int):
        raise TypeError("model config does not expose hidden size")
    return width


def _target_normalizers(
    cached: tuple[tuple[torch.Tensor, ...], ...],
    epoch_index: int,
    batch_index: int,
    target: TopKTeacherBatch,
    *,
    device: str,
) -> torch.Tensor:
    selected = cached[epoch_index][batch_index]
    if selected.shape != (target.top_values.shape[0],):
        raise ValueError("teacher normalizer selection differs from cached top-k targets")
    return selected.to(device)


def _block_nrmse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return math.sqrt(
        float((candidate.float() - reference.float()).square().sum())
        / max(float(reference.float().square().sum()), 1e-30)
    )


def _monitor(
    student: nn.Module,
    snapshot: Path,
    tokens: torch.Tensor,
    config: TopKDistillationConfig,
    *,
    device: str,
    local_files_only: bool,
) -> dict[str, object]:
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(model_config)
    teacher = load_causal_language_model(
        snapshot,
        torch_dtype=_checkpoint_dtype(model_config),
        attention_implementation=adapter.attention_implementation,
        local_files_only=local_files_only,
    ).to(device)
    teacher.eval()
    student.eval()
    total = TailMassSums(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    with torch.no_grad():
        for row in tokens:
            batch = row.unsqueeze(0).to(device)
            teacher_logits = cast(Any, teacher)(input_ids=batch, use_cache=False).logits[:, :-1]
            student_logits = cast(Any, student)(input_ids=batch, use_cache=False).logits[:, :-1]
            total += _tail_mass_sums(
                teacher_logits,
                student_logits,
                batch[:, 1:],
                top_k=config.top_k,
            )
            del batch, teacher_logits, student_logits
    teacher_context = _capture_mlp_context(teacher, (24, 25), tokens, device=device)
    student_context = _capture_mlp_context(student, (24, 25), tokens, device=device)
    block_output_nrmse = {
        str(block): _block_nrmse(
            teacher_context.block_outputs[block],
            student_context.block_outputs[block],
        )
        for block in (24, 25)
    }
    teacher.cpu()
    del teacher, teacher_context, student_context
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"distribution": _means(total), "block_output_nrmse": block_output_nrmse}


def _report_protocol(
    args: argparse.Namespace,
    config: TopKDistillationConfig,
    calibration_tokens: torch.Tensor,
    monitor_tokens: torch.Tensor,
) -> dict[str, object]:
    source_manifest = json.loads((args.run_output / "manifest.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "objective": (
            "teacher-top-k-plus-aggregated-tail-v1"
            if args.objective == "topk_tail"
            else "teacher-conditional-top-k-control-v1"
        ),
        "mass_loss_weight": args.mass_loss_weight,
        "source_run_output": str(args.run_output.resolve()),
        "source_resident_config_hash": source_manifest["config_hash"],
        "model_revision": args.model_revision,
        "distillation_config": to_dict(config),
        "maximum_batches_per_epoch": args.maximum_batches_per_epoch,
        "calibration_token_hash": _token_hash(calibration_tokens),
        "teacher_cache_manifest_sha256": hash_file(
            args.run_output / "global-distillation-cache.json"
        ),
        "monitor_split": args.monitor_split,
        "monitor_offset": args.monitor_offset,
        "monitor_samples": args.monitor_samples,
        "monitor_sequence_length": args.monitor_sequence_length,
        "monitor_token_hash": _token_hash(monitor_tokens),
    }


def _load_report(path: Path, protocol: dict[str, object]) -> dict[str, object]:
    if not path.is_file():
        return {
            "schema_version": 1,
            "status": "in_progress",
            "role": "analysis-only global KD objective ablation",
            "protocol": protocol,
            "checkpoints": [],
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != protocol or not isinstance(report.get("checkpoints"), list):
        raise ValueError("existing top-k tail KD report uses a different protocol")
    return cast(dict[str, object], report)


def run(args: argparse.Namespace) -> int:
    if (
        min(
            args.epochs,
            args.monitor_samples,
            args.monitor_sequence_length - 1,
            args.mass_loss_weight,
        )
        <= 0
        or args.monitor_offset < 0
        or (
            args.maximum_batches_per_epoch is not None
            and args.maximum_batches_per_epoch <= 0
        )
        or (args.interrupt_after_epoch is not None and args.interrupt_after_epoch <= 0)
    ):
        raise ValueError("top-k tail distillation probe protocol is invalid")
    config = _distillation_config(args.run_output, args.epochs)
    calibration_tokens = _load_calibration(args.run_output)
    cache = _load_training_cache(args.run_output, epochs=args.epochs)
    batches_by_epoch = tuple(
        epoch
        if args.maximum_batches_per_epoch is None
        else epoch[: args.maximum_batches_per_epoch]
        for epoch in cache.epochs
    )
    all_monitor_tokens, monitor_fingerprint, monitor_bos = _split_tokens(
        args.snapshot,
        split=args.monitor_split,
        samples=args.monitor_offset + args.monitor_samples,
        sequence_length=args.monitor_sequence_length,
        local_files_only=args.local_files_only,
    )
    monitor_tokens = all_monitor_tokens[
        args.monitor_offset : args.monitor_offset + args.monitor_samples
    ]
    protocol = _report_protocol(args, config, calibration_tokens, monitor_tokens)
    report_path = args.output_directory / "report.json"
    report = _load_report(report_path, protocol)
    report["monitor_dataset_fingerprint"] = monitor_fingerprint
    report["monitor_bos_token_id"] = monitor_bos
    started = time.perf_counter()
    with acquire_device_lease(args.device):
        normalizers = (
            _load_or_create_teacher_normalizers(
                args.output_directory,
                args.snapshot,
                calibration_tokens,
                batches_by_epoch,
                config,
                args.model_revision,
                device=args.device,
                local_files_only=args.local_files_only,
            )
            if args.objective == "topk_tail"
            else None
        )
        run_artifacts = LocalArtifactStore(args.run_output / "artifacts")
        loaded = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name=MODEL_SOURCE,
            revision=args.model_revision,
            device="cpu",
            verify_hashes=False,
            backend="factorized",
            use_global_tuning=False,
        )
        trainable = _thaw_frozen_layers(loaded, LocalTensorStore(run_artifacts))
        selected_ids, _auxiliary = _selected_parameters(loaded.model, trainable)
        selected_parameters = [
            (name, parameter)
            for name, parameter in loaded.model.named_parameters()
            if id(parameter) in selected_ids
        ]
        if not selected_parameters:
            raise ValueError("top-k tail KD selected no trainable parameters")
        checkpoint_artifacts = LocalArtifactStore(args.output_directory / "artifacts")
        token_bytes = calibration_tokens.contiguous().view(torch.uint8).numpy().tobytes()
        token_hash = "sha256:" + hashlib.sha256(token_bytes).hexdigest()
        identity = DistillationCheckpointIdentity(
            tuple(block.teacher_outputs.artifact for block in loaded.blocks),
            semantic_hash(protocol),
            token_hash,
            semantic_hash({"objective": protocol["objective"]}),
        )
        checkpoint = active_distillation_checkpoint(
            args.output_directory,
            identity,
            checkpoint_artifacts,
        )
        for parameter in loaded.model.parameters():
            parameter.requires_grad_(False)
        for _name, parameter in selected_parameters:
            parameter.requires_grad_(True)
        if checkpoint is not None:
            values = dict(checkpoint.state.parameter_values)
            if set(values) != {name for name, _parameter in selected_parameters}:
                raise ValueError("top-k tail KD resume parameters differ from the selector")
            with torch.no_grad():
                for name, parameter in selected_parameters:
                    parameter.copy_(values[name].to(dtype=parameter.dtype))
        student = loaded.model
        cast(Any, student).config.use_cache = False
        if config.gradient_checkpointing:
            enable_checkpointing = getattr(student, "gradient_checkpointing_enable", None)
            if callable(enable_checkpointing):
                enable_checkpointing()
            enable_inputs = getattr(student, "enable_input_require_grads", None)
            if callable(enable_inputs):
                enable_inputs()
        student.to(args.device)
        optimizer = ParityAdamW(
            [parameter for _name, parameter in selected_parameters],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = sum(len(epoch) for epoch in batches_by_epoch)
        starting_epoch = 0
        steps = 0
        epoch_losses: list[float] = []
        if checkpoint is not None:
            starting_epoch = checkpoint.state.completed_epochs
            steps = checkpoint.state.steps_completed
            epoch_losses = list(checkpoint.state.epoch_losses)
            restore_optimizer_state(
                optimizer,
                selected_parameters,
                checkpoint.state.optimizer_states,
                steps,
                operation="top-k tail distillation",
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
        )
        restore_cosine_annealing_state(
            optimizer,
            scheduler,
            steps,
            total_steps,
            config.learning_rate,
        )
        checkpoints = cast(list[dict[str, object]], report["checkpoints"])
        reported_epoch = max((int(item["epoch"]) for item in checkpoints), default=-1)
        if reported_epoch > starting_epoch:
            raise ValueError("top-k tail KD report is ahead of its active checkpoint")
        if checkpoint is None and not checkpoints:
            checkpoints.append(
                {
                    "epoch": 0,
                    "steps": 0,
                    "training_loss": None,
                    "monitor": _monitor(
                        student,
                        args.snapshot,
                        monitor_tokens,
                        config,
                        device=args.device,
                        local_files_only=args.local_files_only,
                    ),
                }
            )
            atomic_write_json(report_path, report)
        elif checkpoint is not None and reported_epoch < starting_epoch:
            # The checkpoint is authoritative if power was lost after its
            # activation but before the corresponding report update.
            checkpoints.append(
                {
                    "epoch": starting_epoch,
                    "steps": steps,
                    "training_loss": epoch_losses[-1],
                    "checkpoint": to_dict(checkpoint.reference),
                    "monitor": _monitor(
                        student,
                        args.snapshot,
                        monitor_tokens,
                        config,
                        device=args.device,
                        local_files_only=args.local_files_only,
                    ),
                    "recovered_from_active_checkpoint": True,
                }
            )
            atomic_write_json(report_path, report)
        for epoch_index in range(starting_epoch, config.epochs):
            student.train()
            total_loss = 0.0
            epoch_batches = batches_by_epoch[epoch_index]
            for batch_index, target in enumerate(epoch_batches):
                sample_indices = torch.tensor(target.sample_indices, dtype=torch.long)
                batch = calibration_tokens.index_select(0, sample_indices).to(args.device)
                selected_tokens = target.token_indices.to(device=args.device, dtype=torch.long)
                hidden = _hidden_states(student, batch).reshape(-1, _hidden_width(student))
                hidden = hidden.index_select(0, selected_tokens)
                common = {
                    "temperature": config.temperature,
                    "token_chunk_size": config.token_chunk_size,
                    "token_weights": (
                        None if target.token_weights is None else target.token_weights.to(args.device)
                    ),
                }
                loss = (
                    topk_tail_distillation_loss(
                        hidden,
                        target.top_values.to(args.device),
                        target.top_indices.to(device=args.device, dtype=torch.long),
                        _target_normalizers(
                            cast(tuple[tuple[torch.Tensor, ...], ...], normalizers),
                            epoch_index,
                            batch_index,
                            target,
                            device=args.device,
                        ),
                        _lm_head(student),
                        vocabulary_chunk_size=config.vocabulary_chunk_size,
                        mass_loss_weight=args.mass_loss_weight,
                        **common,
                    )
                    if args.objective == "topk_tail"
                    else topk_distillation_loss(
                        hidden,
                        target.top_values.to(args.device),
                        target.top_indices.to(device=args.device, dtype=torch.long),
                        _lm_head(student),
                        **common,
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                torch.autograd.backward(loss)
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.detach())
                steps += 1
                del batch, selected_tokens, hidden, loss
            epoch_loss = total_loss / len(epoch_batches)
            epoch_losses.append(epoch_loss)
            state = DistillationResumeState(
                epoch_index + 1,
                tuple(epoch_losses),
                steps,
                tuple(
                    (name, parameter.detach().cpu().clone())
                    for name, parameter in selected_parameters
                ),
                capture_optimizer_state(optimizer, selected_parameters),
            )
            committed = commit_distillation_checkpoint(state, identity, checkpoint_artifacts)
            activate_distillation_checkpoint(args.output_directory, committed.reference)
            checkpoints.append(
                {
                    "epoch": epoch_index + 1,
                    "steps": steps,
                    "training_loss": epoch_loss,
                    "checkpoint": to_dict(committed.reference),
                    "monitor": _monitor(
                        student,
                        args.snapshot,
                        monitor_tokens,
                        config,
                        device=args.device,
                        local_files_only=args.local_files_only,
                    ),
                }
            )
            report["steps_completed"] = steps
            report["epoch_losses"] = epoch_losses
            report["wall_seconds"] = time.perf_counter() - started
            atomic_write_json(report_path, report)
            objective_label = (
                "tail-bucket" if args.objective == "topk_tail" else "conditional top-k"
            )
            print(
                f"{objective_label} KD epoch {epoch_index + 1}/{config.epochs}: "
                f"loss={epoch_loss:.6f}, steps={steps}",
                flush=True,
            )
            if args.interrupt_after_epoch == epoch_index + 1:
                raise InterruptedError(
                    f"requested interruption after tail-bucket epoch {epoch_index + 1}"
                )
        report["status"] = "completed"
        report["steps_completed"] = steps
        report["epoch_losses"] = epoch_losses
        report["selected_parameter_count"] = len(selected_parameters)
        report["teacher_cache_bytes"] = cache.bytes
        report["wall_seconds"] = time.perf_counter() - started
        atomic_write_json(report_path, report)
        student.cpu()
        del student, loaded, optimizer
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
