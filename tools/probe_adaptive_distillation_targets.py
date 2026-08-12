"""Matched bounded screen for richer compressed distillation targets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_global_foldable_mlp_multipliers import (
    _checkpoint_dtype,
    _hidden_states,
    _hidden_states_width,
    _lm_head,
    _load_calibration,
    _load_training_cache,
)
from probe_mlp_overlays_kl import _split_tokens
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION
from probe_non_wikitext_kd_quality import C4_REVISION, _load_c4_tokens
from probe_topk_tail_mass import TailMassSums, _means, _tail_mass_sums
from torch import nn

from nanoquant.application.distillation import (
    TopKDistillationConfig,
    TopKTeacherBatch,
    adaptive_topk_tail_distillation_loss,
    cache_topk_teacher_epoch,
    multiband_tail_distillation_loss,
    topk_tail_distillation_loss,
    topk_tail_with_hard_labels_loss,
    variable_top_p_tail_distillation_loss,
)
from nanoquant.application.parity_adamw import ParityAdamW
from nanoquant.config.codec import to_dict
from nanoquant.global_distillation import _selected_parameters, _thaw_frozen_layers
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.reproducibility import deterministic_torch_execution
from nanoquant.infrastructure.tensor_store import LocalTensorStore

ARMS = ("fixed_0p5", "adaptive", "multiband", "variable_top_p", "hard_label")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-batches", type=int, default=32)
    parser.add_argument("--monitor-dataset", choices=("wikitext", "c4"), default="c4")
    parser.add_argument("--monitor-offset", type=int, default=440)
    parser.add_argument("--monitor-samples", type=int, default=8)
    parser.add_argument("--monitor-sequence-length", type=int, default=512)
    parser.add_argument("--c4-file", type=Path)
    parser.add_argument("--c4-documents", type=int, default=1100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _config(run_output: Path, maximum_batches: int) -> TopKDistillationConfig:
    manifest = json.loads((run_output / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["resolved_config"]["canonical_run_config"]["distillation"]
    names = {field.name for field in fields(TopKDistillationConfig)}
    values = {name: source[name] for name in names if name in source}
    values.update(
        {
            "objective": "top_k_tail",
            "epochs": 1,
            "maximum_batches_per_epoch": maximum_batches,
            "gradient_checkpointing": True,
        }
    )
    return TopKDistillationConfig(**values)


def _token_hash(tokens: torch.Tensor) -> str:
    payload = tokens.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _load_teacher(args: argparse.Namespace) -> nn.Module:
    model_config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(model_config)
    teacher = load_causal_language_model(
        args.snapshot,
        torch_dtype=_checkpoint_dtype(model_config),
        attention_implementation=adapter.attention_implementation,
        local_files_only=args.local_files_only,
    ).to(args.device)
    teacher.eval()
    return teacher


def _load_student(args: argparse.Namespace, config: TopKDistillationConfig) -> tuple[nn.Module, list[nn.Parameter]]:
    loaded = load_frozen_run(
        args.run_output,
        args.snapshot,
        source_name=MODEL_SOURCE,
        revision=PINNED_MODEL_REVISION,
        device="cpu",
        verify_hashes=False,
        backend="factorized",
        use_global_tuning=False,
    )
    artifacts = LocalArtifactStore(args.run_output / "artifacts")
    trainable = _thaw_frozen_layers(loaded, LocalTensorStore(artifacts))
    selected_ids, _auxiliary = _selected_parameters(loaded.model, trainable)
    selected = [parameter for parameter in loaded.model.parameters() if id(parameter) in selected_ids]
    if not selected:
        raise ValueError("adaptive distillation screen selected no trainable parameters")
    for parameter in loaded.model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    student = loaded.model
    cast(Any, student).config.use_cache = False
    if config.gradient_checkpointing:
        enable = getattr(student, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()
        enable_inputs = getattr(student, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
    student.to(args.device)
    return student, selected


def _validate_plans(
    fixed: tuple[TopKTeacherBatch, ...], rich: tuple[TopKTeacherBatch, ...]
) -> None:
    if len(fixed) != len(rich):
        raise ValueError("top-64 and top-256 teacher plans differ in length")
    for left, right in zip(fixed, rich, strict=True):
        if left.sample_indices != right.sample_indices or not torch.equal(
            left.token_indices, right.token_indices
        ):
            raise ValueError("top-64 and top-256 teacher token selections differ")
        if left.teacher_log_normalizers is None or right.teacher_log_normalizers is None:
            raise ValueError("adaptive target screen requires full-vocabulary teacher normalizers")
        if left.token_weights is not None or right.token_weights is not None:
            raise ValueError("adaptive target screen does not support weighted retained tokens")


@torch.no_grad()
def _monitor(
    student: nn.Module, teacher: nn.Module, tokens: torch.Tensor, *, top_k: int, device: str
) -> dict[str, object]:
    student.eval()
    teacher.eval()
    total = TailMassSums(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    per_sequence = []
    for row in tokens:
        batch = row.unsqueeze(0).to(device)
        teacher_logits = cast(Any, teacher)(input_ids=batch, use_cache=False).logits[:, :-1]
        student_logits = cast(Any, student)(input_ids=batch, use_cache=False).logits[:, :-1]
        current = _tail_mass_sums(teacher_logits, student_logits, batch[:, 1:], top_k=top_k)
        total += current
        per_sequence.append(_means(current))
        del batch, teacher_logits, student_logits
    result = _means(total)
    result["sequences"] = per_sequence
    return result


def _labels(batch: torch.Tensor, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = batch.shape[1]
    valid = selected.remainder(width) < width - 1
    flattened = batch.reshape(-1)
    labels = flattened.index_select(0, (selected + 1).clamp_max(flattened.numel() - 1))
    return labels, valid


def _objective_loss(
    arm: str,
    student: nn.Module,
    calibration: torch.Tensor,
    fixed: TopKTeacherBatch,
    rich: TopKTeacherBatch,
    config: TopKDistillationConfig,
    device: str,
) -> torch.Tensor:
    sample_indices = torch.tensor(fixed.sample_indices, dtype=torch.long)
    batch = calibration.index_select(0, sample_indices).to(device)
    selected = fixed.token_indices.to(device=device, dtype=torch.long)
    hidden = _hidden_states(student, batch).reshape(-1, _hidden_states_width(student))
    hidden = hidden.index_select(0, selected)
    common = {
        "temperature": config.temperature,
        "vocabulary_chunk_size": config.vocabulary_chunk_size,
        "token_chunk_size": config.token_chunk_size,
    }
    if arm == "fixed_0p5":
        return topk_tail_distillation_loss(
            hidden,
            fixed.top_values.to(device),
            fixed.top_indices.to(device=device, dtype=torch.long),
            cast(torch.Tensor, fixed.teacher_log_normalizers).to(device),
            _lm_head(student),
            mass_loss_weight=0.5,
            **common,
        )
    if arm == "adaptive":
        return adaptive_topk_tail_distillation_loss(
            hidden,
            fixed.top_values.to(device),
            fixed.top_indices.to(device=device, dtype=torch.long),
            cast(torch.Tensor, fixed.teacher_log_normalizers).to(device),
            _lm_head(student),
            minimum_mass_weight=0.25,
            maximum_mass_weight=1.0,
            **common,
        )
    if arm == "multiband":
        return multiband_tail_distillation_loss(
            hidden,
            rich.top_values.to(device),
            rich.top_indices.to(device=device, dtype=torch.long),
            cast(torch.Tensor, rich.teacher_log_normalizers).to(device),
            _lm_head(student),
            explicit_tokens=64,
            **common,
        )
    if arm == "variable_top_p":
        return variable_top_p_tail_distillation_loss(
            hidden,
            rich.top_values.to(device),
            rich.top_indices.to(device=device, dtype=torch.long),
            cast(torch.Tensor, rich.teacher_log_normalizers).to(device),
            _lm_head(student),
            probability=0.9,
            **common,
        )
    labels, label_mask = _labels(batch, selected)
    return topk_tail_with_hard_labels_loss(
        hidden,
        fixed.top_values.to(device),
        fixed.top_indices.to(device=device, dtype=torch.long),
        cast(torch.Tensor, fixed.teacher_log_normalizers).to(device),
        labels,
        _lm_head(student),
        hard_label_weight=0.1,
        hard_label_mask=label_mask,
        mass_loss_weight=0.5,
        **common,
    )


def _run_arm(
    args: argparse.Namespace,
    arm: str,
    config: TopKDistillationConfig,
    calibration: torch.Tensor,
    fixed: tuple[TopKTeacherBatch, ...],
    rich: tuple[TopKTeacherBatch, ...],
    monitor_tokens: torch.Tensor,
    teacher: nn.Module,
) -> dict[str, object]:
    student, parameters = _load_student(args, config)
    optimizer = ParityAdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(fixed))
    losses = []
    started = time.perf_counter()
    student.train()
    for index, (fixed_target, rich_target) in enumerate(zip(fixed, rich, strict=True), start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = _objective_loss(
            arm, student, calibration, fixed_target, rich_target, config, args.device
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if index == 1 or index % 4 == 0 or index == len(fixed):
            print(f"{arm}: {index}/{len(fixed)} loss={losses[-1]:.7f}", flush=True)
    monitor = _monitor(student, teacher, monitor_tokens, top_k=config.top_k, device=args.device)
    student.cpu()
    del student, parameters, optimizer, scheduler
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "steps": len(fixed),
        "training_loss_mean": statistics.fmean(losses),
        "training_loss_final": losses[-1],
        "monitor": monitor,
        "wall_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> int:
    if min(args.maximum_batches, args.monitor_samples, args.monitor_sequence_length - 1) <= 0:
        raise ValueError("adaptive distillation probe protocol is invalid")
    config = _config(args.run_output, args.maximum_batches)
    calibration = _load_calibration(args.run_output)
    fixed_cache = _load_training_cache(args.run_output, epochs=1)
    fixed = fixed_cache.epochs[0][: args.maximum_batches]
    if args.monitor_dataset == "c4":
        if args.c4_file is None:
            raise ValueError("C4 monitoring requires --c4-file")
        monitor_tokens, fingerprint, bos_id = _load_c4_tokens(
            args.snapshot,
            revision=C4_REVISION,
            data_file=str(args.c4_file),
            documents=args.c4_documents,
            offset=args.monitor_offset,
            samples=args.monitor_samples,
            sequence_length=args.monitor_sequence_length,
            local_files_only=args.local_files_only,
        )
    else:
        all_monitor, fingerprint, bos_id = _split_tokens(
            args.snapshot,
            split="validation",
            samples=args.monitor_offset + args.monitor_samples,
            sequence_length=args.monitor_sequence_length,
            local_files_only=args.local_files_only,
        )
        monitor_tokens = all_monitor[args.monitor_offset : args.monitor_offset + args.monitor_samples]
    protocol = {
        "schema_version": 1,
        "source_run": str(args.run_output.resolve()),
        "model_revision": PINNED_MODEL_REVISION,
        "distillation_config": to_dict(config),
        "arms": list(ARMS),
        "fixed_tail_mass_weight": 0.5,
        "adaptive_weight_range": [0.25, 1.0],
        "multiband": [64, 256, "remainder"],
        "variable_top_p": 0.9,
        "hard_label_weight": 0.1,
        "monitor_dataset": args.monitor_dataset,
        "monitor_offset": args.monitor_offset,
        "monitor_samples": args.monitor_samples,
        "monitor_sequence_length": args.monitor_sequence_length,
        "monitor_token_hash": _token_hash(monitor_tokens),
        "dataset_fingerprint": fingerprint,
        "bos_token_id": bos_id,
    }
    report = {
        "schema_version": 1,
        "status": "in_progress",
        "role": "analysis-only matched adaptive distillation target screen",
        "protocol": protocol,
        "results": {},
    }
    if args.output.is_file():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report.get("protocol") != protocol:
            raise ValueError("existing adaptive distillation report uses a different protocol")
    results = cast(dict[str, object], report["results"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with acquire_device_lease(args.device), deterministic_torch_execution(config.seed, args.device):
        teacher = _load_teacher(args)
        rich_config = replace(config, top_k=256)
        print("building matched top-256 teacher targets", flush=True)
        rich, _bytes = cache_topk_teacher_epoch(
            teacher,
            calibration,
            _lm_head(teacher),
            _hidden_states,
            rich_config,
            epoch_index=0,
            device=args.device,
            pad_token_id=None,
        )
        _validate_plans(fixed, rich)
        if "baseline" not in results:
            student, _parameters = _load_student(args, config)
            results["baseline"] = {
                "monitor": _monitor(
                    student, teacher, monitor_tokens, top_k=config.top_k, device=args.device
                )
            }
            student.cpu()
            del student, _parameters
            gc.collect()
            torch.cuda.empty_cache()
            atomic_write_json(args.output, report)
        for arm in ARMS:
            if arm in results:
                continue
            results[arm] = _run_arm(
                args, arm, config, calibration, fixed, rich, monitor_tokens, teacher
            )
            atomic_write_json(args.output, report)
        teacher.cpu()
        del teacher
    report["status"] = "complete"
    report["wall_seconds"] = sum(
        float(cast(dict[str, object], value).get("wall_seconds", 0.0))
        for value in results.values()
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
