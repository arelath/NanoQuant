"""End-to-end top-k distillation of a complete committed frozen run."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.distillation import (
    DistillationMetrics,
    DistillationResumeState,
    TopKDistillationConfig,
    cache_topk_teacher_epoch,
    distill_topk,
)
from nanoquant.application.layers import BlockEditor, LayerFreezer, TrainableFactorizedLinear
from nanoquant.config.codec import canonical_json, to_dict
from nanoquant.domain.models import ArtifactRef, FrozenBlockState, FrozenNanoQuantState, GlobalTuningResult
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.distillation_cache import (
    TeacherCacheIdentity,
    commit_teacher_epoch,
    load_teacher_cache_journal,
    materialize_teacher_cache,
    record_teacher_epoch,
)
from nanoquant.infrastructure.distillation_checkpoint import (
    DistillationCheckpointIdentity,
    activate_distillation_checkpoint,
    active_distillation_checkpoint,
    commit_distillation_checkpoint,
)
from nanoquant.infrastructure.frozen_model_loader import LoadedFrozenModel, load_frozen_run
from nanoquant.infrastructure.global_tuning import activate_global_tuning, commit_global_tuning
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True, slots=True)
class GlobalDistillationRequest:
    run_output: Path
    snapshot: Path
    source: str
    revision: str
    token_ids: torch.Tensor | tuple[tuple[int, ...], ...]
    config: TopKDistillationConfig = TopKDistillationConfig()
    device: str = "cuda"
    pad_token_id: int | None = None
    verify_hashes: bool = True
    replace_existing_global_tuning: bool = False
    interrupt_after_epoch_commits: int | None = None
    initial_cooldown_seconds: float = 0.0
    epoch_cooldown_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class GlobalDistillationRunResult:
    reference: ArtifactRef
    result: GlobalTuningResult
    metrics: DistillationMetrics


def _tokens(value: torch.Tensor | tuple[tuple[int, ...], ...]) -> torch.Tensor:
    result = value.detach().cpu().long() if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("global distillation tokens must be a non-empty rank-two tensor")
    return result


def _checkpoint_dtype(snapshot: Path) -> torch.dtype:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(config.get("torch_dtype"), torch.float32)


def _decoder_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("model does not expose a mutable decoder layer stack")
    return tuple(layers)


def _hidden_states(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    text_stack = getattr(model, "model", None)
    if not isinstance(text_stack, nn.Module):
        language_model = getattr(model, "language_model", None)
        text_stack = getattr(language_model, "model", None)
    if not isinstance(text_stack, nn.Module):
        raise TypeError("model does not expose a supported text stack")
    outputs = cast(Any, text_stack)(input_ids=token_ids, use_cache=False)
    if isinstance(outputs, tuple):
        value = outputs[0]
    else:
        value = getattr(outputs, "last_hidden_state", None)
    if not isinstance(value, torch.Tensor):
        raise TypeError("model text stack did not return hidden states")
    return value


def _lm_head(model: nn.Module) -> nn.Module:
    value = getattr(model, "lm_head", None)
    if isinstance(value, nn.Module):
        return value
    output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(output_embeddings):
        value = output_embeddings()
    if not isinstance(value, nn.Module):
        raise TypeError("model does not expose an LM head")
    return value


def _storage_dtype(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }[name]
    except KeyError as exc:
        raise ValueError(f"unsupported global-tuning storage dtype: {name}") from exc


def _thaw_frozen_layers(
    loaded: LoadedFrozenModel,
    tensors: LocalTensorStore,
) -> dict[tuple[int, str], TrainableFactorizedLinear]:
    editor = BlockEditor()
    freezer = LayerFreezer()
    trainable: dict[tuple[int, str], TrainableFactorizedLinear] = {}
    for block_result, block in zip(loaded.blocks, _decoder_layers(loaded.model), strict=True):
        for state in block_result.frozen_state.quantized_layers:
            frozen = freezer.load(
                state,
                tensors,
                device="cpu",
                # Legacy NanoQuantLinear fixes its trainable factor, scale, and
                # salient paths to BF16.  Loading them as FP32 changes both the
                # optimizer recurrence (no Kahan compensation) and the cost of
                # every activation-dtype conversion in the factorized forward.
                dtype=torch.bfloat16,
                backend="factorized",
            ).module
            module = TrainableFactorizedLinear(
                frozen.left_binary,
                frozen.right_binary,
                frozen.scale_pre,
                frozen.scale_mid,
                frozen.scale_post,
                bias=frozen.bias,
                outlier_indices=frozen.outlier_indices,
                outlier_values=frozen.outlier_values,
                outlier_scales=frozen.outlier_scales,
                immutable_binary_factors=True,
            )
            editor.install_trainable_layer(block, state.layer.path, module)
            trainable[(state.layer.block.index, state.layer.path)] = module
    return trainable


def _selected_parameters(
    model: nn.Module,
    trainable: dict[tuple[int, str], TrainableFactorizedLinear],
) -> tuple[set[int], tuple[str, ...]]:
    selected = set()
    for module in trainable.values():
        for name, parameter in module.named_parameters():
            if name in {"scale_pre", "scale_mid", "scale_post", "outlier_values", "bias"}:
                selected.add(id(parameter))
    auxiliary = []
    for module_name, module in model.named_modules():
        if "norm" not in module.__class__.__name__.lower():
            continue
        for name, parameter in module.named_parameters(recurse=False):
            if parameter.ndim == 1 and name in {"weight", "bias"}:
                selected.add(id(parameter))
                auxiliary.append(f"{module_name}.{name}" if module_name else name)
    return selected, tuple(auxiliary)


def _restore_storage_dtype(module: TrainableFactorizedLinear, state: FrozenNanoQuantState) -> None:
    with torch.no_grad():
        module.scale_pre.data = module.scale_pre.data.to(_storage_dtype(state.scales.pre.spec.dtype))
        if state.scales.mid is None:
            raise ValueError("global distillation source state is missing its mid scale")
        module.scale_mid.data = module.scale_mid.data.to(_storage_dtype(state.scales.mid.spec.dtype))
        module.scale_post.data = module.scale_post.data.to(_storage_dtype(state.scales.post.spec.dtype))
        if module.outlier_values is not None and state.outliers is not None:
            module.outlier_values.data = module.outlier_values.data.to(_storage_dtype(state.outliers.values.spec.dtype))
        if module.bias is not None and state.bias is not None:
            module.bias.data = module.bias.data.to(_storage_dtype(state.bias.spec.dtype))


def _freeze_tuned_blocks(
    loaded: LoadedFrozenModel,
    trainable: dict[tuple[int, str], TrainableFactorizedLinear],
    tensors: LocalTensorStore,
) -> tuple[FrozenBlockState, ...]:
    freezer = LayerFreezer()
    result = []
    for block_result in loaded.blocks:
        states = []
        for state in block_result.frozen_state.quantized_layers:
            module = trainable[(state.layer.block.index, state.layer.path)].cpu()
            _restore_storage_dtype(module, state)
            states.append(freezer.freeze(state.layer, module, tensors, outliers=state.outliers).state)
        result.append(
            FrozenBlockState(
                block_result.block,
                tuple(states),
                block_result.frozen_state.passthrough_tensors,
                block_result.frozen_state.auxiliary_parameters,
            )
        )
    return tuple(result)


def _offload_student(student: nn.Module, device: str) -> None:
    """Finish CUDA work and release model allocations before the device lease."""

    student.cpu()
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _run_global_topk_distillation(request: GlobalDistillationRequest) -> GlobalDistillationRunResult:
    started = time.perf_counter()
    if request.interrupt_after_epoch_commits is not None and request.interrupt_after_epoch_commits <= 0:
        raise ValueError("distillation epoch interrupt count must be positive")
    if not 0.0 <= request.initial_cooldown_seconds < float("inf"):
        raise ValueError("distillation initial cooldown must be finite and non-negative")
    if not 0.0 <= request.epoch_cooldown_seconds < float("inf"):
        raise ValueError("distillation epoch cooldown must be finite and non-negative")
    if request.initial_cooldown_seconds:
        time.sleep(request.initial_cooldown_seconds)
    tokens = _tokens(request.token_ids)
    token_bytes = tokens.contiguous().view(torch.uint8).numpy().tobytes()
    protocol_hash = "sha256:" + hashlib.sha256(canonical_json(request.config).encode()).hexdigest()
    teacher_protocol = to_dict(request.config)
    if not isinstance(teacher_protocol, dict):
        raise TypeError("distillation config did not encode as an object")
    teacher_protocol.pop("optimizer_version")
    # Teacher-cache schema v1 included weight decay even though it cannot
    # affect teacher targets. Normalize it to the original protocol value so
    # the legacy-zero-decay correction can reuse the already committed cache.
    teacher_protocol["weight_decay"] = 0.01
    teacher_protocol_hash = "sha256:" + hashlib.sha256(canonical_json(teacher_protocol).encode()).hexdigest()
    token_hash = "sha256:" + hashlib.sha256(token_bytes).hexdigest()
    cache_identity = TeacherCacheIdentity(teacher_protocol_hash, token_hash)
    artifacts = LocalArtifactStore(request.run_output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    loaded = load_frozen_run(
        request.run_output,
        request.snapshot,
        source_name=request.source,
        revision=request.revision,
        device="cpu",
        verify_hashes=request.verify_hashes,
        backend="factorized",
        use_global_tuning=not request.replace_existing_global_tuning,
    )
    if loaded.global_tuning is not None:
        raise ValueError("run already has an active global tuning result")
    trainable = _thaw_frozen_layers(loaded, tensors)
    selected, auxiliary_names = _selected_parameters(loaded.model, trainable)

    if request.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(request.device)
    cache_journal = load_teacher_cache_journal(
        request.run_output,
        cache_identity,
        request.config.epochs,
        replace_mismatched=request.replace_existing_global_tuning,
    )
    if any(reference is None for reference in cache_journal.epochs):
        teacher = cast(
            nn.Module,
            AutoModelForCausalLM.from_pretrained(
                request.snapshot,
                local_files_only=True,
                torch_dtype=_checkpoint_dtype(request.snapshot),
                attn_implementation=cast(Any, loaded.model).config._attn_implementation,
            ),
        ).to(request.device)
        cast(Any, teacher).config.use_cache = False
        teacher_head = _lm_head(teacher)
        for epoch_index, reference in enumerate(cache_journal.epochs):
            if reference is not None:
                continue
            batches, _cache_bytes = cache_topk_teacher_epoch(
                teacher,
                tokens,
                teacher_head,
                _hidden_states,
                request.config,
                epoch_index=epoch_index,
                device=request.device,
                pad_token_id=request.pad_token_id,
            )
            committed_epoch = commit_teacher_epoch(epoch_index, batches, cache_identity, artifacts)
            cache_journal = record_teacher_epoch(
                request.run_output,
                cache_journal,
                epoch_index,
                committed_epoch.reference,
            )
        teacher.cpu()
        del teacher_head, teacher
        gc.collect()
        if request.device.startswith("cuda"):
            torch.cuda.empty_cache()
    teacher_cache = materialize_teacher_cache(cache_journal, artifacts)

    student = loaded.model
    cast(Any, student).config.use_cache = False
    if request.config.gradient_checkpointing:
        enable_checkpointing = getattr(student, "gradient_checkpointing_enable", None)
        if callable(enable_checkpointing):
            enable_checkpointing()
        enable_input_gradients = getattr(student, "enable_input_require_grads", None)
        if callable(enable_input_gradients):
            enable_input_gradients()
    student.to(request.device)
    checkpoint_identity = DistillationCheckpointIdentity(
        tuple(block.teacher_outputs.artifact for block in loaded.blocks),
        protocol_hash,
        token_hash,
    )
    active_checkpoint = active_distillation_checkpoint(request.run_output, checkpoint_identity, artifacts)

    epoch_commits = 0

    def checkpoint_sink(state: DistillationResumeState) -> None:
        nonlocal epoch_commits
        committed_checkpoint = commit_distillation_checkpoint(state, checkpoint_identity, artifacts)
        activate_distillation_checkpoint(request.run_output, committed_checkpoint.reference)
        epoch_commits += 1
        if (
            request.interrupt_after_epoch_commits is not None
            and epoch_commits >= request.interrupt_after_epoch_commits
        ):
            raise InterruptedError(
                f"requested interruption after {epoch_commits} distillation epoch checkpoint(s)"
            )
        if state.completed_epochs < request.config.epochs and request.epoch_cooldown_seconds:
            time.sleep(request.epoch_cooldown_seconds)

    try:
        metrics = distill_topk(
            student,
            tokens,
            _lm_head(student),
            _hidden_states,
            teacher_cache,
            request.config,
            lambda _name, parameter: id(parameter) in selected,
            device=request.device,
            resume=None if active_checkpoint is None else active_checkpoint.state,
            checkpoint_sink=checkpoint_sink,
        )
    except BaseException:
        try:
            _offload_student(student, request.device)
        except Exception:
            # Preserve the training/checkpoint exception. The lease still stays
            # held until this cleanup attempt returns.
            pass
        raise
    peak_gpu = int(torch.cuda.max_memory_allocated(request.device)) if request.device.startswith("cuda") else 0
    _offload_student(student, request.device)

    tuned_blocks = _freeze_tuned_blocks(loaded, trainable, tensors)
    parameter_map = dict(student.named_parameters())
    auxiliary_refs = tensors.put(
        "global-tuning-parameters",
        {name: parameter_map[name].detach().cpu() for name in auxiliary_names},
    )
    result = GlobalTuningResult(
        1,
        tuple(block.teacher_outputs.artifact for block in loaded.blocks),
        tuned_blocks,
        tuple((name, auxiliary_refs[name]) for name in auxiliary_names),
        protocol_hash,
        token_hash,
        metrics.epoch_losses,
        metrics.steps_completed,
        metrics.selected_parameter_count,
        metrics.teacher_cache_bytes,
        time.perf_counter() - started,
        peak_gpu,
        peak_process_memory_bytes(),
    )
    committed = commit_global_tuning(result, artifacts)
    activate_global_tuning(request.run_output, committed.reference)
    return GlobalDistillationRunResult(committed.reference, result, metrics)


def run_global_topk_distillation(request: GlobalDistillationRequest) -> GlobalDistillationRunResult:
    if request.device.startswith("cuda"):
        with acquire_device_lease(request.device):
            return _run_global_topk_distillation(request)
    return _run_global_topk_distillation(request)
