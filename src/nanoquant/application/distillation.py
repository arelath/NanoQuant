"""Memory-bounded top-k model distillation services."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import nn

from nanoquant.application.parity_adamw import (
    ParityAdamW,
    ParityAdamWState,
    capture_optimizer_state,
    restore_cosine_annealing_state,
    restore_optimizer_state,
)
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder

HiddenStatesFunction = Callable[[nn.Module, torch.Tensor], torch.Tensor]
ParameterSelector = Callable[[str, nn.Parameter], bool]


@dataclass(frozen=True, slots=True)
class TopKDistillationConfig:
    epochs: int = 8
    batch_size: int = 1
    learning_rate: float = 1e-5
    temperature: float = 1.0
    top_k: int = 64
    vocabulary_chunk_size: int = 8192
    token_chunk_size: int = 128
    maximum_tokens_per_batch: int | None = 512
    gradient_checkpointing: bool = True
    weight_decay: float = 0.0
    seed: int = 0
    optimizer_version: str = "legacy-optimi-adamw-v1"
    sampling_version: str = "legacy-python-device-rng-v1"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("distillation epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.temperature <= 0:
            raise ValueError("distillation learning rate and temperature must be positive")
        if self.top_k <= 0 or self.vocabulary_chunk_size <= 0 or self.token_chunk_size <= 0:
            raise ValueError("distillation chunk sizes and top-k must be positive")
        if self.maximum_tokens_per_batch is not None and self.maximum_tokens_per_batch <= 0:
            raise ValueError("distillation maximum tokens per batch must be positive when provided")
        if self.weight_decay < 0:
            raise ValueError("distillation weight decay cannot be negative")
        if self.optimizer_version != "legacy-optimi-adamw-v1":
            raise ValueError("unsupported distillation optimizer version")
        if self.sampling_version != "legacy-python-device-rng-v1":
            raise ValueError("unsupported distillation sampling version")


@dataclass(frozen=True, slots=True)
class TopKTeacherBatch:
    sample_indices: tuple[int, ...]
    token_indices: torch.Tensor
    top_values: torch.Tensor
    top_indices: torch.Tensor

    def __post_init__(self) -> None:
        if not self.sample_indices:
            raise ValueError("teacher target batch must contain sample indices")
        if self.token_indices.ndim != 1:
            raise ValueError("teacher target token indices must be rank one")
        if self.top_values.ndim != 2 or self.top_indices.shape != self.top_values.shape:
            raise ValueError("teacher target top-k tensors must be matching rank-two tensors")
        if self.top_values.shape[0] != self.token_indices.shape[0]:
            raise ValueError("teacher target token and top-k counts differ")
        if self.token_indices.device.type != "cpu" or self.top_values.device.type != "cpu":
            raise ValueError("teacher targets must be cached on CPU")
        if self.top_indices.device.type != "cpu" or self.top_indices.dtype is not torch.int32:
            raise ValueError("teacher target vocabulary indices must be CPU int32")


@dataclass(frozen=True, slots=True)
class TopKTeacherCache:
    epochs: tuple[tuple[TopKTeacherBatch, ...], ...]
    bytes: int


@dataclass(frozen=True, slots=True)
class DistillationMetrics:
    epoch_losses: tuple[float, ...]
    steps_completed: int
    selected_parameter_count: int
    teacher_cache_bytes: int


DistillationOptimizerState: TypeAlias = ParityAdamWState


@dataclass(frozen=True, slots=True)
class DistillationResumeState:
    completed_epochs: int
    epoch_losses: tuple[float, ...]
    steps_completed: int
    parameter_values: tuple[tuple[str, torch.Tensor], ...]
    optimizer_states: tuple[DistillationOptimizerState, ...]


DistillationCheckpointSink = Callable[[DistillationResumeState], None]


def select_kd_token_indices(
    mask: torch.Tensor,
    maximum_tokens: int | None,
    generator: torch.Generator,
) -> torch.Tensor:
    flat_mask = mask.reshape(-1).bool()
    valid = torch.nonzero(flat_mask, as_tuple=False).flatten()
    if maximum_tokens is not None and valid.numel() > maximum_tokens:
        order = torch.randperm(valid.numel(), generator=generator, device="cpu")[:maximum_tokens]
        valid = valid.index_select(0, order.to(valid.device))
    return valid


@torch.no_grad()
def teacher_topk_logits(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    *,
    top_k: int,
    vocabulary_chunk_size: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("distillation LM head must expose a rank-two weight")
    bias = getattr(lm_head, "bias", None)
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError("distillation LM-head bias is not a tensor")
    requested_k = min(top_k, weight.shape[0])
    best_values: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    for start in range(0, weight.shape[0], vocabulary_chunk_size):
        end = min(start + vocabulary_chunk_size, weight.shape[0])
        chunk_bias = None if bias is None else bias[start:end]
        logits = torch.nn.functional.linear(hidden_states, weight[start:end], chunk_bias)
        if temperature != 1.0:
            logits = logits / temperature
        chunk_k = min(requested_k, logits.shape[-1])
        values, indices = torch.topk(logits, chunk_k, dim=-1)
        indices = indices + start
        if best_values is None or best_indices is None:
            best_values, best_indices = values, indices
            continue
        merged_values = torch.cat((best_values, values), dim=-1)
        merged_indices = torch.cat((best_indices, indices), dim=-1)
        best_values, order = torch.topk(merged_values, requested_k, dim=-1)
        best_indices = merged_indices.gather(-1, order)
    if best_values is None or best_indices is None:
        raise ValueError("distillation LM head has an empty vocabulary")
    return best_values, best_indices


def selected_lm_head_logits(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    vocabulary_indices: torch.Tensor,
    *,
    token_chunk_size: int,
    temperature: float,
) -> torch.Tensor:
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("distillation LM head must expose a rank-two weight")
    bias = getattr(lm_head, "bias", None)
    pieces = []
    for start in range(0, hidden_states.shape[0], token_chunk_size):
        end = min(start + token_chunk_size, hidden_states.shape[0])
        indices = vocabulary_indices[start:end].to(device=hidden_states.device, dtype=torch.long)
        selected = weight.index_select(0, indices.reshape(-1)).view(indices.shape[0], indices.shape[1], -1)
        logits = torch.bmm(selected, hidden_states[start:end].unsqueeze(-1)).squeeze(-1)
        if isinstance(bias, torch.Tensor):
            logits = logits + bias.index_select(0, indices.reshape(-1)).view_as(indices)
        pieces.append(logits if temperature == 1.0 else logits / temperature)
    if not pieces:
        return hidden_states.new_empty((0, vocabulary_indices.shape[-1]))
    return torch.cat(pieces, dim=0)


def topk_distillation_loss(
    student_hidden_states: torch.Tensor,
    teacher_top_values: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    lm_head: nn.Module,
    *,
    temperature: float,
    token_chunk_size: int,
) -> torch.Tensor:
    if student_hidden_states.shape[0] == 0:
        return student_hidden_states.sum() * 0
    losses = []
    for start in range(0, student_hidden_states.shape[0], token_chunk_size):
        end = min(start + token_chunk_size, student_hidden_states.shape[0])
        student_logits = selected_lm_head_logits(
            student_hidden_states[start:end],
            lm_head,
            teacher_top_indices[start:end],
            token_chunk_size=token_chunk_size,
            temperature=temperature,
        )
        teacher_probabilities = torch.nn.functional.softmax(teacher_top_values[start:end].float(), dim=-1)
        student_log_probabilities = torch.nn.functional.log_softmax(student_logits.float(), dim=-1)
        losses.append(
            -(teacher_probabilities.to(student_log_probabilities.device) * student_log_probabilities).sum(dim=-1)
        )
    loss = torch.cat(losses).mean()
    return loss if temperature == 1.0 else loss * temperature**2


def cache_topk_teacher_targets(
    teacher: nn.Module,
    token_ids: torch.Tensor,
    lm_head: nn.Module,
    hidden_states: HiddenStatesFunction,
    config: TopKDistillationConfig,
    *,
    device: str | torch.device,
    pad_token_id: int | None,
    recorder: PhaseRecorder = NULL_RECORDER,
) -> TopKTeacherCache:
    epochs = []
    cache_bytes = 0
    for epoch_index in range(config.epochs):
        batches, epoch_bytes = cache_topk_teacher_epoch(
            teacher,
            token_ids,
            lm_head,
            hidden_states,
            config,
            epoch_index=epoch_index,
            device=device,
            pad_token_id=pad_token_id,
            recorder=recorder,
        )
        epochs.append(batches)
        cache_bytes += epoch_bytes
    return TopKTeacherCache(tuple(epochs), cache_bytes)


def cache_topk_teacher_epoch(
    teacher: nn.Module,
    token_ids: torch.Tensor,
    lm_head: nn.Module,
    hidden_states: HiddenStatesFunction,
    config: TopKDistillationConfig,
    *,
    epoch_index: int,
    device: str | torch.device,
    pad_token_id: int | None,
    recorder: PhaseRecorder = NULL_RECORDER,
) -> tuple[tuple[TopKTeacherBatch, ...], int]:
    if token_ids.ndim != 2 or token_ids.shape[0] == 0:
        raise ValueError("distillation token IDs must be a non-empty rank-two tensor")
    if epoch_index < 0 or epoch_index >= config.epochs:
        raise ValueError("distillation teacher-cache epoch index is out of range")
    with recorder.phase("planning"):
        cpu_tokens = token_ids.detach().cpu()
        # Match the legacy cache plan exactly. Hugging Face ``set_seed`` seeded
        # Python's RNG and the default RNG on the training device once; the legacy
        # loop then shuffled one persistent Python list and selected token positions
        # with ``torch.randperm(..., device=batch.device)`` across all epochs.
        # Replaying from the seed through ``epoch_index`` keeps independently
        # resumable epoch commits bitwise-compatible with that stateful loop.
        order_generator = random.Random(config.seed)
        generator_device = torch.device(device)
        token_generator = torch.Generator(device=generator_device).manual_seed(config.seed)
        order = list(range(cpu_tokens.shape[0]))
        epoch_plan: list[tuple[torch.Tensor, torch.Tensor]] = []
        for current_epoch in range(epoch_index + 1):
            order_generator.shuffle(order)
            for start in range(0, len(order), config.batch_size):
                indices = torch.tensor(order[start : start + config.batch_size], dtype=torch.long)
                batch = cpu_tokens.index_select(0, indices)
                mask = torch.ones_like(batch, dtype=torch.bool) if pad_token_id is None else batch != pad_token_id
                valid = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
                maximum_tokens = config.maximum_tokens_per_batch
                if maximum_tokens is not None and valid.numel() > maximum_tokens:
                    permutation = torch.randperm(
                        valid.numel(),
                        generator=token_generator,
                        device=generator_device,
                    )[:maximum_tokens].cpu()
                    selected = valid.index_select(0, permutation)
                else:
                    selected = valid
                if current_epoch == epoch_index and selected.numel() > 0:
                    epoch_plan.append((indices.clone(), selected))
    batches = []
    cache_bytes = 0
    teacher.eval()
    with torch.no_grad():
        for indices, selected in epoch_plan:
            if recorder is NULL_RECORDER:
                batch = cpu_tokens.index_select(0, indices).to(device)
                teacher_hidden = hidden_states(teacher, batch)
                teacher_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
                teacher_hidden = teacher_hidden.index_select(0, selected.to(teacher_hidden.device))
                values, vocabulary_indices = teacher_topk_logits(
                    teacher_hidden,
                    lm_head,
                    top_k=config.top_k,
                    vocabulary_chunk_size=config.vocabulary_chunk_size,
                    temperature=config.temperature,
                )
                selected_cpu = selected.cpu()
                values_cpu = values.cpu()
                vocabulary_indices_cpu = vocabulary_indices.to(device="cpu", dtype=torch.int32)
            else:
                with recorder.phase("h2d"):
                    batch = cpu_tokens.index_select(0, indices).to(device)
                with recorder.phase("forward"):
                    teacher_hidden = hidden_states(teacher, batch)
                    teacher_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
                    teacher_hidden = teacher_hidden.index_select(0, selected.to(teacher_hidden.device))
                with recorder.phase("topk"):
                    values, vocabulary_indices = teacher_topk_logits(
                        teacher_hidden,
                        lm_head,
                        top_k=config.top_k,
                        vocabulary_chunk_size=config.vocabulary_chunk_size,
                        temperature=config.temperature,
                    )
                with recorder.phase("d2h"):
                    selected_cpu = selected.cpu()
                    values_cpu = values.cpu()
                    vocabulary_indices_cpu = vocabulary_indices.to(device="cpu", dtype=torch.int32)
                recorder.add("distillation.teacher_batches", 1)
                recorder.add("distillation.teacher_tokens", selected.numel())
            cache_bytes += sum(
                value.numel() * value.element_size()
                for value in (selected_cpu, values_cpu, vocabulary_indices_cpu)
            )
            batches.append(
                TopKTeacherBatch(
                    tuple(int(index) for index in indices),
                    selected_cpu,
                    values_cpu,
                    vocabulary_indices_cpu,
                )
            )
    recorder.add("distillation.teacher_cache_bytes", cache_bytes)
    return tuple(batches), cache_bytes


def distill_topk(
    student: nn.Module,
    token_ids: torch.Tensor,
    lm_head: nn.Module,
    hidden_states: HiddenStatesFunction,
    teacher_cache: TopKTeacherCache,
    config: TopKDistillationConfig,
    selector: ParameterSelector,
    *,
    device: str | torch.device,
    resume: DistillationResumeState | None = None,
    checkpoint_sink: DistillationCheckpointSink | None = None,
    recorder: PhaseRecorder = NULL_RECORDER,
) -> DistillationMetrics:
    if len(teacher_cache.epochs) != config.epochs:
        raise ValueError("teacher target cache epoch count does not match distillation config")
    selected_parameters = [
        (name, parameter) for name, parameter in student.named_parameters() if selector(name, parameter)
    ]
    if not selected_parameters:
        raise ValueError("distillation selector chose no parameters")
    prior_requires_grad = {id(parameter): parameter.requires_grad for parameter in student.parameters()}
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for _name, parameter in selected_parameters:
        parameter.requires_grad_(True)
    selected_by_name = dict(selected_parameters)
    if resume is not None:
        if resume.completed_epochs < 0 or resume.completed_epochs > config.epochs:
            raise ValueError("distillation resume epoch is out of range")
        if len(resume.epoch_losses) != resume.completed_epochs:
            raise ValueError("distillation resume losses do not match completed epochs")
        if set(dict(resume.parameter_values)) != set(selected_by_name):
            raise ValueError("distillation resume parameters do not match the selector")
        with torch.no_grad():
            for name, value in resume.parameter_values:
                parameter = selected_by_name[name]
                if parameter.shape != value.shape:
                    raise ValueError(f"distillation resume parameter shape differs: {name}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    optimizer = ParityAdamW(
        [parameter for _name, parameter in selected_parameters],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = sum(len(epoch) for epoch in teacher_cache.epochs)
    starting_steps = 0 if resume is None else resume.steps_completed
    if starting_steps < 0 or starting_steps > total_steps:
        raise ValueError("distillation resume step is out of range")
    if resume is not None:
        restore_optimizer_state(
            optimizer,
            selected_parameters,
            resume.optimizer_states,
            starting_steps,
            operation="distillation",
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
    )
    restore_cosine_annealing_state(
        optimizer,
        scheduler,
        starting_steps,
        total_steps,
        config.learning_rate,
    )
    cpu_tokens = token_ids.detach().cpu()
    epoch_losses = [] if resume is None else list(resume.epoch_losses)
    steps = starting_steps
    starting_epoch = 0 if resume is None else resume.completed_epochs
    student.train()
    try:
        for epoch_index, epoch in enumerate(teacher_cache.epochs[starting_epoch:], start=starting_epoch):
            with recorder.phase("epoch", epoch=epoch_index):
                total_loss = 0.0
                for target in epoch:
                    if recorder is NULL_RECORDER:
                        sample_indices = torch.tensor(target.sample_indices, dtype=torch.long)
                        batch = cpu_tokens.index_select(0, sample_indices).to(device)
                        selected_tokens = target.token_indices.to(device=device, dtype=torch.long)
                        student_hidden = hidden_states(student, batch)
                        student_hidden = student_hidden.reshape(-1, student_hidden.shape[-1]).index_select(
                            0, selected_tokens
                        )
                        loss = topk_distillation_loss(
                            student_hidden,
                            target.top_values.to(device),
                            target.top_indices.to(device=device, dtype=torch.long),
                            lm_head,
                            temperature=config.temperature,
                            token_chunk_size=config.token_chunk_size,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        torch.autograd.backward(loss)
                        optimizer.step()
                        scheduler.step()
                    else:
                        with recorder.phase("h2d"):
                            sample_indices = torch.tensor(target.sample_indices, dtype=torch.long)
                            batch = cpu_tokens.index_select(0, sample_indices).to(device)
                            selected_tokens = target.token_indices.to(device=device, dtype=torch.long)
                        with recorder.phase("forward"):
                            student_hidden = hidden_states(student, batch)
                            student_hidden = student_hidden.reshape(-1, student_hidden.shape[-1]).index_select(
                                0, selected_tokens
                            )
                        with recorder.phase("loss"):
                            loss = topk_distillation_loss(
                                student_hidden,
                                target.top_values.to(device),
                                target.top_indices.to(device=device, dtype=torch.long),
                                lm_head,
                                temperature=config.temperature,
                                token_chunk_size=config.token_chunk_size,
                            )
                        with recorder.phase("zero_grad"):
                            optimizer.zero_grad(set_to_none=True)
                        with recorder.phase("backward"):
                            torch.autograd.backward(loss)
                        with recorder.phase("optimizer_step"):
                            optimizer.step()
                            scheduler.step()
                        recorder.add("distillation.steps", 1)
                        recorder.add("distillation.tokens", selected_tokens.numel())
                    total_loss += float(loss.detach())
                    steps += 1
                epoch_losses.append(total_loss / max(1, len(epoch)))
                if checkpoint_sink is not None:
                    with recorder.phase("checkpoint_snapshot"):
                        optimizer_states = capture_optimizer_state(optimizer, selected_parameters)
                        checkpoint_sink(
                            DistillationResumeState(
                                epoch_index + 1,
                                tuple(epoch_losses),
                                steps,
                                tuple(
                                    (name, parameter.detach().cpu().clone())
                                    for name, parameter in selected_parameters
                                ),
                                optimizer_states,
                            )
                        )
    finally:
        for parameter in student.parameters():
            parameter.requires_grad_(prior_requires_grad[id(parameter)])
        student.eval()
    return DistillationMetrics(tuple(epoch_losses), steps, len(selected_parameters), teacher_cache.bytes)
