"""Independent typed non-factorized, factorized, and post-block tuning services."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import nn

from nanoquant.application.device_batches import iter_device_batches
from nanoquant.application.parity_adamw import (
    ParityAdamW,
    ParityAdamWState,
    capture_optimizer_state,
    restore_cosine_annealing_state,
    restore_optimizer_state,
)
from nanoquant.domain.models import LossMetrics, TuningMetrics
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder

ForwardFunction = Callable[[nn.Module, torch.Tensor], torch.Tensor]
ParameterSelector = Callable[[str, nn.Parameter], bool]
EpochLossMode: TypeAlias = Literal["full_evaluation", "legacy_training"]

_CUDA_CACHE_PRESSURE_FRACTION = 0.8


@dataclass(frozen=True, slots=True)
class TuningRequest:
    inputs: torch.Tensor
    targets: torch.Tensor
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    early_stop_relative_tolerance: float | None = None
    objective: str = "mean_squared_error"
    output_importance: torch.Tensor | None = None
    seed: int = 0
    microbatch_size: int | None = None
    epoch_observer: Callable[[int, float], None] | None = None
    restore_best_state: bool = True
    epoch_loss_mode: EpochLossMode = "full_evaluation"


TuningOptimizerState: TypeAlias = ParityAdamWState


@dataclass(frozen=True, slots=True)
class TuningResumeState:
    completed_epochs: int
    epoch_losses: tuple[float | None, ...]
    steps_completed: int
    parameter_values: tuple[tuple[str, torch.Tensor], ...]
    best_parameter_values: tuple[tuple[str, torch.Tensor], ...]
    optimizer_states: tuple[TuningOptimizerState, ...]
    best_epoch: int
    stopped_early: bool


TuningCheckpointSink = Callable[[TuningResumeState], None]


@dataclass(frozen=True, slots=True)
class _TrainingMicrobatch:
    indexes: torch.Tensor
    batch_elements: int
    starts_step: bool
    finishes_step: bool


@dataclass(frozen=True, slots=True)
class _StagedTrainingBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    slot: int | None


class _PinnedBatchStager:
    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        maximum_batch_size: int,
        device: torch.device,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.device = device
        self.copy_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        self.compute_stream = torch.cuda.current_stream(device)
        self.input_buffers = tuple(
            torch.empty(
                (maximum_batch_size, *inputs.shape[1:]),
                dtype=inputs.dtype,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        )
        self.target_buffers = tuple(
            torch.empty(
                (maximum_batch_size, *targets.shape[1:]),
                dtype=targets.dtype,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        )
        self.device_input_buffers = tuple(
            torch.empty(
                (maximum_batch_size, *inputs.shape[1:]),
                dtype=inputs.dtype,
                device=device,
            )
            for _ in range(2)
        )
        self.device_target_buffers = tuple(
            torch.empty(
                (maximum_batch_size, *targets.shape[1:]),
                dtype=targets.dtype,
                device=device,
            )
            for _ in range(2)
        )
        self.ready_events = tuple(torch.cuda.Event() for _ in range(2))  # type: ignore[no-untyped-call]
        self.consumed_events = tuple(torch.cuda.Event() for _ in range(2))  # type: ignore[no-untyped-call]
        self.ready_recorded = [False, False]
        self.consumed_recorded = [False, False]

    def _schedule(self, indexes: torch.Tensor, slot: int) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]:
        if self.ready_recorded[slot]:
            # The previous H2D copy has to stop reading this pinned host slot
            # before index_select refills it. This event normally completed while
            # the prior batch was computing and does not wait for compute itself.
            self.ready_events[slot].synchronize()
        count = indexes.numel()
        input_host = self.input_buffers[slot][:count]
        target_host = self.target_buffers[slot][:count]
        torch.index_select(self.inputs, 0, indexes, out=input_host)
        torch.index_select(self.targets, 0, indexes, out=target_host)
        with torch.cuda.stream(self.copy_stream):
            # Reusing fixed device slots avoids creating one allocator block per
            # queued microbatch. The copy stream waits on device consumption
            # without blocking the host, then overlaps this copy with compute in
            # the other slot.
            if self.consumed_recorded[slot]:
                self.copy_stream.wait_event(self.consumed_events[slot])
            input_batch = self.device_input_buffers[slot][:count]
            target_batch = self.device_target_buffers[slot][:count]
            input_batch.copy_(input_host, non_blocking=True)
            target_batch.copy_(target_host, non_blocking=True)
            self.ready_events[slot].record(self.copy_stream)
        self.ready_recorded[slot] = True
        self.consumed_recorded[slot] = False
        return input_batch, target_batch, self.ready_events[slot]

    def batches(self, indexes: tuple[torch.Tensor, ...]) -> Iterator[_StagedTrainingBatch]:
        if not indexes:
            return
        current = (*self._schedule(indexes[0], 0), 0)
        next_slot = 1
        for next_index in range(1, len(indexes) + 1):
            input_batch, target_batch, ready, slot = current
            self.compute_stream.wait_event(ready)
            input_batch.record_stream(self.compute_stream)
            target_batch.record_stream(self.compute_stream)
            following = (
                (*self._schedule(indexes[next_index], next_slot), next_slot)
                if next_index < len(indexes)
                else None
            )
            next_slot = (next_slot + 1) % len(self.ready_events)
            yield _StagedTrainingBatch(input_batch, target_batch, slot)
            if following is None:
                break
            current = following

    def mark_consumed(self, slot: int) -> None:
        self.consumed_events[slot].record(self.compute_stream)
        self.consumed_recorded[slot] = True

    def close(self) -> None:
        self.compute_stream.wait_stream(self.copy_stream)
        # The final yielded batch is normally marked consumed explicitly. A
        # synchronization here also closes the exceptional path where a caller
        # exits between yield and mark_consumed, and is free on the normal tuning
        # path because final loss materialization has already synchronized.
        self.compute_stream.synchronize()


def _pinned_batch_stager(
    request: TuningRequest, device: torch.device, maximum_batch_size: int
) -> _PinnedBatchStager | None:
    if (
        device.type != "cuda"
        or request.inputs.device.type != "cpu"
        or request.targets.device.type != "cpu"
    ):
        return None
    return _PinnedBatchStager(request.inputs, request.targets, maximum_batch_size, device)


def _training_microbatches(
    order: torch.Tensor, batch_size: int, microbatch_size: int | None
) -> tuple[_TrainingMicrobatch, ...]:
    batches = []
    for start in range(0, order.numel(), batch_size):
        indexes = order[start : start + batch_size]
        size = microbatch_size or indexes.numel()
        starts = tuple(range(0, indexes.numel(), size))
        for position, microbatch_start in enumerate(starts):
            batches.append(
                _TrainingMicrobatch(
                    indexes[microbatch_start : microbatch_start + size],
                    indexes.numel(),
                    position == 0,
                    position == len(starts) - 1,
                )
            )
    return tuple(batches)


def _resolve_output_importance(
    importance: torch.Tensor | None, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
    return None if importance is None else importance.to(device=device, dtype=dtype)


def _release_cuda_cache_under_pressure(device: torch.device) -> None:
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    if reserved >= total * _CUDA_CACHE_PRESSURE_FRACTION:
        torch.cuda.empty_cache()


def _synchronize_gradient_handoff(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def _loss_sum(prediction: torch.Tensor, target: torch.Tensor, importance: torch.Tensor | None) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("tuning prediction and target shapes differ")
    error = (prediction.float() - target.float()).square()
    if importance is not None:
        if importance.ndim != 1 or importance.shape[0] != prediction.shape[-1]:
            raise ValueError("tuning output importance must match the output feature dimension")
        # .to() is a no-op (same tensor, no copy) when already matching, so callers on a hot
        # loop should resolve device/dtype once with _resolve_output_importance and reuse it.
        error = error * importance.to(device=error.device, dtype=error.dtype)
    return error.sum()


def _logical_token_count(value: torch.Tensor) -> int:
    if value.ndim < 2 or value.shape[-1] == 0:
        return value.numel()
    return value.numel() // value.shape[-1]


def _record_transfer(
    recorder: PhaseRecorder,
    source_device: torch.device,
    destination_device: torch.device,
    *values: torch.Tensor,
) -> None:
    if source_device.type == "cpu" and destination_device.type == "cuda":
        recorder.add(
            "transfer.h2d_bytes",
            sum(value.numel() * value.element_size() for value in values),
        )


def _evaluate_loss(
    model: nn.Module,
    request: TuningRequest,
    forward: ForwardFunction,
    recorder: PhaseRecorder = NULL_RECORDER,
) -> float:
    parameter = next(iter(model.parameters()), None)
    device = request.inputs.device if parameter is None else parameter.device
    importance = _resolve_output_importance(request.output_importance, device, torch.float32)
    total = torch.zeros((), device=device)
    with torch.no_grad():
        evaluation_batch_size = request.microbatch_size or request.batch_size
        batches = iter(iter_device_batches((request.inputs, request.targets), evaluation_batch_size, device))
        while True:
            try:
                if recorder is NULL_RECORDER:
                    input_batch, target = next(batches)
                else:
                    with recorder.phase("batch_stage"):
                        input_batch, target = next(batches)
                        _record_transfer(recorder, request.inputs.device, device, input_batch, target)
            except StopIteration:
                break
            if recorder is NULL_RECORDER:
                prediction = forward(model, input_batch)
                total += _loss_sum(prediction, target, importance)
            else:
                with recorder.phase("forward"):
                    prediction = forward(model, input_batch)
                with recorder.phase("loss"):
                    total += _loss_sum(prediction, target, importance)
            del input_batch, prediction, target
    if recorder is NULL_RECORDER:
        return float(total / request.targets.numel())
    with recorder.phase("synchronize"):
        return float(total / request.targets.numel())


def _loss(model: nn.Module, request: TuningRequest, forward: ForwardFunction) -> torch.Tensor:
    prediction = forward(model, request.inputs)
    if prediction.shape != request.targets.shape:
        raise ValueError("tuning prediction and target shapes differ")
    return _loss_sum(prediction, request.targets, request.output_importance) / request.targets.numel()


def tune(
    model: nn.Module,
    request: TuningRequest,
    forward: ForwardFunction,
    selector: ParameterSelector,
    recorder: PhaseRecorder = NULL_RECORDER,
    *,
    resume: TuningResumeState | None = None,
    checkpoint_sink: TuningCheckpointSink | None = None,
) -> TuningMetrics:
    if (
        request.epochs < 0
        or request.batch_size <= 0
        or request.learning_rate <= 0
        or (request.microbatch_size is not None and request.microbatch_size <= 0)
    ):
        raise ValueError("invalid tuning loop settings")
    if request.epoch_loss_mode not in ("full_evaluation", "legacy_training"):
        raise ValueError(f"unsupported tuning epoch loss mode: {request.epoch_loss_mode}")
    if request.epoch_loss_mode == "legacy_training" and request.restore_best_state:
        raise ValueError("legacy training loss mode cannot restore a best evaluation state")
    selected = [(name, parameter) for name, parameter in model.named_parameters() if selector(name, parameter)]
    if not selected:
        raise ValueError("tuning selector chose no parameters")
    original_requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    selected_by_name = dict(selected)
    model_parameter = next(iter(model.parameters()), None)
    device = request.inputs.device if model_parameter is None else model_parameter.device
    importance = _resolve_output_importance(request.output_importance, device, torch.float32)
    omit_initial_evaluation = request.epoch_loss_mode == "legacy_training" and request.epochs > 0
    epoch_losses: list[float | None]
    if resume is not None:
        if resume.completed_epochs < 0 or resume.completed_epochs > request.epochs:
            raise ValueError("tuning resume epoch is out of range")
        if len(resume.epoch_losses) != resume.completed_epochs + 1:
            raise ValueError("tuning resume losses do not match completed epochs")
        if resume.best_epoch < -1 or resume.best_epoch >= resume.completed_epochs:
            raise ValueError("tuning resume best epoch is out of range")
        if (resume.epoch_losses[0] is None) != omit_initial_evaluation:
            raise ValueError("tuning resume initial loss mode differs from the request")
        if any(value is None for value in resume.epoch_losses[1:]):
            raise ValueError("tuning resume contains a missing completed-epoch loss")
        parameter_values = dict(resume.parameter_values)
        best_parameter_values = dict(resume.best_parameter_values)
        if set(parameter_values) != set(selected_by_name):
            raise ValueError("tuning resume parameters do not match the selector")
        if request.restore_best_state:
            if set(best_parameter_values) != set(selected_by_name):
                raise ValueError("tuning resume best parameters do not match the selector")
        elif best_parameter_values and set(best_parameter_values) != set(selected_by_name):
            raise ValueError("tuning resume best parameters do not match the selector")
        with torch.no_grad():
            for name, parameter in selected:
                value = parameter_values[name]
                if value.shape != parameter.shape:
                    raise ValueError(f"tuning resume parameter shape differs: {name}")
                if name in best_parameter_values and best_parameter_values[name].shape != parameter.shape:
                    raise ValueError(f"tuning resume best parameter shape differs: {name}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        before_value = resume.epoch_losses[0]
        observed_losses: list[tuple[int, float]] = [
            (index, value)
            for index, value in enumerate(resume.epoch_losses)
            if value is not None
        ]
        if not observed_losses:
            raise ValueError("tuning resume contains no observed loss")
        best_loss_index, best_value = min(observed_losses, key=lambda item: item[1])
        expected_best_epoch = best_loss_index - 1
        if resume.best_epoch != expected_best_epoch:
            raise ValueError("tuning resume best epoch disagrees with losses")
        best_epoch = resume.best_epoch
        best_state = (
            {
                name: value.to(device=selected_by_name[name].device, dtype=selected_by_name[name].dtype)
                for name, value in best_parameter_values.items()
            }
            if request.restore_best_state
            else None
        )
        epoch_losses = list(resume.epoch_losses)
    else:
        if omit_initial_evaluation:
            before_value = None
            best_value = math.inf
            best_epoch = -1
            best_state = None
            epoch_losses = [None]
        else:
            if recorder is NULL_RECORDER:
                before_value = _evaluate_loss(model, request, forward)
            else:
                with recorder.phase("initial_evaluation"):
                    before_value = _evaluate_loss(model, request, forward, recorder)
            best_value = before_value
            best_epoch = -1
            if request.restore_best_state and recorder is NULL_RECORDER:
                best_state = {name: parameter.detach().clone() for name, parameter in selected}
            elif request.restore_best_state:
                with recorder.phase("best_state_clone"):
                    best_state = {name: parameter.detach().clone() for name, parameter in selected}
                    recorder.add("tuning.best_state_clones", 1)
            else:
                best_state = None
            epoch_losses = [before_value]
    if request.epoch_observer is not None:
        for epoch, observed_loss in enumerate(epoch_losses):
            if observed_loss is not None:
                request.epoch_observer(epoch, observed_loss)
    optimizer = ParityAdamW(
        [parameter for _, parameter in selected], lr=request.learning_rate, weight_decay=request.weight_decay
    )
    steps_per_epoch = (request.inputs.shape[0] + request.batch_size - 1) // request.batch_size
    total_steps = max(1, request.epochs * steps_per_epoch)
    starting_steps = 0 if resume is None else resume.steps_completed
    if starting_steps != (0 if resume is None else resume.completed_epochs * steps_per_epoch):
        raise ValueError("tuning resume step count disagrees with completed epochs")
    if resume is not None:
        restore_optimizer_state(optimizer, selected, resume.optimizer_states, starting_steps, operation="tuning")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=request.learning_rate * 1e-4
    )
    restore_cosine_annealing_state(
        optimizer,
        scheduler,
        starting_steps,
        total_steps,
        request.learning_rate,
        eta_min=request.learning_rate * 1e-4,
    )
    generator = torch.Generator(device="cpu").manual_seed(request.seed)
    starting_epoch = 0 if resume is None else resume.completed_epochs
    for _epoch in range(starting_epoch):
        torch.randperm(request.inputs.shape[0], generator=generator)
    maximum_microbatch_size = min(request.inputs.shape[0], request.microbatch_size or request.batch_size)
    stager = _pinned_batch_stager(request, device, maximum_microbatch_size)
    epochs_completed = starting_epoch
    stopped_early = False if resume is None else resume.stopped_early
    try:
        for epoch in range(starting_epoch, request.epochs):
            if stopped_early:
                break
            with recorder.phase("epoch", epoch=epoch):
                epoch_loss_sum = (
                    torch.zeros((), device=device)
                    if request.epoch_loss_mode == "legacy_training"
                    else None
                )
                order = torch.randperm(request.inputs.shape[0], generator=generator)
                microbatches = _training_microbatches(order, request.batch_size, request.microbatch_size)
                device_batches: Iterator[_StagedTrainingBatch] = (
                    (
                        _StagedTrainingBatch(
                            request.inputs[item.indexes].to(device, non_blocking=True),
                            request.targets[item.indexes].to(device, non_blocking=True),
                            None,
                        )
                        for item in microbatches
                    )
                    if stager is None
                    else stager.batches(tuple(item.indexes for item in microbatches))
                )
                device_batch_iterator = iter(device_batches)
                for item in microbatches:
                    if recorder is NULL_RECORDER:
                        staged_batch = next(device_batch_iterator)
                    else:
                        with recorder.phase("batch_stage"):
                            staged_batch = next(device_batch_iterator)
                            _record_transfer(
                                recorder,
                                request.inputs.device,
                                device,
                                staged_batch.inputs,
                                staged_batch.targets,
                            )
                            recorder.add("tuning.tokens", _logical_token_count(staged_batch.inputs))
                    input_batch = staged_batch.inputs
                    target_batch = staged_batch.targets
                    if item.starts_step:
                        if recorder is NULL_RECORDER:
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            with recorder.phase("zero_grad"):
                                optimizer.zero_grad(set_to_none=True)
                    if recorder is NULL_RECORDER:
                        prediction = forward(model, input_batch)
                        loss_sum = _loss_sum(prediction, target_batch, importance)
                        loss = loss_sum / max(1, item.batch_elements)
                        torch.autograd.backward(loss)
                    else:
                        with recorder.phase("forward"):
                            prediction = forward(model, input_batch)
                        with recorder.phase("loss"):
                            loss_sum = _loss_sum(prediction, target_batch, importance)
                            loss = loss_sum / max(1, item.batch_elements)
                        with recorder.phase("backward"):
                            torch.autograd.backward(loss)
                    if epoch_loss_sum is not None:
                        epoch_loss_sum.add_(loss_sum.detach())
                    # Do not retain the final microbatch's autograd graph through
                    # optimizer/evaluation/factorization phase boundaries.
                    del input_batch, target_batch, prediction, loss_sum, loss
                    if stager is not None and staged_batch.slot is not None:
                        stager.mark_consumed(staged_batch.slot)
                    del staged_batch
                    if item.finishes_step:
                        # The production Gemma backward can still be completing
                        # asynchronously when the custom foreach optimizer starts.
                        # Materializing a loss happened to hide this race; make the
                        # required gradient handoff explicit instead.
                        _synchronize_gradient_handoff(device)
                        if recorder is NULL_RECORDER:
                            optimizer.step()
                            scheduler.step()
                        else:
                            with recorder.phase("optimizer_step"):
                                optimizer.step()
                                scheduler.step()
                                recorder.add("tuning.steps", 1)
                epochs_completed = epoch + 1
                if epoch_loss_sum is not None:
                    if recorder is NULL_RECORDER:
                        current = float(epoch_loss_sum / request.targets.numel())
                    else:
                        with recorder.phase("epoch_training_loss"):
                            current = float(epoch_loss_sum / request.targets.numel())
                    del epoch_loss_sum
                elif recorder is NULL_RECORDER:
                    current = _evaluate_loss(model, request, forward)
                else:
                    with recorder.phase("epoch_evaluation"):
                        current = _evaluate_loss(model, request, forward, recorder)
                if request.epoch_observer is not None:
                    request.epoch_observer(epoch + 1, current)
                epoch_losses.append(current)
                previous_epoch_loss = epoch_losses[-2] if len(epoch_losses) > 2 else None
                if current < best_value:
                    comparison_loss = (
                        previous_epoch_loss
                        if request.epoch_loss_mode == "legacy_training" and previous_epoch_loss is not None
                        else best_value
                    )
                    improvement = (comparison_loss - current) / max(abs(comparison_loss), 1e-12)
                    best_value = current
                    best_epoch = epoch
                    if request.restore_best_state and recorder is NULL_RECORDER:
                        best_state = {name: parameter.detach().clone() for name, parameter in selected}
                    elif request.restore_best_state:
                        with recorder.phase("best_state_clone"):
                            best_state = {name: parameter.detach().clone() for name, parameter in selected}
                            recorder.add("tuning.best_state_clones", 1)
                    if (
                        request.early_stop_relative_tolerance is not None
                        and (
                            request.epoch_loss_mode != "legacy_training"
                            or previous_epoch_loss is not None
                        )
                        and improvement < request.early_stop_relative_tolerance
                    ):
                        stopped_early = True
                elif (
                    request.epoch_loss_mode == "legacy_training"
                    and request.early_stop_relative_tolerance is not None
                    and previous_epoch_loss is not None
                ):
                    improvement = (previous_epoch_loss - current) / max(abs(previous_epoch_loss), 1e-12)
                    if improvement < request.early_stop_relative_tolerance:
                        stopped_early = True
                if checkpoint_sink is not None:
                    with recorder.phase("checkpoint_snapshot"):
                        optimizer_states = capture_optimizer_state(optimizer, selected)
                        checkpoint_sink(
                            TuningResumeState(
                                epoch + 1,
                                tuple(epoch_losses),
                                int(optimizer_states[0].step.item()),
                                tuple((name, parameter.detach().cpu().clone()) for name, parameter in selected),
                                (
                                    ()
                                    if best_state is None
                                    else tuple(
                                        (name, value.detach().cpu().clone())
                                        for name, value in best_state.items()
                                    )
                                ),
                                optimizer_states,
                                best_epoch,
                                stopped_early,
                            )
                        )
                if stopped_early:
                    break
        if request.restore_best_state:
            if best_state is None:
                raise AssertionError("best-state restoration requested without a best state")
            parameter_map = dict(model.named_parameters())
            if recorder is NULL_RECORDER:
                with torch.no_grad():
                    for name, value in best_state.items():
                        parameter_map[name].copy_(value)
            else:
                with recorder.phase("restore_best"):
                    with torch.no_grad():
                        for name, value in best_state.items():
                            parameter_map[name].copy_(value)
        if request.epoch_loss_mode == "legacy_training":
            final_value = epoch_losses[-1]
            if final_value is None:
                raise AssertionError("legacy tuning completed without an epoch loss")
        elif recorder is NULL_RECORDER:
            final_value = _evaluate_loss(model, request, forward)
        else:
            with recorder.phase("final_evaluation"):
                final_value = _evaluate_loss(model, request, forward, recorder)
    finally:
        if stager is not None:
            stager.close()
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
    elements = request.targets.numel()
    del optimizer, scheduler, best_state
    if device.type == "cuda":
        _release_cuda_cache_under_pressure(device)
    return TuningMetrics(
        None if before_value is None else LossMetrics(before_value, elements, request.objective),
        LossMetrics(best_value, elements, request.objective),
        LossMetrics(final_value, elements, request.objective),
        epochs_completed,
        best_epoch,
        stopped_early,
        None,
    )


def tune_non_factorized(
    model: nn.Module,
    request: TuningRequest,
    forward: ForwardFunction,
    recorder: PhaseRecorder = NULL_RECORDER,
    *,
    resume: TuningResumeState | None = None,
    checkpoint_sink: TuningCheckpointSink | None = None,
) -> TuningMetrics:
    factorized_prefixes = {
        name for name, module in model.named_modules() if module.__class__.__name__ == "TrainableFactorizedLinear"
    }
    return tune(
        model,
        request,
        forward,
        lambda name, _parameter: (
            not any(name == prefix or name.startswith(prefix + ".") for prefix in factorized_prefixes)
        ),
        recorder,
        resume=resume,
        checkpoint_sink=checkpoint_sink,
    )


def tune_factorized(
    model: nn.Module,
    module_path: str,
    request: TuningRequest,
    forward: ForwardFunction,
    recorder: PhaseRecorder = NULL_RECORDER,
    *,
    resume: TuningResumeState | None = None,
    checkpoint_sink: TuningCheckpointSink | None = None,
) -> TuningMetrics:
    prefix = module_path + "."
    return tune(
        model,
        request,
        forward,
        lambda name, _parameter: name.startswith(prefix),
        recorder,
        resume=resume,
        checkpoint_sink=checkpoint_sink,
    )


def post_block_refit(
    model: nn.Module,
    request: TuningRequest,
    forward: ForwardFunction,
    recorder: PhaseRecorder = NULL_RECORDER,
    *,
    resume: TuningResumeState | None = None,
    checkpoint_sink: TuningCheckpointSink | None = None,
) -> TuningMetrics:
    tunable_suffixes = ("scale_pre", "scale_mid", "scale_post", "outlier_values", "bias")
    return tune(
        model,
        request,
        forward,
        lambda name, _parameter: name.endswith(tunable_suffixes),
        recorder,
        resume=resume,
        checkpoint_sink=checkpoint_sink,
    )
