"""Independent typed non-factorized, factorized, and post-block tuning services."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import nn

from nanoquant.application.device_batches import iter_device_batches
from nanoquant.application.parity_adamw import ParityAdamW
from nanoquant.domain.models import LossMetrics, TuningMetrics

ForwardFunction = Callable[[nn.Module, torch.Tensor], torch.Tensor]
ParameterSelector = Callable[[str, nn.Parameter], bool]

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


@dataclass(frozen=True, slots=True)
class _TrainingMicrobatch:
    indexes: torch.Tensor
    batch_elements: int
    starts_step: bool
    finishes_step: bool


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
        self.copy_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        self.compute_stream = torch.cuda.current_stream(device)
        self.events = tuple(torch.cuda.Event() for _ in range(2))  # type: ignore[no-untyped-call]
        self.recorded = [False, False]

    def _schedule(self, indexes: torch.Tensor, slot: int) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]:
        if self.recorded[slot]:
            self.events[slot].synchronize()
        count = indexes.numel()
        input_host = self.input_buffers[slot][:count]
        target_host = self.target_buffers[slot][:count]
        torch.index_select(self.inputs, 0, indexes, out=input_host)
        torch.index_select(self.targets, 0, indexes, out=target_host)
        with torch.cuda.stream(self.copy_stream):
            input_batch = input_host.to(self.device, non_blocking=True)
            target_batch = target_host.to(self.device, non_blocking=True)
            self.events[slot].record(self.copy_stream)
        self.recorded[slot] = True
        return input_batch, target_batch, self.events[slot]

    def batches(self, indexes: tuple[torch.Tensor, ...]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if not indexes:
            return
        current = self._schedule(indexes[0], 0)
        next_slot = 1
        for next_index in range(1, len(indexes) + 1):
            input_batch, target_batch, ready = current
            self.compute_stream.wait_event(ready)
            input_batch.record_stream(self.compute_stream)
            target_batch.record_stream(self.compute_stream)
            following = self._schedule(indexes[next_index], next_slot) if next_index < len(indexes) else None
            next_slot = (next_slot + 1) % len(self.events)
            yield input_batch, target_batch
            if following is None:
                break
            current = following

    def close(self) -> None:
        self.compute_stream.wait_stream(self.copy_stream)
        for event, recorded in zip(self.events, self.recorded, strict=True):
            if recorded:
                event.synchronize()


def _pinned_batch_stager(
    request: TuningRequest, device: torch.device, maximum_batch_size: int
) -> _PinnedBatchStager | None:
    if (
        device.type != "cuda"
        or request.inputs.device.type != "cpu"
        or request.targets.device.type != "cpu"
        or not request.inputs.is_pinned()
        or not request.targets.is_pinned()
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


def _evaluate_loss(model: nn.Module, request: TuningRequest, forward: ForwardFunction) -> float:
    parameter = next(iter(model.parameters()), None)
    device = request.inputs.device if parameter is None else parameter.device
    importance = _resolve_output_importance(request.output_importance, device, torch.float32)
    total = torch.zeros((), device=device)
    with torch.no_grad():
        evaluation_batch_size = request.microbatch_size or request.batch_size
        for input_batch, target in iter_device_batches(
            (request.inputs, request.targets), evaluation_batch_size, device
        ):
            prediction = forward(model, input_batch)
            total += _loss_sum(prediction, target, importance)
            del input_batch, prediction, target
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
) -> TuningMetrics:
    if (
        request.epochs < 0
        or request.batch_size <= 0
        or request.learning_rate <= 0
        or (request.microbatch_size is not None and request.microbatch_size <= 0)
    ):
        raise ValueError("invalid tuning loop settings")
    selected = [(name, parameter) for name, parameter in model.named_parameters() if selector(name, parameter)]
    if not selected:
        raise ValueError("tuning selector chose no parameters")
    original_requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    model_parameter = next(iter(model.parameters()), None)
    device = request.inputs.device if model_parameter is None else model_parameter.device
    importance = _resolve_output_importance(request.output_importance, device, torch.float32)
    before_value = _evaluate_loss(model, request, forward)
    best_value = before_value
    best_epoch = -1
    best_state = {name: parameter.detach().clone() for name, parameter in selected}
    optimizer = ParityAdamW(
        [parameter for _, parameter in selected], lr=request.learning_rate, weight_decay=request.weight_decay
    )
    total_steps = max(1, request.epochs * ((request.inputs.shape[0] + request.batch_size - 1) // request.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=request.learning_rate * 1e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(request.seed)
    maximum_microbatch_size = min(
        request.inputs.shape[0], request.microbatch_size or request.batch_size
    )
    stager = _pinned_batch_stager(request, device, maximum_microbatch_size)
    epochs_completed = 0
    stopped_early = False
    try:
        for epoch in range(request.epochs):
            order = torch.randperm(request.inputs.shape[0], generator=generator)
            microbatches = _training_microbatches(order, request.batch_size, request.microbatch_size)
            device_batches: Iterator[tuple[torch.Tensor, torch.Tensor]] = (
                (
                    request.inputs[item.indexes].to(device, non_blocking=True),
                    request.targets[item.indexes].to(device, non_blocking=True),
                )
                for item in microbatches
            ) if stager is None else stager.batches(tuple(item.indexes for item in microbatches))
            for item, (input_batch, target_batch) in zip(microbatches, device_batches, strict=True):
                if item.starts_step:
                    optimizer.zero_grad(set_to_none=True)
                prediction = forward(model, input_batch)
                loss = _loss_sum(prediction, target_batch, importance) / max(1, item.batch_elements)
                torch.autograd.backward(loss)
                # Do not retain the final microbatch's autograd graph through
                # optimizer/evaluation/factorization phase boundaries.
                del input_batch, target_batch, prediction, loss
                if item.finishes_step:
                    optimizer.step()
                    scheduler.step()
            epochs_completed = epoch + 1
            current = _evaluate_loss(model, request, forward)
            if current < best_value:
                improvement = (best_value - current) / max(abs(best_value), 1e-12)
                best_value = current
                best_epoch = epoch
                best_state = {name: parameter.detach().clone() for name, parameter in selected}
                if (
                    request.early_stop_relative_tolerance is not None
                    and improvement < request.early_stop_relative_tolerance
                ):
                    stopped_early = True
                    break
        parameter_map = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in best_state.items():
                parameter_map[name].copy_(value)
        final_value = _evaluate_loss(model, request, forward)
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
        LossMetrics(before_value, elements, request.objective),
        LossMetrics(best_value, elements, request.objective),
        LossMetrics(final_value, elements, request.objective),
        epochs_completed,
        best_epoch,
        stopped_early,
        None,
    )


def tune_non_factorized(model: nn.Module, request: TuningRequest, forward: ForwardFunction) -> TuningMetrics:
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
    )


def tune_factorized(
    model: nn.Module, module_path: str, request: TuningRequest, forward: ForwardFunction
) -> TuningMetrics:
    prefix = module_path + "."
    return tune(model, request, forward, lambda name, _parameter: name.startswith(prefix))


def post_block_refit(model: nn.Module, request: TuningRequest, forward: ForwardFunction) -> TuningMetrics:
    tunable_suffixes = ("scale_pre", "scale_mid", "scale_post", "outlier_values", "bias")
    return tune(model, request, forward, lambda name, _parameter: name.endswith(tunable_suffixes))
