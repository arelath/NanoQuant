"""Independent typed non-factorized, factorized, and post-block tuning services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from nanoquant.domain.models import LossMetrics, TuningMetrics

ForwardFunction = Callable[[nn.Module, torch.Tensor], torch.Tensor]
ParameterSelector = Callable[[str, nn.Parameter], bool]


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


def _loss_sum(prediction: torch.Tensor, target: torch.Tensor, importance: torch.Tensor | None) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("tuning prediction and target shapes differ")
    error = (prediction.float() - target.float()).square()
    if importance is not None:
        if importance.ndim != 1 or importance.shape[0] != prediction.shape[-1]:
            raise ValueError("tuning output importance must match the output feature dimension")
        error = error * importance.to(device=error.device, dtype=error.dtype)
    return error.sum()


def _evaluate_loss(model: nn.Module, request: TuningRequest, forward: ForwardFunction) -> float:
    total = torch.zeros((), device=request.inputs.device)
    with torch.no_grad():
        for start in range(0, request.inputs.shape[0], request.batch_size):
            end = min(start + request.batch_size, request.inputs.shape[0])
            prediction = forward(model, request.inputs[start:end])
            total += _loss_sum(prediction, request.targets[start:end], request.output_importance)
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
    if request.epochs < 0 or request.batch_size <= 0 or request.learning_rate <= 0:
        raise ValueError("invalid tuning loop settings")
    selected = [(name, parameter) for name, parameter in model.named_parameters() if selector(name, parameter)]
    if not selected:
        raise ValueError("tuning selector chose no parameters")
    original_requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    before_value = _evaluate_loss(model, request, forward)
    best_value = before_value
    best_epoch = -1
    best_state = {name: parameter.detach().clone() for name, parameter in selected}
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected], lr=request.learning_rate, weight_decay=request.weight_decay
    )
    total_steps = max(1, request.epochs * ((request.inputs.shape[0] + request.batch_size - 1) // request.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=request.learning_rate * 1e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(request.seed)
    epochs_completed = 0
    stopped_early = False
    try:
        for epoch in range(request.epochs):
            order = torch.randperm(request.inputs.shape[0], generator=generator)
            for start in range(0, request.inputs.shape[0], request.batch_size):
                indexes = order[start : start + request.batch_size].to(request.inputs.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(model, request.inputs[indexes])
                loss = _loss_sum(prediction, request.targets[indexes], request.output_importance) / max(
                    1, indexes.numel()
                )
                torch.autograd.backward(loss)
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
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
    elements = request.targets.numel()
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
