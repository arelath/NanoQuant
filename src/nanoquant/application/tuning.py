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


def _loss(model: nn.Module, request: TuningRequest, forward: ForwardFunction) -> torch.Tensor:
    prediction = forward(model, request.inputs)
    if prediction.shape != request.targets.shape:
        raise ValueError("tuning prediction and target shapes differ")
    return (prediction.float() - request.targets.float()).square().mean()


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
    before_value = float(_loss(model, request, forward).detach())
    best_value = before_value
    best_epoch = -1
    best_state = {name: parameter.detach().clone() for name, parameter in selected}
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected], lr=request.learning_rate, weight_decay=request.weight_decay
    )
    epochs_completed = 0
    stopped_early = False
    try:
        for epoch in range(request.epochs):
            for start in range(0, request.inputs.shape[0], request.batch_size):
                end = min(start + request.batch_size, request.inputs.shape[0])
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(model, request.inputs[start:end])
                loss = (prediction.float() - request.targets[start:end].float()).square().mean()
                torch.autograd.backward(loss)
                optimizer.step()
            epochs_completed = epoch + 1
            current = float(_loss(model, request, forward).detach())
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
        final_value = float(_loss(model, request, forward).detach())
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
