"""Legacy-compatible AdamW updates used by NanoQuant tuning.

The original implementation used Optimi AdamW's debiased-beta recurrence and
Kahan-compensated updates for BF16/FP16 parameters. Keeping that behavior local
avoids making the auditable pipeline depend on an optimizer's optional kernels.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, overload

import torch
from torch import Tensor
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class ParityAdamWState:
    """Portable state for one named parameter in :class:`ParityAdamW`."""

    parameter_name: str
    step: Tensor
    exponential_average: Tensor
    exponential_average_squared: Tensor
    kahan_compensation: Tensor | None = None


def _debiased_beta(beta: float, step: int) -> float:
    return (beta**step - beta) / (beta**step - 1.0)


def _foreach_update(
    parameters: list[Tensor],
    gradients: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    denominators: list[Tensor],
    compensations: list[Tensor] | None,
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    weight_decay: float,
) -> None:
    """Apply the legacy elementwise recurrence with one launch per operation."""

    if weight_decay:
        torch._foreach_mul_(parameters, 1.0 - learning_rate * weight_decay)
    torch._foreach_lerp_(exp_avgs, gradients, weight=1.0 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, gradients, gradients, value=1.0 - beta2)
    torch._foreach_copy_(denominators, exp_avg_sqs)
    torch._foreach_sqrt_(denominators)
    torch._foreach_add_(denominators, epsilon)
    if compensations is not None:
        torch._foreach_addcdiv_(compensations, exp_avgs, denominators, value=-learning_rate)
        torch._foreach_copy_(gradients, parameters)
        torch._foreach_add_(parameters, compensations)
        torch._foreach_sub_(gradients, parameters)
        torch._foreach_add_(compensations, gradients)
    else:
        torch._foreach_addcdiv_(parameters, exp_avgs, denominators, value=-learning_rate)


class ParityAdamW(Optimizer):
    """AdamW with the numerical defaults and low-precision updates used by legacy NanoQuant."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("learning rate must be positive")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("optimizer betas must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("optimizer epsilon must be positive")
        if weight_decay < 0.0:
            raise ValueError("weight decay must be non-negative")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    @overload
    def step(self, closure: None = None) -> None: ...

    @torch.no_grad()
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            step = int(group.get("step", 0)) + 1
            group["step"] = step
            beta1, beta2 = group["betas"]
            beta1_hat = _debiased_beta(float(beta1), step)
            beta2_hat = _debiased_beta(float(beta2), step)
            learning_rate = float(group["lr"])
            epsilon = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            buckets: dict[
                tuple[torch.device, torch.dtype, bool],
                tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor]],
            ] = {}
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["kahan_comp"] = (
                        torch.zeros_like(parameter, memory_format=torch.preserve_format)
                        if parameter.dtype in {torch.float16, torch.bfloat16}
                        else None
                    )
                    state["denominator"] = torch.empty_like(parameter, memory_format=torch.preserve_format)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                compensation = state["kahan_comp"]
                denominator = state.get("denominator")
                if not isinstance(denominator, Tensor):
                    denominator = torch.empty_like(parameter, memory_format=torch.preserve_format)
                    state["denominator"] = denominator
                key = (parameter.device, parameter.dtype, isinstance(compensation, Tensor))
                bucket = buckets.setdefault(key, ([], [], [], [], [], []))
                bucket[0].append(parameter)
                bucket[1].append(gradient)
                bucket[2].append(exp_avg)
                bucket[3].append(exp_avg_sq)
                bucket[4].append(denominator)
                if isinstance(compensation, Tensor):
                    bucket[5].append(compensation)
            for parameters, gradients, exp_avgs, exp_avg_sqs, denominators, compensations in buckets.values():
                _foreach_update(
                    parameters,
                    gradients,
                    exp_avgs,
                    exp_avg_sqs,
                    denominators,
                    compensations if compensations else None,
                    beta1=beta1_hat,
                    beta2=beta2_hat,
                    learning_rate=learning_rate,
                    epsilon=epsilon,
                    weight_decay=weight_decay,
                )
        return loss


def restore_optimizer_state(
    optimizer: ParityAdamW,
    named_parameters: Iterable[tuple[str, Tensor]],
    states: Iterable[ParityAdamWState],
    completed_steps: int,
    *,
    operation: str,
) -> None:
    """Hydrate portable optimizer state onto each parameter's device."""

    parameters = tuple(named_parameters)
    state_by_name = {state.parameter_name: state for state in states}
    if set(state_by_name) != {name for name, _parameter in parameters}:
        raise ValueError(f"{operation} resume optimizer states do not match the selector")
    saved_steps = {int(state.step.item()) for state in state_by_name.values()}
    if saved_steps != {completed_steps}:
        raise ValueError(f"{operation} resume optimizer steps differ from completed steps")
    for name, parameter in parameters:
        state = state_by_name[name]
        if state.exponential_average.shape != parameter.shape:
            raise ValueError(f"{operation} resume optimizer shape differs: {name}")
        if parameter.dtype in {torch.float16, torch.bfloat16} and state.kahan_compensation is None:
            raise ValueError(f"{operation} resume Kahan state is missing: {name}")
        optimizer.state[parameter] = {
            "exp_avg": state.exponential_average.to(device=parameter.device, dtype=parameter.dtype),
            "exp_avg_sq": state.exponential_average_squared.to(
                device=parameter.device, dtype=parameter.dtype
            ),
            "kahan_comp": None
            if state.kahan_compensation is None
            else state.kahan_compensation.to(device=parameter.device, dtype=parameter.dtype),
            "denominator": torch.empty_like(parameter),
        }
    for group in optimizer.param_groups:
        group["step"] = completed_steps


def capture_optimizer_state(
    optimizer: ParityAdamW,
    named_parameters: Iterable[tuple[str, Tensor]],
) -> tuple[ParityAdamWState, ...]:
    """Clone optimizer state to CPU for a durable training checkpoint."""

    step = torch.tensor(int(optimizer.param_groups[0].get("step", 0)), dtype=torch.int64)
    captured = []
    for name, parameter in named_parameters:
        state = optimizer.state[parameter]
        exponential_average = state.get("exp_avg")
        exponential_average_squared = state.get("exp_avg_sq")
        compensation = state.get("kahan_comp")
        if not isinstance(exponential_average, Tensor) or not isinstance(exponential_average_squared, Tensor):
            raise ValueError(f"optimizer state is incomplete: {name}")
        captured.append(
            ParityAdamWState(
                name,
                step.clone(),
                exponential_average.detach().cpu().clone(),
                exponential_average_squared.detach().cpu().clone(),
                None if compensation is None else compensation.detach().cpu().clone(),
            )
        )
    return tuple(captured)


def restore_cosine_annealing_state(
    optimizer: ParityAdamW,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    completed_steps: int,
    total_steps: int,
    initial_learning_rate: float,
    *,
    eta_min: float = 0.0,
) -> None:
    """Restore the closed-form scheduler position without replaying steps."""

    if completed_steps == 0:
        return
    current_lr = eta_min + (initial_learning_rate - eta_min) * (
        1 + math.cos(math.pi * completed_steps / max(1, total_steps))
    ) / 2
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    scheduler_state = scheduler.state_dict()
    scheduler_state["last_epoch"] = completed_steps
    scheduler_state["_step_count"] = completed_steps + 1
    scheduler_state["_last_lr"] = [current_lr] * len(optimizer.param_groups)
    scheduler.load_state_dict(scheduler_state)
