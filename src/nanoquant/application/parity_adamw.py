"""Legacy-compatible AdamW updates used by NanoQuant tuning.

The original implementation used Optimi AdamW's debiased-beta recurrence and
Kahan-compensated updates for BF16/FP16 parameters. Keeping that behavior local
avoids making the auditable pipeline depend on an optimizer's optional kernels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from torch import Tensor
from torch.optim import Optimizer


def _debiased_beta(beta: float, step: int) -> float:
    return (beta**step - beta) / (beta**step - 1.0)


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
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if weight_decay:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                exp_avg.lerp_(gradient, weight=1.0 - beta1_hat)
                exp_avg_sq.mul_(beta2_hat).addcmul_(gradient, gradient, value=1.0 - beta2_hat)
                denominator = exp_avg_sq.sqrt().add_(epsilon)
                compensation = state["kahan_comp"]
                if isinstance(compensation, Tensor):
                    compensation.addcdiv_(exp_avg, denominator, value=-learning_rate)
                    gradient.copy_(parameter.detach())
                    parameter.add_(compensation)
                    compensation.add_(gradient.sub_(parameter))
                else:
                    parameter.addcdiv_(exp_avg, denominator, value=-learning_rate)
        return loss
