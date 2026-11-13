"""FP32 scale masters and portable state for opt-in native fused AdamW tuning."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn

from nanoquant.application.parity_adamw import ParityAdamWState


@contextmanager
def fp32_scale_masters(module: nn.Module) -> Iterator[None]:
    parameters = tuple(
        (parameter, parameter.dtype)
        for name, parameter in module.named_parameters()
        if name.split(".")[-1] in {"scale_pre", "scale_mid", "scale_post"}
    )
    try:
        for parameter, _dtype in parameters:
            parameter.data = parameter.data.float()
        yield
    finally:
        # The trainable forward casts scales to activation precision. Freeze and
        # subsequent parity stages retain the original storage precision/BPW.
        for parameter, dtype in parameters:
            parameter.grad = None
            parameter.data = parameter.data.to(dtype=dtype)


def capture_fused_state(
    optimizer: torch.optim.Optimizer, named_parameters: Iterable[tuple[str, Tensor]],
) -> tuple[ParityAdamWState, ...]:
    result = []
    for name, parameter in named_parameters:
        state = optimizer.state[parameter]
        if not all(isinstance(state.get(key), Tensor) for key in ("step", "exp_avg", "exp_avg_sq")):
            raise ValueError(f"fused optimizer state is incomplete: {name}")
        result.append(ParityAdamWState(
            name, state["step"].detach().cpu().to(torch.int64),
            state["exp_avg"].detach().cpu().clone(), state["exp_avg_sq"].detach().cpu().clone(),
        ))
    return tuple(result)


def restore_fused_state(
    optimizer: torch.optim.Optimizer, named_parameters: Iterable[tuple[str, Tensor]],
    states: Iterable[ParityAdamWState], completed_steps: int,
) -> None:
    parameters = tuple(named_parameters)
    saved = tuple(states)
    by_name = {state.parameter_name: state for state in saved}
    if len(by_name) != len(saved) or set(by_name) != {name for name, _parameter in parameters}:
        raise ValueError("fused optimizer resume parameter inventory differs")
    for name, parameter in parameters:
        state = by_name[name]
        if state.kahan_compensation is not None:
            raise ValueError("cannot restore parity optimizer state into fused AdamW")
        if state.step.numel() != 1 or int(state.step.item()) != completed_steps:
            raise ValueError("fused optimizer resume step count differs")
        if any(moment.shape != parameter.shape or moment.dtype != parameter.dtype for moment in (
            state.exponential_average, state.exponential_average_squared,
        )):
            raise ValueError(f"fused optimizer resume shape or master precision differs: {name}")
        optimizer.state[parameter] = {
            "step": torch.tensor(float(completed_steps), device=parameter.device, dtype=torch.float32),
            "exp_avg": state.exponential_average.to(device=parameter.device).clone(),
            "exp_avg_sq": state.exponential_average_squared.to(device=parameter.device).clone(),
        }
