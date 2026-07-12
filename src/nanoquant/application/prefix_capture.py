"""Scoped model-prefix input capture without replacement or control-flow exceptions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CapturedBlockInvocation:
    positional: tuple[object, ...]
    keyword: dict[str, object]


def _detach(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _detach(item) for key, item in value.items()}
    return value


def capture_prefix_invocations(
    first_block: nn.Module, model_calls: tuple[Callable[[], object], ...]
) -> tuple[CapturedBlockInvocation, ...]:
    """Run normal model calls and capture the exact first-block args/metadata."""
    captured: list[CapturedBlockInvocation] = []

    def hook(_module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        captured.append(
            CapturedBlockInvocation(
                tuple(_detach(value) for value in args),
                {key: _detach(value) for key, value in kwargs.items()},
            )
        )

    handle = first_block.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for call in model_calls:
            call()
    finally:
        handle.remove()
    if len(captured) != len(model_calls):
        raise RuntimeError(f"prefix capture expected {len(model_calls)} block calls but observed {len(captured)}")
    return tuple(captured)
