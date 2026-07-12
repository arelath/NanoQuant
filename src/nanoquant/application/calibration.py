"""Typed block calibration with scoped hooks and deterministic pass ordering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from nanoquant.domain.calibration_math import (
    FixedClippedAccumulator,
    OnlineClippedAccumulator,
    robust_tau,
    shrink_importance,
)
from nanoquant.ports.activation_store import ActivationStore


@dataclass(frozen=True, slots=True)
class MaterializedLayerCalibration:
    path: str
    input_importance: torch.Tensor
    output_importance: torch.Tensor
    sample_count: int
    method: str


BatchRunner = Callable[[nn.Module, torch.Tensor], torch.Tensor]
LossBuilder = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Accumulator(Protocol):
    def update(self, tensor: torch.Tensor) -> None: ...
    def finalize(self) -> torch.Tensor: ...


class UnsupportedCalibrationMode(ValueError):
    code = "CAL004"


def _linears(block: nn.Module, paths: tuple[str, ...]) -> dict[str, nn.Linear]:
    modules = dict(block.named_modules())
    result = {}
    for path in paths:
        module = modules.get(path)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"calibration target is not a linear layer: {path}")
        result[path] = module
    return result


def calibrate_block(
    block: nn.Module,
    batches: tuple[torch.Tensor, ...],
    layer_paths: tuple[str, ...],
    runner: BatchRunner,
    *,
    method: str = "online_fisher",
    shrinkage: float = 0.0,
    loss_builder: LossBuilder | None = None,
) -> tuple[MaterializedLayerCalibration, ...]:
    if not batches:
        raise ValueError("calibration requires at least one batch")
    if method not in {"online_fisher", "two_phase_fisher", "forward_only"}:
        raise UnsupportedCalibrationMode(f"CAL004 unsupported calibration method: {method}")
    linears = _linears(block, layer_paths)
    requires_backward = method != "forward_only"
    objective = loss_builder or (lambda output, _batch: output.float().square().mean())

    def execute(input_accumulators: dict[str, Accumulator], output_accumulators: dict[str, Accumulator]) -> None:
        handles: list[torch.utils.hooks.RemovableHandle] = []
        try:
            for path, module in linears.items():

                def forward_hook(
                    _module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: object, path: str = path
                ) -> None:
                    input_accumulators[path].update(inputs[0])

                handles.append(module.register_forward_hook(forward_hook))
                if requires_backward:

                    def backward_hook(
                        _module: nn.Module,
                        _inputs: tuple[torch.Tensor | None, ...],
                        outputs: tuple[torch.Tensor | None, ...],
                        path: str = path,
                    ) -> None:
                        if outputs[0] is not None:
                            output_accumulators[path].update(outputs[0])

                    handles.append(module.register_full_backward_hook(backward_hook))
            for batch in batches:
                block.zero_grad(set_to_none=True)
                batch_input = batch.detach().requires_grad_(requires_backward)
                output = runner(block, batch_input)
                if requires_backward:
                    torch.autograd.backward(objective(output, batch))
        finally:
            for handle in handles:
                handle.remove()

    if method == "two_phase_fisher":
        input_thresholds = {path: torch.zeros(()) for path in linears}
        output_thresholds = {path: torch.zeros(()) for path in linears}
        handles = []
        try:
            for path, module in linears.items():

                def profile_forward(
                    _module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: object, path: str = path
                ) -> None:
                    input_thresholds[path] = torch.maximum(input_thresholds[path], robust_tau(inputs[0]).cpu())

                def profile_backward(
                    _module: nn.Module,
                    _inputs: tuple[torch.Tensor | None, ...],
                    outputs: tuple[torch.Tensor | None, ...],
                    path: str = path,
                ) -> None:
                    if outputs[0] is not None:
                        output_thresholds[path] = torch.maximum(
                            output_thresholds[path], robust_tau(outputs[0], pre_scale=1e6).cpu()
                        )

                handles.append(module.register_forward_hook(profile_forward))
                handles.append(module.register_full_backward_hook(profile_backward))
            for batch in batches:
                block.zero_grad(set_to_none=True)
                output = runner(block, batch.detach().requires_grad_(True))
                torch.autograd.backward(objective(output, batch))
        finally:
            for handle in handles:
                handle.remove()
        inputs: dict[str, Accumulator] = {
            path: FixedClippedAccumulator(module.in_features, input_thresholds[path])
            for path, module in linears.items()
        }
        outputs: dict[str, Accumulator] = {
            path: FixedClippedAccumulator(module.out_features, output_thresholds[path], 1e6, 1e-6)
            for path, module in linears.items()
        }
    else:
        inputs = {path: OnlineClippedAccumulator(module.in_features) for path, module in linears.items()}
        outputs = (
            {path: OnlineClippedAccumulator(module.out_features, 1e6, 1e-6) for path, module in linears.items()}
            if requires_backward
            else {}
        )
    execute(inputs, outputs)
    return tuple(
        MaterializedLayerCalibration(
            path,
            shrink_importance(inputs[path].finalize(), shrinkage),
            shrink_importance(outputs[path].finalize(), shrinkage)
            if requires_backward
            else torch.ones(module.out_features),
            sum(batch.shape[0] for batch in batches),
            method,
        )
        for path, module in linears.items()
    )


def calibrate_block_streamed(
    block: nn.Module,
    activations: ActivationStore,
    key: str,
    layer_paths: tuple[str, ...],
    runner: BatchRunner,
    *,
    batch_size: int,
    device: str = "cpu",
    shrinkage: float = 0.0,
) -> tuple[MaterializedLayerCalibration, ...]:
    """Accumulate forward-only statistics from batch views over an activation generation."""
    if batch_size <= 0:
        raise ValueError("streamed calibration batch size must be positive")
    with activations.read(key, device) as values:
        if values.ndim == 0 or values.shape[0] == 0:
            raise ValueError("streamed calibration activation generation is empty")
        batches = tuple(values[start : start + batch_size] for start in range(0, values.shape[0], batch_size))
        return calibrate_block(
            block,
            batches,
            layer_paths,
            runner,
            method="forward_only",
            shrinkage=shrinkage,
        )
