"""Typed block calibration with scoped hooks and deterministic pass ordering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

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


@dataclass(frozen=True, slots=True)
class OnlineAccumulatorSnapshot:
    total: torch.Tensor
    global_max: torch.Tensor | None
    batch_count: int
    pre_scale: float
    post_scale: float
    percentile: float


@dataclass(frozen=True, slots=True)
class CausalOnlineLayerSnapshot:
    path: str
    inputs: OnlineAccumulatorSnapshot
    outputs: OnlineAccumulatorSnapshot


@dataclass(frozen=True, slots=True)
class CausalOnlineCalibrationState:
    layers: tuple[CausalOnlineLayerSnapshot, ...]
    processed_samples: int

    @property
    def sample_count(self) -> int:
        if self.processed_samples < 0:
            raise ValueError("causal calibration processed-sample count is negative")
        return self.processed_samples


BatchRunner = Callable[[nn.Module, torch.Tensor], torch.Tensor]
LossBuilder = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Accumulator(Protocol):
    def update(self, tensor: torch.Tensor) -> None: ...
    def finalize(self) -> torch.Tensor: ...


class UnsupportedCalibrationMode(ValueError):
    code = "CAL004"


def causal_language_model_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Return the exact next-token cross entropy used by legacy Fisher calibration."""
    if logits.ndim != 3 or token_ids.ndim != 2:
        raise ValueError("causal calibration expects rank-3 logits and rank-2 token ids")
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("causal calibration logits and token ids must share batch/sequence dimensions")
    if token_ids.shape[1] < 2:
        raise ValueError("causal calibration requires at least two tokens per sequence")
    return torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        token_ids[:, 1:].reshape(-1),
    )


def calibrate_causal_model(
    model: nn.Module,
    batches: tuple[torch.Tensor, ...],
    layers: tuple[tuple[str, nn.Linear], ...],
    *,
    method: str = "online_fisher",
    shrinkage: float = 0.0,
    initial_state: CausalOnlineCalibrationState | None = None,
    state_sink: Callable[[CausalOnlineCalibrationState], None] | None = None,
) -> tuple[MaterializedLayerCalibration, ...]:
    """Collect legacy-compatible full-model input and output Fisher diagonals.

    Each batch contributes one accumulator update, so callers seeking exact
    Experiment 018/019 behavior must supply one sequence per batch.
    """
    if not batches:
        raise ValueError("calibration requires at least one batch")
    if not layers:
        raise ValueError("causal calibration requires at least one linear layer")
    if method not in {"online_fisher", "two_phase_fisher"}:
        raise UnsupportedCalibrationMode(f"CAL004 unsupported causal calibration method: {method}")
    if len({path for path, _module in layers}) != len(layers):
        raise ValueError("causal calibration layer paths must be unique")
    if method != "online_fisher" and (initial_state is not None or state_sink is not None):
        raise ValueError("durable causal calibration state is supported only for online Fisher")
    state_by_path = {} if initial_state is None else {layer.path: layer for layer in initial_state.layers}
    if initial_state is not None and set(state_by_path) != {path for path, _module in layers}:
        raise ValueError("causal calibration state does not exactly match requested layers")

    original_training = model.training
    original_requires_grad = tuple(parameter.requires_grad for parameter in model.parameters())
    original_gradient_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    config = getattr(model, "config", None)
    original_use_cache = getattr(config, "use_cache", None)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    enable_inputs = getattr(model, "enable_input_require_grads", None)
    disable_inputs = getattr(model, "disable_input_require_grads", None)
    if callable(enable_inputs):
        enable_inputs()
    enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
    if callable(enable_checkpointing) and not original_gradient_checkpointing:
        enable_checkpointing()
    if config is not None and original_use_cache is not None:
        config.use_cache = False
    model.train()

    def run_batch(batch: torch.Tensor) -> torch.Tensor:
        output = cast(Any, model)(input_ids=batch, use_cache=False)
        logits = getattr(output, "logits", output)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("causal calibration model did not return tensor logits")
        return logits

    def backward_batch(batch: torch.Tensor) -> None:
        text_model = getattr(model, "model", None)
        lm_head = getattr(model, "lm_head", None)
        if isinstance(text_model, nn.Module) and isinstance(lm_head, nn.Module):
            text_output = cast(Any, text_model)(input_ids=batch, use_cache=False)
            hidden = getattr(text_output, "last_hidden_state", None)
            if hidden is None and isinstance(text_output, (tuple, list)) and text_output:
                hidden = text_output[0]
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("causal calibration text model did not return hidden states")
            hidden_gradient = torch.zeros_like(hidden)
            target_count = batch.shape[0] * (batch.shape[1] - 1)
            token_chunk = 128
            for start in range(0, batch.shape[1] - 1, token_chunk):
                end = min(start + token_chunk, batch.shape[1] - 1)
                detached = hidden[:, start:end].detach().requires_grad_(True)
                logits = lm_head(detached)
                loss = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    batch[:, start + 1 : end + 1].reshape(-1),
                    reduction="sum",
                ) / target_count
                gradient = torch.autograd.grad(loss, detached)[0]
                hidden_gradient[:, start:end].copy_(gradient)
            torch.autograd.backward(hidden, hidden_gradient)
            return
        logits = run_batch(batch)
        torch.autograd.backward(causal_language_model_loss(logits, batch))

    def execute(
        input_accumulators: dict[str, Accumulator], output_accumulators: dict[str, Accumulator]
    ) -> None:
        handles: list[torch.utils.hooks.RemovableHandle] = []
        try:
            for path, module in layers:

                def forward_hook(
                    _module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: object, path: str = path
                ) -> None:
                    input_accumulators[path].update(inputs[0])

                def backward_hook(
                    _module: nn.Module,
                    _inputs: tuple[torch.Tensor | None, ...],
                    outputs: tuple[torch.Tensor | None, ...],
                    path: str = path,
                ) -> None:
                    if outputs[0] is not None:
                        output_accumulators[path].update(outputs[0])

                handles.append(module.register_forward_hook(forward_hook))
                handles.append(module.register_full_backward_hook(backward_hook))
            for batch in batches:
                model.zero_grad(set_to_none=True)
                backward_batch(batch)
        finally:
            for handle in handles:
                handle.remove()

    try:
        if method == "two_phase_fisher":
            input_thresholds = {path: torch.zeros(()) for path, _module in layers}
            output_thresholds = {path: torch.zeros(()) for path, _module in layers}
            handles: list[torch.utils.hooks.RemovableHandle] = []
            try:
                for path, module in layers:

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
                    model.zero_grad(set_to_none=True)
                    backward_batch(batch)
            finally:
                for handle in handles:
                    handle.remove()
            inputs: dict[str, Accumulator] = {
                path: FixedClippedAccumulator(module.in_features, input_thresholds[path]) for path, module in layers
            }
            outputs: dict[str, Accumulator] = {
                path: FixedClippedAccumulator(module.out_features, output_thresholds[path], 1e6, 1e-6)
                for path, module in layers
            }
        else:
            inputs = {
                path: _restore_online_accumulator(
                    module.in_features,
                    None if initial_state is None else state_by_path[path].inputs,
                )
                for path, module in layers
            }
            outputs = {
                path: _restore_online_accumulator(
                    module.out_features,
                    None if initial_state is None else state_by_path[path].outputs,
                    pre_scale=1e6,
                    post_scale=1e-6,
                )
                for path, module in layers
            }
        execute(inputs, outputs)
        logical_sample_count = (0 if initial_state is None else initial_state.sample_count) + sum(
            batch.shape[0] for batch in batches
        )
        if state_sink is not None:
            state_sink(
                CausalOnlineCalibrationState(
                    tuple(
                        CausalOnlineLayerSnapshot(
                            path,
                            _snapshot_online_accumulator(cast(OnlineClippedAccumulator, inputs[path])),
                            _snapshot_online_accumulator(cast(OnlineClippedAccumulator, outputs[path])),
                        )
                        for path, _module in layers
                    ),
                    logical_sample_count,
                )
            )
        return tuple(
            MaterializedLayerCalibration(
                path,
                shrink_importance(cast(Any, inputs[path]).total / logical_sample_count, shrinkage),
                shrink_importance(cast(Any, outputs[path]).total / logical_sample_count, shrinkage),
                logical_sample_count,
                method,
            )
            for path, _module in layers
        )
    finally:
        model.zero_grad(set_to_none=True)
        if callable(disable_inputs):
            disable_inputs()
        disable_checkpointing = getattr(model, "gradient_checkpointing_disable", None)
        if callable(disable_checkpointing) and not original_gradient_checkpointing:
            disable_checkpointing()
        if config is not None and original_use_cache is not None:
            config.use_cache = original_use_cache
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad, strict=True):
            parameter.requires_grad_(requires_grad)
        model.train(original_training)
        if any(batch.is_cuda for batch in batches):
            torch.cuda.empty_cache()


def _restore_online_accumulator(
    width: int,
    snapshot: OnlineAccumulatorSnapshot | None,
    *,
    pre_scale: float = 1.0,
    post_scale: float = 1.0,
) -> OnlineClippedAccumulator:
    if snapshot is None:
        return OnlineClippedAccumulator(width, pre_scale, post_scale)
    if snapshot.total.shape != (width,):
        raise ValueError("online calibration accumulator width changed")
    if snapshot.pre_scale != pre_scale or snapshot.post_scale != post_scale:
        raise ValueError("online calibration accumulator scaling changed")
    accumulator = OnlineClippedAccumulator(width, snapshot.pre_scale, snapshot.post_scale, snapshot.percentile)
    accumulator.total.copy_(snapshot.total)
    accumulator.global_max = None if snapshot.global_max is None else snapshot.global_max.detach().clone()
    accumulator.batch_count = snapshot.batch_count
    return accumulator


def _snapshot_online_accumulator(accumulator: OnlineClippedAccumulator) -> OnlineAccumulatorSnapshot:
    return OnlineAccumulatorSnapshot(
        accumulator.total.detach().clone(),
        None if accumulator.global_max is None else accumulator.global_max.detach().clone(),
        accumulator.batch_count,
        accumulator.pre_scale,
        accumulator.post_scale,
        accumulator.percentile,
    )


def materialize_causal_online_state(
    state: CausalOnlineCalibrationState,
    *,
    shrinkage: float = 0.0,
) -> tuple[MaterializedLayerCalibration, ...]:
    sample_count = state.sample_count
    if sample_count <= 0:
        raise ValueError("cannot materialize an empty causal calibration state")
    return tuple(
        MaterializedLayerCalibration(
            layer.path,
            shrink_importance(layer.inputs.total / sample_count, shrinkage),
            shrink_importance(layer.outputs.total / sample_count, shrinkage),
            sample_count,
            "online_fisher",
        )
        for layer in state.layers
    )


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
