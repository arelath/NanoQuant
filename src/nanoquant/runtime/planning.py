"""Ahead-of-inference backend planning with explicit fallback reporting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

import torch

from nanoquant.runtime.backend import (
    DeviceLike,
    PreparedLayer,
    QuantizedLinearSpec,
    RuntimeBackend,
    RuntimeLayerState,
    SupportResult,
    WorkloadSpec,
)
from nanoquant.runtime.logical import canonical_torch_dtype


@dataclass(frozen=True, slots=True)
class BackendRejection:
    backend_name: str
    backend_version: str
    support: SupportResult


@dataclass(frozen=True, slots=True)
class LayerDispatch:
    layer_name: str
    backend_name: str
    backend_version: str
    fallback: bool
    rejected_backends: tuple[BackendRejection, ...]


@dataclass(frozen=True, slots=True)
class BackendPlan:
    workload: WorkloadSpec
    layers: tuple[LayerDispatch, ...]
    fallback_count: int


@dataclass(frozen=True, slots=True)
class ExecutionPlans:
    """Independently selected prefill and decode plans for one layer inventory."""

    prefill: BackendPlan
    decode: BackendPlan

    def __post_init__(self) -> None:
        if self.prefill.workload.kind != "prefill" or self.decode.workload.kind != "decode":
            raise ValueError("runtime execution plans require prefill and decode workloads")
        if self.decode.workload.token_count != 1:
            raise ValueError("runtime decode plans require exactly one token per batch item")
        if self.prefill.workload.device_type != self.decode.workload.device_type:
            raise ValueError("runtime prefill and decode plans must use the same device type")
        prefill_names = tuple(item.layer_name for item in self.prefill.layers)
        decode_names = tuple(item.layer_name for item in self.decode.layers)
        if prefill_names != decode_names:
            raise ValueError("runtime prefill and decode plan layer inventories differ")


class BackendPlanningError(RuntimeError):
    pass


def plan_backends(
    layers: tuple[QuantizedLinearSpec, ...],
    workload: WorkloadSpec,
    backends: tuple[RuntimeBackend, ...],
    *,
    strict: bool,
) -> BackendPlan:
    if not layers:
        raise ValueError("runtime backend planning requires at least one layer")
    if len({layer.name for layer in layers}) != len(layers):
        raise ValueError("runtime backend planning layer names must be unique")
    if not backends:
        raise ValueError("runtime backend planning requires at least one backend")
    backend_keys = [(backend.name, backend.version) for backend in backends]
    if len(set(backend_keys)) != len(backend_keys):
        raise ValueError("runtime backend names and versions must be unique")
    dispatches = []
    for layer in layers:
        rejected = []
        selected: RuntimeBackend | None = None
        for backend in backends:
            support = backend.supports(layer, workload)
            if support.supported:
                selected = backend
                break
            rejected.append(BackendRejection(backend.name, backend.version, support))
        if selected is None:
            details = "; ".join(
                f"{item.backend_name}@{item.backend_version}: {item.support.code} {item.support.reason}"
                for item in rejected
            )
            raise BackendPlanningError(f"no runtime backend supports {layer.name}: {details}")
        fallback = bool(rejected)
        if strict and fallback:
            first = rejected[0]
            raise BackendPlanningError(
                f"strict runtime plan rejected fallback for {layer.name}: "
                f"{first.backend_name}@{first.backend_version} "
                f"{first.support.code} {first.support.reason}"
            )
        dispatches.append(
            LayerDispatch(layer.name, selected.name, selected.version, fallback, tuple(rejected))
        )
    result = tuple(dispatches)
    return BackendPlan(workload, result, sum(item.fallback for item in result))


def plan_execution_workloads(
    layers: tuple[QuantizedLinearSpec, ...],
    *,
    prefill: WorkloadSpec,
    decode: WorkloadSpec,
    prefill_backends: tuple[RuntimeBackend, ...],
    decode_backends: tuple[RuntimeBackend, ...],
    strict: bool,
) -> ExecutionPlans:
    """Resolve prefill and decode independently while preserving one layer order."""

    if prefill.kind != "prefill":
        raise ValueError("runtime prefill plan requires a prefill workload")
    if decode.kind != "decode":
        raise ValueError("runtime decode plan requires a decode workload")
    return ExecutionPlans(
        plan_backends(layers, prefill, prefill_backends, strict=strict),
        plan_backends(layers, decode, decode_backends, strict=strict),
    )


@dataclass(frozen=True, slots=True)
class PreparedDispatch:
    plan: LayerDispatch
    backend: RuntimeBackend
    layer: PreparedLayer

    def linear(self, value: torch.Tensor) -> torch.Tensor:
        return self.backend.linear(value, self.layer)


@dataclass(frozen=True, slots=True)
class PreparedBackendPlan:
    plan: BackendPlan
    dispatches: tuple[PreparedDispatch, ...]

    def __post_init__(self) -> None:
        if len(self.plan.layers) != len(self.dispatches):
            raise ValueError("prepared runtime dispatch count differs from its plan")
        for planned, prepared in zip(self.plan.layers, self.dispatches, strict=True):
            if prepared.plan != planned:
                raise ValueError("prepared runtime dispatch order differs from its plan")

    def linear_at(self, layer_index: int, value: torch.Tensor) -> torch.Tensor:
        """Execute one planned layer after enforcing this workload's token geometry."""

        if layer_index < 0 or layer_index >= len(self.dispatches):
            raise IndexError(f"runtime plan layer index is outside the inventory: {layer_index}")
        dispatch = self.dispatches[layer_index]
        workload = self.plan.workload
        if value.device.type != workload.device_type:
            raise ValueError(
                f"runtime {workload.kind} input device {value.device.type!r} differs from "
                f"planned device {workload.device_type!r}"
            )
        if canonical_torch_dtype(value.dtype) != workload.input_dtype:
            raise ValueError(
                f"runtime {workload.kind} input dtype differs from planned dtype "
                f"{workload.input_dtype!r}"
            )
        expected_slots = workload.batch_size * workload.token_count
        if value.ndim == 0 or value.shape[-1] != dispatch.layer.spec.in_features:
            raise ValueError(
                f"runtime {workload.kind} input feature dimension differs for "
                f"{dispatch.plan.layer_name}"
            )
        actual_slots = prod(value.shape[:-1])
        if actual_slots != expected_slots:
            raise ValueError(
                f"runtime {workload.kind} input has {actual_slots} token slots, "
                f"expected {expected_slots}"
            )
        return dispatch.linear(value)


@dataclass(frozen=True, slots=True)
class PreparedExecutionPlans:
    prefill: PreparedBackendPlan
    decode: PreparedBackendPlan

    def __post_init__(self) -> None:
        ExecutionPlans(self.prefill.plan, self.decode.plan)


def _validate_state_inventory(
    plan: BackendPlan,
    states: Mapping[str, RuntimeLayerState],
) -> None:
    expected_names = {item.layer_name for item in plan.layers}
    if set(states) != expected_names:
        missing = sorted(expected_names - set(states))
        unexpected = sorted(set(states) - expected_names)
        raise ValueError(
            f"runtime layer state inventory differs from the plan: missing={missing}, unexpected={unexpected}"
        )


def _backend_registry(
    backends: tuple[RuntimeBackend, ...],
) -> dict[tuple[str, str], RuntimeBackend]:
    registry = {(backend.name, backend.version): backend for backend in backends}
    if len(registry) != len(backends):
        raise ValueError("runtime backend names and versions must be unique")
    return registry


def _prepare_dispatches(
    plan: BackendPlan,
    states: Mapping[str, RuntimeLayerState],
    registry: Mapping[tuple[str, str], RuntimeBackend],
    device: DeviceLike,
    cache: dict[tuple[str, str, str], PreparedLayer],
) -> tuple[PreparedDispatch, ...]:
    prepared = []
    for item in plan.layers:
        backend_key = (item.backend_name, item.backend_version)
        if backend_key not in registry:
            raise ValueError(
                f"planned runtime backend is unavailable: {item.backend_name}@{item.backend_version}"
            )
        state = states[item.layer_name]
        if state.spec.name != item.layer_name:
            raise ValueError(f"runtime layer state name differs from its plan entry: {item.layer_name}")
        backend = registry[backend_key]
        cache_key = (item.layer_name, *backend_key)
        layer = cache.get(cache_key)
        if layer is None:
            layer = backend.prepare(state, device)
            if layer.spec != state.spec or (layer.backend_name, layer.backend_version) != backend_key:
                raise ValueError(f"runtime backend prepared an inconsistent layer: {item.layer_name}")
            cache[cache_key] = layer
        prepared.append(PreparedDispatch(item, backend, layer))
    return tuple(prepared)


def prepare_plan(
    plan: BackendPlan,
    states: Mapping[str, RuntimeLayerState],
    backends: tuple[RuntimeBackend, ...],
    device: DeviceLike,
) -> tuple[PreparedDispatch, ...]:
    _validate_state_inventory(plan, states)
    return _prepare_dispatches(plan, states, _backend_registry(backends), device, {})


def prepare_execution_workloads(
    plans: ExecutionPlans,
    states: Mapping[str, RuntimeLayerState],
    backends: tuple[RuntimeBackend, ...],
    device: DeviceLike,
) -> PreparedExecutionPlans:
    """Prepare both workload plans, sharing identical layer/backend payloads."""

    target = torch.device(device)
    if target.type != plans.prefill.workload.device_type:
        raise ValueError(
            f"runtime preparation device {target.type!r} differs from planned device "
            f"{plans.prefill.workload.device_type!r}"
        )
    _validate_state_inventory(plans.prefill, states)
    _validate_state_inventory(plans.decode, states)
    registry = _backend_registry(backends)
    cache: dict[tuple[str, str, str], PreparedLayer] = {}
    prefill = PreparedBackendPlan(
        plans.prefill,
        _prepare_dispatches(plans.prefill, states, registry, device, cache),
    )
    decode = PreparedBackendPlan(
        plans.decode,
        _prepare_dispatches(plans.decode, states, registry, device, cache),
    )
    return PreparedExecutionPlans(prefill, decode)
