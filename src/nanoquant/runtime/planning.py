"""Ahead-of-inference backend planning with explicit fallback reporting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class PreparedDispatch:
    plan: LayerDispatch
    backend: RuntimeBackend
    layer: PreparedLayer

    def linear(self, value: torch.Tensor) -> torch.Tensor:
        return self.backend.linear(value, self.layer)


def prepare_plan(
    plan: BackendPlan,
    states: Mapping[str, RuntimeLayerState],
    backends: tuple[RuntimeBackend, ...],
    device: DeviceLike,
) -> tuple[PreparedDispatch, ...]:
    expected_names = {item.layer_name for item in plan.layers}
    if set(states) != expected_names:
        missing = sorted(expected_names - set(states))
        unexpected = sorted(set(states) - expected_names)
        raise ValueError(
            f"runtime layer state inventory differs from the plan: missing={missing}, unexpected={unexpected}"
        )
    registry = {(backend.name, backend.version): backend for backend in backends}
    prepared = []
    for item in plan.layers:
        key = (item.backend_name, item.backend_version)
        if key not in registry:
            raise ValueError(f"planned runtime backend is unavailable: {item.backend_name}@{item.backend_version}")
        state = states[item.layer_name]
        if state.spec.name != item.layer_name:
            raise ValueError(f"runtime layer state name differs from its plan entry: {item.layer_name}")
        backend = registry[key]
        layer = backend.prepare(state, device)
        if layer.spec != state.spec or (layer.backend_name, layer.backend_version) != key:
            raise ValueError(f"runtime backend prepared an inconsistent layer: {item.layer_name}")
        prepared.append(PreparedDispatch(item, backend, layer))
    return tuple(prepared)
