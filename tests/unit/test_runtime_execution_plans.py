from __future__ import annotations

import pytest
import torch

from nanoquant.runtime import (
    BackendCapabilities,
    ExecutionPlans,
    LogicalLayerState,
    PreparedLayer,
    QuantizedLinearSpec,
    SupportResult,
    WorkloadSpec,
    plan_execution_workloads,
    prepare_execution_workloads,
)
from nanoquant.runtime.backend import RuntimeLayerState


def _state(name: str) -> LogicalLayerState:
    spec = QuantizedLinearSpec(name, "nanoquant-v1", 4, 3, 2, "float32", "float32")
    return LogicalLayerState(
        spec,
        torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]),
        torch.ones(4),
        torch.ones(2),
        torch.ones(3),
    )


class _KindBackend:
    version = "1"

    def __init__(self, name: str, kinds: tuple[str, ...]) -> None:
        self.name = name
        self.kinds = kinds
        self.prepare_count = 0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            logical_formats=("nanoquant-v1",),
            device_types=("cpu",),
            input_dtypes=("float32",),
            factor_dtypes=("float32",),
            scale_dtypes=("float32",),
            outlier_value_dtypes=(),
            workload_kinds=("prefill", "decode"),
            supports_bias=False,
            supports_outliers=False,
            supports_deterministic=True,
        )

    def supports(self, op: QuantizedLinearSpec, workload: WorkloadSpec) -> SupportResult:
        del op
        if workload.kind not in self.kinds:
            return SupportResult.rejected("NQ-INF-WORKLOAD", f"{workload.kind} is not selected")
        return SupportResult.accepted()

    def prepare(self, state: RuntimeLayerState, device: str | torch.device) -> PreparedLayer:
        assert torch.device(device).type == "cpu"
        self.prepare_count += 1
        return PreparedLayer(self.name, self.version, state.spec, state)

    def linear(self, value: torch.Tensor, layer: PreparedLayer) -> torch.Tensor:
        return torch.zeros(*value.shape[:-1], layer.spec.out_features, dtype=value.dtype)


def _workloads() -> tuple[WorkloadSpec, WorkloadSpec]:
    return (
        WorkloadSpec("prefill", "cpu", "float32", 2, 3, deterministic=True),
        WorkloadSpec("decode", "cpu", "float32", 2, 1, deterministic=True),
    )


def test_execution_plans_select_prefill_and_decode_independently() -> None:
    states = {name: _state(name) for name in ("blocks.0.proj", "blocks.1.proj")}
    prefill_workload, decode_workload = _workloads()
    prefill_backend = _KindBackend("prefill-only", ("prefill",))
    decode_backend = _KindBackend("decode-only", ("decode",))

    plans = plan_execution_workloads(
        tuple(state.spec for state in states.values()),
        prefill=prefill_workload,
        decode=decode_workload,
        prefill_backends=(prefill_backend,),
        decode_backends=(decode_backend,),
        strict=True,
    )
    prepared = prepare_execution_workloads(
        plans,
        states,
        (prefill_backend, decode_backend),
        "cpu",
    )

    assert {item.backend_name for item in plans.prefill.layers} == {"prefill-only"}
    assert {item.backend_name for item in plans.decode.layers} == {"decode-only"}
    assert plans.prefill.fallback_count == 0
    assert plans.decode.fallback_count == 0
    assert prefill_backend.prepare_count == 2
    assert decode_backend.prepare_count == 2
    assert prepared.prefill.linear_at(0, torch.ones(2, 3, 4)).shape == (2, 3, 3)
    assert prepared.decode.linear_at(0, torch.ones(2, 4)).shape == (2, 3)


def test_execution_plan_preparation_reuses_identical_backend_payloads() -> None:
    states = {name: _state(name) for name in ("blocks.0.proj", "blocks.1.proj")}
    prefill, decode = _workloads()
    backend = _KindBackend("both", ("prefill", "decode"))
    plans = plan_execution_workloads(
        tuple(state.spec for state in states.values()),
        prefill=prefill,
        decode=decode,
        prefill_backends=(backend,),
        decode_backends=(backend,),
        strict=True,
    )

    prepared = prepare_execution_workloads(plans, states, (backend,), "cpu")

    assert backend.prepare_count == 2
    for prefill_dispatch, decode_dispatch in zip(
        prepared.prefill.dispatches,
        prepared.decode.dispatches,
        strict=True,
    ):
        assert prefill_dispatch.layer is decode_dispatch.layer


def test_prepared_workload_plan_rejects_wrong_token_geometry() -> None:
    state = _state("blocks.0.proj")
    prefill, decode = _workloads()
    backend = _KindBackend("both", ("prefill", "decode"))
    plans = plan_execution_workloads(
        (state.spec,),
        prefill=prefill,
        decode=decode,
        prefill_backends=(backend,),
        decode_backends=(backend,),
        strict=True,
    )
    prepared = prepare_execution_workloads(plans, {state.spec.name: state}, (backend,), "cpu")

    with pytest.raises(ValueError, match="prefill input has 2 token slots, expected 6"):
        prepared.prefill.linear_at(0, torch.ones(2, 4))
    with pytest.raises(ValueError, match="decode input has 6 token slots, expected 2"):
        prepared.decode.linear_at(0, torch.ones(2, 3, 4))
    with pytest.raises(ValueError, match="prefill input dtype differs"):
        prepared.prefill.linear_at(0, torch.ones(2, 3, 4, dtype=torch.float16))
    with pytest.raises(IndexError, match="outside the inventory"):
        prepared.decode.linear_at(1, torch.ones(2, 4))


def test_execution_plans_reject_invalid_pairing_and_device() -> None:
    state = _state("blocks.0.proj")
    prefill, decode = _workloads()
    backend = _KindBackend("both", ("prefill", "decode"))

    with pytest.raises(ValueError, match="decode plans require exactly one token"):
        plan_execution_workloads(
            (state.spec,),
            prefill=prefill,
            decode=WorkloadSpec("decode", "cpu", "float32", 1, 2),
            prefill_backends=(backend,),
            decode_backends=(backend,),
            strict=True,
        )
    plans = plan_execution_workloads(
        (state.spec,),
        prefill=prefill,
        decode=decode,
        prefill_backends=(backend,),
        decode_backends=(backend,),
        strict=True,
    )
    with pytest.raises(ValueError, match="preparation device 'cuda' differs"):
        prepare_execution_workloads(plans, {state.spec.name: state}, (backend,), "cuda")

    with pytest.raises(ValueError, match="layer inventories differ"):
        ExecutionPlans(
            plans.prefill,
            plan_execution_workloads(
                (_state("blocks.9.proj").spec,),
                prefill=prefill,
                decode=decode,
                prefill_backends=(backend,),
                decode_backends=(backend,),
                strict=True,
            ).decode,
        )
