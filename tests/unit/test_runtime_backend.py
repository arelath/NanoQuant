from __future__ import annotations

import importlib.util

import pytest
import torch

from nanoquant.runtime import (
    CUDA_PACKED_REFERENCE_SHA256,
    BackendCapabilities,
    BackendPlanningError,
    CudaPackedBackend,
    DenseReferenceBackend,
    FactorizedReferenceBackend,
    LogicalLayerState,
    PreparedLayer,
    QuantizedLinearSpec,
    SupportResult,
    WorkloadSpec,
    plan_backends,
    prepare_plan,
)
from nanoquant.runtime.backend import RuntimeLayerState, evaluate_capabilities


def _spec(name: str = "blocks.0.mlp.proj", *, outliers: bool = True) -> QuantizedLinearSpec:
    return QuantizedLinearSpec(
        name,
        "nanoquant-v1",
        4,
        3,
        2,
        "float32",
        "float32",
        outlier_count=1 if outliers else 0,
        outlier_value_dtype="int8" if outliers else None,
        has_outlier_scales=outliers,
        has_bias=True,
    )


def _state(name: str = "blocks.0.mlp.proj", *, outliers: bool = True) -> LogicalLayerState:
    return LogicalLayerState(
        _spec(name, outliers=outliers),
        torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]),
        torch.tensor([0.75, 1.0, 1.25, 1.5]),
        torch.tensor([0.5, 1.5]),
        torch.tensor([1.25, 0.75, 1.5]),
        bias=torch.tensor([0.1, -0.2, 0.3]),
        outlier_indices=torch.tensor([1]) if outliers else None,
        outlier_values=torch.tensor([[2], [-1], [3]], dtype=torch.int8) if outliers else None,
        outlier_scales=torch.tensor([[0.25], [0.5], [0.125]]) if outliers else None,
    )


def _workload(*, device: str = "cpu", dtype: str = "float32") -> WorkloadSpec:
    return WorkloadSpec("decode", device, dtype, 1, 1, deterministic=True)


def test_deployment_reference_backends_match_with_bias_and_quantized_outliers() -> None:
    state = _state()
    value = torch.tensor([[[0.25, -0.5, 1.0, 0.75], [-1.0, 0.5, 0.125, -0.25]]])
    dense = DenseReferenceBackend()
    factorized = FactorizedReferenceBackend()

    dense_output = dense.linear(value, dense.prepare(state, "cpu"))
    factorized_output = factorized.linear(value, factorized.prepare(state, "cpu"))

    torch.testing.assert_close(factorized_output, dense_output, rtol=1e-6, atol=1e-6)


def test_logical_layer_state_rejects_noncanonical_binary_and_outlier_inventory() -> None:
    state = _state()

    with pytest.raises(ValueError, match=r"other than -1 or \+1"):
        LogicalLayerState(
            state.spec,
            state.left_binary.clone().index_put_((torch.tensor([0]), torch.tensor([0])), torch.tensor([0.0])),
            state.right_binary,
            state.scale_pre,
            state.scale_mid,
            state.scale_post,
            state.bias,
            state.outlier_indices,
            state.outlier_values,
            state.outlier_scales,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        duplicated = QuantizedLinearSpec(
            "blocks.0.mlp.duplicate",
            "nanoquant-v1",
            4,
            3,
            2,
            "float32",
            "float32",
            outlier_count=2,
            outlier_value_dtype="int8",
            has_outlier_scales=True,
            has_bias=True,
        )
        LogicalLayerState(
            duplicated,
            state.left_binary,
            state.right_binary,
            state.scale_pre,
            state.scale_mid,
            state.scale_post,
            state.bias,
            torch.tensor([1, 1]),
            torch.ones(3, 2, dtype=torch.int8),
            torch.ones(3),
        )


def test_logical_layer_state_rejects_nonfinite_runtime_values() -> None:
    state = _state(outliers=False)

    with pytest.raises(ValueError, match="scale contains a non-finite"):
        LogicalLayerState(
            state.spec,
            state.left_binary,
            state.right_binary,
            state.scale_pre,
            state.scale_mid,
            torch.tensor([1.25, float("nan"), 1.5]),
            state.bias,
        )


def test_capability_matching_returns_stable_alignment_reason() -> None:
    capabilities = BackendCapabilities(
        logical_formats=("nanoquant-v1",),
        device_types=("cuda",),
        input_dtypes=("bfloat16",),
        factor_dtypes=("float32",),
        scale_dtypes=("float32",),
        outlier_value_dtypes=("int8",),
        workload_kinds=("decode",),
        supports_bias=True,
        supports_outliers=True,
        supports_deterministic=True,
        in_feature_alignment=8,
    )
    op = QuantizedLinearSpec(
        "blocks.0.attn.q_proj",
        "nanoquant-v1",
        12,
        8,
        8,
        "float32",
        "float32",
    )

    result = evaluate_capabilities(
        capabilities,
        op,
        WorkloadSpec("decode", "cuda", "bfloat16", 1, 1, deterministic=True),
    )

    assert result == SupportResult.rejected("NQ-INF-IN-ALIGNMENT", "input features must be aligned to 8")


def test_reference_backend_rejects_cuda_when_runtime_has_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = FactorizedReferenceBackend().supports(_spec(), _workload(device="cuda"))

    assert result == SupportResult.rejected(
        "NQ-INF-DEVICE-UNAVAILABLE",
        "CUDA is not available in this runtime",
    )


def test_cuda_packed_backend_declares_pinned_layout_and_capabilities() -> None:
    backend = CudaPackedBackend()
    capabilities = backend.capabilities()

    assert backend.reference_cuda_sha256 == CUDA_PACKED_REFERENCE_SHA256
    assert capabilities.device_types == ("cuda",)
    assert capabilities.workload_kinds == ("prefill", "decode")
    assert capabilities.supports_outliers
    assert capabilities.supports_bias
    assert capabilities.supports_deterministic


def test_cuda_packed_backend_reports_missing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CudaPackedBackend()
    workload = WorkloadSpec("decode", "cuda", "bfloat16", 1, 1, deterministic=True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    assert backend.supports(_spec(), workload).code == "NQ-INF-CUDA-KERNEL"

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert backend.supports(_spec(), workload).code == "NQ-INF-DEVICE-UNAVAILABLE"


class _RejectingBackend:
    name = "cuda-packed"
    version = "1"

    def capabilities(self) -> BackendCapabilities:
        return FactorizedReferenceBackend().capabilities()

    def supports(self, op: QuantizedLinearSpec, workload: WorkloadSpec) -> SupportResult:
        del op, workload
        return SupportResult.rejected("NQ-INF-RANK-ALIGNMENT", "rank must be aligned to 32")

    def prepare(self, state: RuntimeLayerState, device: str | torch.device) -> PreparedLayer:
        del state, device
        raise AssertionError("rejected backend must not prepare a layer")

    def linear(self, value: torch.Tensor, layer: PreparedLayer) -> torch.Tensor:
        del value, layer
        raise AssertionError("rejected backend must not execute a layer")


def test_backend_planner_reports_named_fallback_and_prepares_every_layer_once() -> None:
    states = {
        "blocks.0.mlp.proj": _state("blocks.0.mlp.proj"),
        "blocks.1.mlp.proj": _state("blocks.1.mlp.proj", outliers=False),
    }
    rejecting = _RejectingBackend()
    fallback = FactorizedReferenceBackend()
    backends = (rejecting, fallback)

    plan = plan_backends(
        tuple(state.spec for state in states.values()),
        _workload(),
        backends,
        strict=False,
    )
    prepared = prepare_plan(plan, states, backends, "cpu")

    assert plan.fallback_count == 2
    assert [item.backend_name for item in plan.layers] == [fallback.name, fallback.name]
    assert [item.rejected_backends[0].support.code for item in plan.layers] == [
        "NQ-INF-RANK-ALIGNMENT",
        "NQ-INF-RANK-ALIGNMENT",
    ]
    assert [item.plan.layer_name for item in prepared] == list(states)
    assert prepared[0].linear(torch.ones(1, 4)).shape == (1, 3)


def test_strict_backend_plan_fails_instead_of_silently_falling_back() -> None:
    rejecting = _RejectingBackend()

    with pytest.raises(
        BackendPlanningError,
        match="strict runtime plan rejected fallback.*NQ-INF-RANK-ALIGNMENT",
    ):
        plan_backends(
            (_spec(),),
            _workload(),
            (rejecting, FactorizedReferenceBackend()),
            strict=True,
        )


def test_backend_plan_rejects_missing_state_and_prepared_backend_mismatch() -> None:
    backend = FactorizedReferenceBackend()
    state = _state()
    plan = plan_backends((state.spec,), _workload(), (backend,), strict=True)

    with pytest.raises(ValueError, match="missing=.*blocks.0.mlp.proj"):
        prepare_plan(plan, {}, (backend,), "cpu")
    dense = DenseReferenceBackend()
    prepared = dense.prepare(state, "cpu")
    with pytest.raises(ValueError, match="different backend"):
        backend.linear(torch.ones(1, 4), prepared)
