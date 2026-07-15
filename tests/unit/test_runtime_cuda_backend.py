from __future__ import annotations

import importlib.util

import pytest
import torch

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.runtime import (
    CUDA_PACKED_REFERENCE_SHA256,
    CudaPackedBackend,
    LogicalLayerState,
    QuantizedLinearSpec,
    WorkloadSpec,
    pack_logical_layer,
)


@pytest.fixture(scope="module", autouse=True)
def _leased_cuda_runtime(tmp_path_factory: pytest.TempPathFactory):
    """Serialize CUDA tests and keep Triton's compiler cache inside the workspace sandbox."""

    if not torch.cuda.is_available():
        yield
        return
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path_factory.mktemp("triton-cache")))
    try:
        with acquire_device_lease("cuda:0"):
            yield
    finally:
        monkeypatch.undo()


def _logical(
    *,
    factor_dtype: torch.dtype = torch.float32,
    scale_dtype: torch.dtype = torch.bfloat16,
    outlier_dtype: torch.dtype | None = torch.bfloat16,
    outlier_count: int = 2,
    bias: bool = True,
) -> LogicalLayerState:
    outlier_count = outlier_count if outlier_dtype is not None else 0
    spec = QuantizedLinearSpec(
        "blocks.0.self_attn.q_proj",
        "nanoquant-v1",
        35,
        17,
        33,
        str(factor_dtype).removeprefix("torch."),
        str(scale_dtype).removeprefix("torch."),
        outlier_count=outlier_count,
        outlier_value_dtype=(str(outlier_dtype).removeprefix("torch.") if outlier_dtype is not None else None),
        has_outlier_scales=outlier_dtype is torch.int8,
        has_bias=bias,
    )
    generator = torch.Generator().manual_seed(19)
    left = torch.where(torch.rand(17, 33, generator=generator) > 0.5, 1.0, -1.0)
    right = torch.where(torch.rand(33, 35, generator=generator) > 0.5, 1.0, -1.0)
    scale_pre = torch.randn(35, generator=generator).to(scale_dtype)
    outlier_indices = torch.arange(outlier_count, dtype=torch.int64)
    if outlier_count:
        scale_pre[outlier_indices] = 0
    if outlier_dtype is None:
        outlier_values = None
    elif outlier_dtype is torch.int8:
        outlier_values = torch.randint(
            -8,
            8,
            (17, outlier_count),
            generator=generator,
            dtype=torch.int8,
        )
    else:
        outlier_values = torch.randn(17, outlier_count, generator=generator).to(outlier_dtype)
    return LogicalLayerState(
        spec,
        left.to(factor_dtype),
        right.to(factor_dtype),
        scale_pre,
        torch.randn(33, generator=generator).to(scale_dtype),
        torch.randn(17, generator=generator).to(scale_dtype),
        bias=torch.randn(17, generator=generator).to(scale_dtype) if bias else None,
        outlier_indices=outlier_indices if outlier_count else None,
        outlier_values=outlier_values,
        outlier_scales=(
            torch.linspace(0.125, 0.25, outlier_count).to(scale_dtype)
            if outlier_dtype is torch.int8
            else None
        ),
    )


def _expected(value: torch.Tensor, state: LogicalLayerState) -> torch.Tensor:
    value_float = value.float()
    latent = torch.nn.functional.linear(
        value_float * state.scale_pre.float(),
        state.right_binary.float(),
    )
    output = torch.nn.functional.linear(
        latent * state.scale_mid.float(),
        state.left_binary.float() * state.scale_post.float().reshape(-1, 1),
    )
    if state.outlier_indices is not None and state.outlier_values is not None:
        weights = state.outlier_values.float()
        if state.outlier_scales is not None:
            weights = weights * state.outlier_scales.float()
        output += torch.nn.functional.linear(
            value_float.index_select(-1, state.outlier_indices.long()),
            weights,
        )
    if state.bias is not None:
        output += state.bias.float()
    return output


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
    state = _logical()
    workload = WorkloadSpec("decode", "cuda", "bfloat16", 1, 1, deterministic=True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    assert backend.supports(state.spec, workload).code == "NQ-INF-CUDA-KERNEL"

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert backend.supports(state.spec, workload).code == "NQ-INF-DEVICE-UNAVAILABLE"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("token_shape", [(1,), (2, 3)])
@pytest.mark.parametrize(
    "outlier_dtype",
    [None, torch.float16, torch.bfloat16, torch.float32, torch.int8],
)
def test_cuda_packed_backend_matches_float32_operation(
    token_shape: tuple[int, ...],
    outlier_dtype: torch.dtype | None,
) -> None:
    logical = _logical(outlier_dtype=outlier_dtype)
    packed = pack_logical_layer(logical)
    backend = CudaPackedBackend()
    workload = WorkloadSpec(
        "decode" if token_shape == (1,) else "prefill",
        "cuda",
        "bfloat16",
        token_shape[0],
        1 if token_shape == (1,) else token_shape[-1],
        deterministic=True,
    )
    assert backend.supports(packed.spec, workload).supported
    prepared = backend.prepare(packed, "cuda")
    generator = torch.Generator().manual_seed(23)
    value = torch.randn(*token_shape, 35, generator=generator, dtype=torch.bfloat16).cuda()

    actual = backend.linear(value, prepared)
    expected = _expected(value.cpu(), logical)
    repeated = backend.linear(value, prepared)

    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-4)
    assert torch.equal(actual, repeated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_packed_backend_executes_every_declared_float_dtype(
    input_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> None:
    logical = _logical(scale_dtype=scale_dtype, outlier_dtype=scale_dtype, bias=False)
    packed = pack_logical_layer(logical)
    backend = CudaPackedBackend()
    prepared = backend.prepare(packed, "cuda")
    value = torch.linspace(-1, 1, 70, dtype=input_dtype).reshape(2, 35).cuda()

    actual = backend.linear(value, prepared)

    torch.testing.assert_close(actual.cpu(), _expected(value.cpu(), logical), rtol=2e-5, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("factor_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_packed_backend_accepts_every_declared_factor_dtype(
    factor_dtype: torch.dtype,
) -> None:
    logical = _logical(factor_dtype=factor_dtype, outlier_dtype=None, bias=False)
    backend = CudaPackedBackend()
    prepared = backend.prepare(pack_logical_layer(logical), "cuda")
    value = torch.linspace(-1, 1, 35, dtype=torch.bfloat16).reshape(1, 35).cuda()

    actual = backend.linear(value, prepared)

    torch.testing.assert_close(actual.cpu(), _expected(value.cpu(), logical), rtol=2e-5, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_packed_backend_streams_more_than_one_salient_tile() -> None:
    logical = _logical(outlier_dtype=torch.float32, outlier_count=33, bias=False)
    backend = CudaPackedBackend()
    prepared = backend.prepare(pack_logical_layer(logical), "cuda")
    value = torch.linspace(-0.5, 0.75, 70, dtype=torch.float32).reshape(2, 35).cuda()

    actual = backend.linear(value, prepared)

    torch.testing.assert_close(actual.cpu(), _expected(value.cpu(), logical), rtol=2e-5, atol=2e-4)
