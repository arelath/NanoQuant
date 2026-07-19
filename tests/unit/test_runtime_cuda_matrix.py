from __future__ import annotations

from itertools import product

import pytest
import torch

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.runtime import (
    CudaPackedBackend,
    LogicalLayerState,
    QuantizedLinearSpec,
    WorkloadSpec,
    pack_logical_layer,
)

pytestmark = [pytest.mark.cuda, pytest.mark.slow]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_OUTLIER_DTYPES = (None, torch.float16, torch.bfloat16, torch.float32, torch.int8)


@pytest.fixture(scope="module", autouse=True)
def _leased_cuda_matrix(tmp_path_factory: pytest.TempPathFactory):
    if not torch.cuda.is_available():
        yield
        return
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path_factory.mktemp("triton-matrix-cache")))
    try:
        with acquire_device_lease("cuda:0"):
            yield
    finally:
        monkeypatch.undo()


def _logical_state(
    *,
    factor_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    outlier_dtype: torch.dtype | None,
    bias: bool,
) -> LogicalLayerState:
    in_features = 35
    out_features = 17
    rank = 33
    outlier_count = 0 if outlier_dtype is None else 3
    spec = QuantizedLinearSpec(
        "blocks.0.self_attn.q_proj",
        "nanoquant-v1",
        in_features,
        out_features,
        rank,
        str(factor_dtype).removeprefix("torch."),
        str(scale_dtype).removeprefix("torch."),
        outlier_count=outlier_count,
        outlier_value_dtype=(
            None if outlier_dtype is None else str(outlier_dtype).removeprefix("torch.")
        ),
        has_outlier_scales=outlier_dtype is torch.int8,
        has_bias=bias,
    )
    generator = torch.Generator().manual_seed(20260715)
    left = torch.where(
        torch.rand(out_features, rank, generator=generator) > 0.5,
        1.0,
        -1.0,
    ).to(factor_dtype)
    right = torch.where(
        torch.rand(rank, in_features, generator=generator) > 0.5,
        1.0,
        -1.0,
    ).to(factor_dtype)
    scale_pre = torch.randn(in_features, generator=generator).to(scale_dtype)
    indices = torch.tensor((0, 17, 34), dtype=torch.int64) if outlier_count else None
    if indices is not None:
        scale_pre[indices] = 0
    if outlier_dtype is None:
        outlier_values = None
    elif outlier_dtype is torch.int8:
        outlier_values = torch.randint(
            -8,
            8,
            (out_features, outlier_count),
            generator=generator,
            dtype=torch.int8,
        )
    else:
        outlier_values = torch.randn(
            out_features,
            outlier_count,
            generator=generator,
        ).to(outlier_dtype)
    return LogicalLayerState(
        spec,
        left,
        right,
        scale_pre,
        torch.randn(rank, generator=generator).to(scale_dtype),
        torch.randn(out_features, generator=generator).to(scale_dtype),
        bias=(
            torch.randn(out_features, generator=generator).to(scale_dtype)
            if bias
            else None
        ),
        outlier_indices=indices,
        outlier_values=outlier_values,
        outlier_scales=(
            torch.tensor((0.125, 0.25, 0.375), dtype=scale_dtype)
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
        outlier_weights = state.outlier_values.float()
        if state.outlier_scales is not None:
            outlier_weights *= state.outlier_scales.float()
        output += torch.nn.functional.linear(
            value_float.index_select(-1, state.outlier_indices),
            outlier_weights,
        )
    if state.bias is not None:
        output += state.bias.float()
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_backend_full_declared_dtype_outlier_workload_matrix() -> None:
    """Cross every finite CUDA capability dimension on packed word-tail geometry."""

    backend = CudaPackedBackend()
    executed = 0
    for factor_dtype, scale_dtype, outlier_dtype, bias in product(
        _FLOAT_DTYPES,
        _FLOAT_DTYPES,
        _OUTLIER_DTYPES,
        (False, True),
    ):
        logical = _logical_state(
            factor_dtype=factor_dtype,
            scale_dtype=scale_dtype,
            outlier_dtype=outlier_dtype,
            bias=bias,
        )
        packed = pack_logical_layer(logical)
        prepared = backend.prepare(packed, "cuda:0")
        for input_dtype, workload_kind in product(
            _FLOAT_DTYPES,
            ("prefill", "decode"),
        ):
            token_shape = (2, 3) if workload_kind == "prefill" else (2,)
            workload = WorkloadSpec(
                workload_kind,
                "cuda",
                str(input_dtype).removeprefix("torch."),
                2,
                3 if workload_kind == "prefill" else 1,
                deterministic=True,
            )
            support = backend.supports(packed.spec, workload)
            assert support.supported, (
                factor_dtype,
                scale_dtype,
                outlier_dtype,
                bias,
                input_dtype,
                workload_kind,
                support,
            )
            value = torch.linspace(
                -0.75,
                0.875,
                35 * 6 if workload_kind == "prefill" else 35 * 2,
                dtype=input_dtype,
            ).reshape(*token_shape, 35).cuda()
            actual = backend.linear(value, prepared)
            repeated = backend.linear(value, prepared)
            expected = _expected(value.cpu(), logical)
            case = (
                factor_dtype,
                scale_dtype,
                outlier_dtype,
                bias,
                input_dtype,
                workload_kind,
            )
            torch.testing.assert_close(
                actual.cpu(),
                expected,
                rtol=2e-5,
                atol=2e-4,
                msg=lambda message, case=case: f"CUDA packed matrix case {case}: {message}",
            )
            assert torch.equal(actual, repeated), case
            executed += 1
    assert executed == 540
