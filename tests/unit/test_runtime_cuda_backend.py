from __future__ import annotations

import importlib.util
from contextlib import nullcontext

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM, apply_rotary_pos_emb

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.runtime import (
    CUDA_PACKED_REFERENCE_SHA256,
    CudaPackedBackend,
    GenerationRequest,
    LogicalLayerState,
    QuantizedLinearSpec,
    TransformersGenerationModel,
    WorkloadSpec,
    batch_prompts,
    generate,
    hybrid_cache_factory,
    pack_logical_layer,
    plan_execution_workloads,
    prepare_execution_workloads,
)
from nanoquant.runtime.cuda_backend import (
    grouped_cuda_projection,
    prepare_cuda_projection_group,
)
from nanoquant.runtime.cuda_kernels import (
    launch_bfloat16_embedding,
    launch_bfloat16_output_projection,
    launch_cache_prefix_update,
    launch_decode_attention,
    launch_decode_rope,
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


def _group_logical(index: int, rank: int, out_features: int) -> LogicalLayerState:
    generator = torch.Generator().manual_seed(700 + index)
    in_features = 64
    outlier_count = 2
    spec = QuantizedLinearSpec(
        f"blocks.0.self_attn.group_{index}",
        "nanoquant-v1",
        in_features,
        out_features,
        rank,
        "float32",
        "bfloat16",
        outlier_count=outlier_count,
        outlier_value_dtype="bfloat16",
    )
    left = torch.where(
        torch.rand(out_features, rank, generator=generator) > 0.5,
        1.0,
        -1.0,
    )
    right = torch.where(
        torch.rand(rank, in_features, generator=generator) > 0.5,
        1.0,
        -1.0,
    )
    scale_pre = torch.randn(in_features, generator=generator).bfloat16()
    outlier_indices = torch.tensor((1, 7), dtype=torch.int64)
    scale_pre[outlier_indices] = 0
    return LogicalLayerState(
        spec,
        left,
        right,
        scale_pre,
        torch.randn(rank, generator=generator).bfloat16(),
        torch.randn(out_features, generator=generator).bfloat16(),
        outlier_indices=outlier_indices,
        outlier_values=torch.randn(
            out_features,
            outlier_count,
            generator=generator,
        ).bfloat16(),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_count", [2, 3])
def test_grouped_cuda_projection_closely_matches_individual_linears(
    group_count: int,
) -> None:
    backend = CudaPackedBackend()
    logical = (
        _group_logical(0, 32, 24),
        _group_logical(1, 40, 16),
        _group_logical(2, 48, 32),
    )
    prepared = tuple(
        backend.prepare(pack_logical_layer(state), "cuda") for state in logical[:group_count]
    )
    group = prepare_cuda_projection_group(prepared)
    assert group is not None
    value = torch.randn(1, 1, 64, generator=torch.Generator().manual_seed(99)).cuda()

    actual = grouped_cuda_projection(value, group)
    expected = tuple(backend.linear(value, layer) for layer in prepared)

    assert len(actual) == group_count
    maximum_errors = tuple(
        float((candidate - reference).abs().max().item())
        for candidate, reference in zip(actual, expected, strict=True)
    )
    assert max(maximum_errors) <= 5e-5
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference, rtol=2e-5, atol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("batch_size", "head_count", "token_count", "position_start"), [(1, 1, 1, 16), (2, 2, 5, 0)])
def test_fused_cache_prefix_update_matches_pytorch_storage_and_views(
    batch_size: int,
    head_count: int,
    token_count: int,
    position_start: int,
) -> None:
    generator = torch.Generator().manual_seed(20260715)
    cache_length = 24
    head_dim = 8
    key_states = torch.randn(
        batch_size,
        head_count,
        token_count,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).cuda()
    value_states = torch.randn(
        batch_size,
        head_count,
        token_count,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).cuda()
    key_cache = torch.randn(
        batch_size,
        head_count,
        cache_length,
        head_dim,
        generator=generator,
        dtype=torch.float16,
    ).cuda()
    value_cache = torch.randn(
        batch_size,
        head_count,
        cache_length,
        head_dim,
        generator=generator,
        dtype=torch.float16,
    ).cuda()
    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    expected_key_cache[:, :, position_start : position_start + token_count] = key_states.to(
        torch.float16
    )
    expected_value_cache[:, :, position_start : position_start + token_count] = value_states.to(
        torch.float16
    )

    key_view, value_view = launch_cache_prefix_update(
        key_states,
        value_states,
        key_cache,
        value_cache,
        position_start,
    )

    assert torch.equal(key_cache, expected_key_cache)
    assert torch.equal(value_cache, expected_value_cache)
    assert torch.equal(key_view, expected_key_cache.float())
    assert torch.equal(value_view, expected_value_cache.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("token_shape", [(1, 1), (2, 3)])
def test_bfloat16_output_projection_matches_f32_reference(
    token_shape: tuple[int, ...],
) -> None:
    generator = torch.Generator().manual_seed(20260715)
    value = torch.randn(
        *token_shape,
        1152,
        generator=generator,
        dtype=torch.float32,
    ).cuda()
    weight = torch.randn(
        257,
        1152,
        generator=generator,
        dtype=torch.bfloat16,
    ).cuda()

    actual = launch_bfloat16_output_projection(value, weight)
    expected = torch.nn.functional.linear(value, weight.float())

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bfloat16_embedding_matches_promoted_f32_lookup_exactly() -> None:
    generator = torch.Generator().manual_seed(20260715)
    input_ids = torch.tensor(((3, 17, 5), (9, 2, 21)), dtype=torch.int64).cuda()
    weight = torch.randn(32, 1152, generator=generator, dtype=torch.bfloat16).cuda()
    scale = torch.tensor(1152**0.5, dtype=torch.float32).cuda()

    actual = launch_bfloat16_embedding(input_ids, weight, scale)
    expected = torch.nn.functional.embedding(input_ids, weight).float() * scale

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("cache_length", [16, 48])
def test_decode_attention_matches_eager_grouped_query_operation(cache_length: int) -> None:
    generator = torch.Generator().manual_seed(20260715)
    query = torch.randn(1, 4, 1, 256, generator=generator).cuda()
    key = torch.randn(1, 1, cache_length, 256, generator=generator).cuda()
    value = torch.randn(1, 1, cache_length, 256, generator=generator).cuda()
    attention_mask = torch.zeros(1, 1, 1, cache_length).cuda()
    attention_mask[..., -3:] = -torch.inf
    repeated_key = key.repeat_interleave(4, dim=1)
    repeated_value = value.repeat_interleave(4, dim=1)
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * 0.0625
    probabilities = torch.softmax(scores + attention_mask, dim=-1, dtype=torch.float32)
    expected = torch.matmul(probabilities, repeated_value).transpose(1, 2).contiguous()

    actual = launch_decode_attention(query, key, value, attention_mask, 0.0625)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cache_prefix_generation_matches_control_across_sliding_rollover() -> None:
    torch.manual_seed(20260715)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=16,
        sliding_window_pattern=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
    )
    model = Gemma3ForCausalLM(config).eval().cuda()
    tokens, mask = batch_prompts(((2, 4), (2, 5, 6)), pad_token_id=0)
    request = GenerationRequest(
        tokens.cuda(),
        mask.cuda(),
        32,
        (1000,),
        0,
        stopping_check_interval=8,
    )

    def run(fused_cache_prefix: bool) -> tuple[torch.Tensor, object]:
        created_caches: list[object] = []
        factory = hybrid_cache_factory(
            config,
            torch.float16,
            fast_sliding_prefix=True,
            fused_cache_prefix=fused_cache_prefix,
        )

        def capture_cache(
            batch_size: int,
            maximum_cache_length: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> object:
            cache = factory(batch_size, maximum_cache_length, device, dtype)
            created_caches.append(cache)
            return cache

        result = generate(
            request,
            TransformersGenerationModel(model, capture_cache, lambda kind: nullcontext()),
        )
        assert len(created_caches) == 1
        return result.token_ids, created_caches[0]

    control_tokens, _control_cache = run(False)
    fused_tokens, fused_cache = run(True)

    assert torch.equal(fused_tokens, control_tokens)
    assert fused_cache.nanoquant_fused_cache_update_count > 0


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_packed_backend_shares_weights_across_prefill_and_decode_plans() -> None:
    logical = _logical()
    packed = pack_logical_layer(logical)
    backend = CudaPackedBackend()
    plans = plan_execution_workloads(
        (packed.spec,),
        prefill=WorkloadSpec("prefill", "cuda", "bfloat16", 1, 4, deterministic=True),
        decode=WorkloadSpec("decode", "cuda", "bfloat16", 1, 1, deterministic=True),
        prefill_backends=(backend,),
        decode_backends=(backend,),
        strict=True,
    )
    prepared = prepare_execution_workloads(
        plans,
        {packed.spec.name: packed},
        (backend,),
        "cuda",
    )
    prefill = torch.linspace(-1, 1, 140, dtype=torch.bfloat16).reshape(1, 4, 35).cuda()
    decode = prefill[:, -1, :].contiguous()

    prefill_output = prepared.prefill.linear_at(0, prefill)
    decode_output = prepared.decode.linear_at(0, decode)

    assert prepared.prefill.dispatches[0].layer is prepared.decode.dispatches[0].layer
    torch.testing.assert_close(prefill_output.cpu(), _expected(prefill.cpu(), logical), rtol=2e-5, atol=2e-4)
    torch.testing.assert_close(decode_output.cpu(), _expected(decode.cpu(), logical), rtol=2e-5, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_decode_rope_matches_pinned_gemma_geometry() -> None:
    generator = torch.Generator().manual_seed(20260715)
    query = torch.randn(1, 4, 1, 256, generator=generator).cuda()
    key = torch.randn(1, 1, 1, 256, generator=generator).cuda()
    cosine = torch.randn(1, 1, 256, generator=generator).cuda()
    sine = torch.randn(1, 1, 256, generator=generator).cuda()

    expected_query, expected_key = apply_rotary_pos_emb(query, key, cosine, sine)
    actual_query, actual_key = launch_decode_rope(query, key, cosine, sine)
    repeated_query, repeated_key = launch_decode_rope(query, key, cosine, sine)

    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)
    assert torch.equal(actual_query, repeated_query)
    assert torch.equal(actual_key, repeated_key)
