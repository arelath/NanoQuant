# mypy: ignore-errors
"""Triton port of the pinned modified llama.cpp NanoQuant CUDA linear operation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _nanoquant_stage1(
    value,
    right_words,
    scale_pre,
    scale_mid,
    latent,
    N_IN: tl.constexpr,
    N_RANK: tl.constexpr,
    WORDS_PER_ROW: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
):
    rank_block = tl.program_id(0)
    token = tl.program_id(1)
    ranks = rank_block * BLOCK_RANK + tl.arange(0, BLOCK_RANK)
    rank_mask = ranks < N_RANK
    accumulator = tl.zeros((BLOCK_RANK,), dtype=tl.float32)
    for start in range(0, N_IN, BLOCK_IN):
        columns = start + tl.arange(0, BLOCK_IN)
        column_mask = columns < N_IN
        words = tl.load(
            right_words + ranks[:, None] * WORDS_PER_ROW + (columns[None, :] // 32),
            mask=rank_mask[:, None] & column_mask[None, :],
            other=0,
        )
        bits = (words >> (columns[None, :] & 31)) & 1
        signs = 1.0 - 2.0 * bits.to(tl.float32)
        inputs = tl.load(value + token * N_IN + columns, mask=column_mask, other=0.0).to(tl.float32)
        pre = tl.load(scale_pre + columns, mask=column_mask, other=0.0).to(tl.float32)
        accumulator += tl.sum(signs * (inputs * pre)[None, :], axis=1)
    mid = tl.load(scale_mid + ranks, mask=rank_mask, other=0.0).to(tl.float32)
    tl.store(latent + token * N_RANK + ranks, accumulator * mid, mask=rank_mask)


@triton.jit
def _nanoquant_stage2(
    value,
    latent,
    left_words,
    scale_post,
    salient_indices,
    salient_values,
    salient_scales,
    bias,
    output,
    N_IN: tl.constexpr,
    N_OUT: tl.constexpr,
    N_RANK: tl.constexpr,
    N_SALIENT: tl.constexpr,
    WORDS_PER_ROW: tl.constexpr,
    HAS_SALIENT_SCALES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
    BLOCK_SALIENT: tl.constexpr,
):
    output_block = tl.program_id(0)
    token = tl.program_id(1)
    outputs = output_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    output_mask = outputs < N_OUT
    accumulator = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    for start in range(0, N_RANK, BLOCK_RANK):
        ranks = start + tl.arange(0, BLOCK_RANK)
        rank_mask = ranks < N_RANK
        words = tl.load(
            left_words + outputs[:, None] * WORDS_PER_ROW + (ranks[None, :] // 32),
            mask=output_mask[:, None] & rank_mask[None, :],
            other=0,
        )
        bits = (words >> (ranks[None, :] & 31)) & 1
        signs = 1.0 - 2.0 * bits.to(tl.float32)
        hidden = tl.load(latent + token * N_RANK + ranks, mask=rank_mask, other=0.0)
        accumulator += tl.sum(signs * hidden[None, :], axis=1)
    post = tl.load(scale_post + outputs, mask=output_mask, other=0.0).to(tl.float32)
    accumulator *= post
    if N_SALIENT > 0:
        for start in range(0, N_SALIENT, BLOCK_SALIENT):
            salient = start + tl.arange(0, BLOCK_SALIENT)
            salient_mask = salient < N_SALIENT
            indices = tl.load(salient_indices + salient, mask=salient_mask, other=0)
            selected = tl.load(
                value + token * N_IN + indices,
                mask=salient_mask,
                other=0.0,
            ).to(tl.float32)
            weights = tl.load(
                salient_values + outputs[:, None] * N_SALIENT + salient[None, :],
                mask=output_mask[:, None] & salient_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if HAS_SALIENT_SCALES:
                scales = tl.load(salient_scales + salient, mask=salient_mask, other=0.0).to(tl.float32)
                weights *= scales[None, :]
            accumulator += tl.sum(weights * selected[None, :], axis=1)
    if HAS_BIAS:
        accumulator += tl.load(bias + outputs, mask=output_mask, other=0.0).to(tl.float32)
    tl.store(output + token * N_OUT + outputs, accumulator, mask=output_mask)


@triton.jit
def _mul_rn_f32(left, right):
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(left, right):
    return tl.inline_asm_elementwise(
        asm="add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _nanoquant_decode_rope(
    query,
    key,
    cosine,
    sine,
    query_output,
    key_output,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    head = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_DIM)
    mask = offsets < HEAD_DIM
    half = HEAD_DIM // 2
    rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
    cosines = tl.load(cosine + offsets, mask=mask, other=0.0).to(tl.float32)
    sines = tl.load(sine + offsets, mask=mask, other=0.0).to(tl.float32)
    query_values = tl.load(query + head * HEAD_DIM + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    query_rotated = tl.load(
        query + head * HEAD_DIM + rotated_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    query_rotated = tl.where(offsets < half, -query_rotated, query_rotated)
    query_product = _mul_rn_f32(query_values, cosines)
    query_rotated_product = _mul_rn_f32(query_rotated, sines)
    tl.store(
        query_output + head * HEAD_DIM + offsets,
        _add_rn_f32(query_product, query_rotated_product),
        mask=mask,
    )
    if head == 0:
        key_values = tl.load(key + offsets, mask=mask, other=0.0).to(tl.float32)
        key_rotated = tl.load(key + rotated_offsets, mask=mask, other=0.0).to(tl.float32)
        key_rotated = tl.where(offsets < half, -key_rotated, key_rotated)
        key_product = _mul_rn_f32(key_values, cosines)
        key_rotated_product = _mul_rn_f32(key_rotated, sines)
        tl.store(
            key_output + offsets,
            _add_rn_f32(key_product, key_rotated_product),
            mask=mask,
        )


@triton.jit
def _nanoquant_cache_prefix_update(
    key_states,
    value_states,
    key_cache,
    value_cache,
    key_output,
    value_output,
    position_start,
    TOKEN_COUNT: tl.constexpr,
    CACHE_LENGTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOTAL_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < TOTAL_ELEMENTS
    dimension = offsets % HEAD_DIM
    cache_position = (offsets // HEAD_DIM) % CACHE_LENGTH
    outer = offsets // (CACHE_LENGTH * HEAD_DIM)
    updated = valid & (cache_position >= position_start) & (
        cache_position < position_start + TOKEN_COUNT
    )
    source_offsets = (
        outer * TOKEN_COUNT * HEAD_DIM
        + (cache_position - position_start) * HEAD_DIM
        + dimension
    )

    existing_keys = tl.load(key_cache + offsets, mask=valid, other=0.0)
    existing_values = tl.load(value_cache + offsets, mask=valid, other=0.0)
    new_keys = tl.load(key_states + source_offsets, mask=updated, other=0.0).to(tl.float16)
    new_values = tl.load(value_states + source_offsets, mask=updated, other=0.0).to(tl.float16)
    selected_keys = tl.where(updated, new_keys, existing_keys)
    selected_values = tl.where(updated, new_values, existing_values)

    tl.store(key_cache + offsets, selected_keys, mask=updated)
    tl.store(value_cache + offsets, selected_values, mask=updated)
    tl.store(key_output + offsets, selected_keys.to(tl.float32), mask=valid)
    tl.store(value_output + offsets, selected_values.to(tl.float32), mask=valid)


@triton.jit
def _nanoquant_bfloat16_output_projection(
    value,
    weight,
    output,
    N_IN: tl.constexpr,
    N_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    output_block = tl.program_id(0)
    token = tl.program_id(1)
    outputs = output_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    output_mask = outputs < N_OUT
    accumulator = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    for start in range(0, N_IN, BLOCK_IN):
        columns = start + tl.arange(0, BLOCK_IN)
        column_mask = columns < N_IN
        inputs = tl.load(
            value + token * N_IN + columns,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(
            weight + outputs[:, None] * N_IN + columns[None, :],
            mask=output_mask[:, None] & column_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(weights * inputs[None, :], axis=1)
    tl.store(output + token * N_OUT + outputs, accumulator, mask=output_mask)


@triton.jit
def _nanoquant_bfloat16_embedding(
    input_ids,
    weight,
    scale,
    output,
    EMBEDDING_DIM: tl.constexpr,
    TOTAL_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < TOTAL_ELEMENTS
    tokens = offsets // EMBEDDING_DIM
    dimensions = offsets % EMBEDDING_DIM
    rows = tl.load(input_ids + tokens, mask=valid, other=0)
    values = tl.load(
        weight + rows * EMBEDDING_DIM + dimensions,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    embedding_scale = tl.load(scale).to(tl.float32)
    tl.store(output + offsets, _mul_rn_f32(values, embedding_scale), mask=valid)


def launch_bfloat16_embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Look up native BF16 rows and apply the exact F32 Gemma scale."""

    if input_ids.dtype != torch.int64 or weight.dtype != torch.bfloat16:
        raise ValueError("embedding requires I64 indices and BF16 weights")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("embedding requires one F32 scale")
    if input_ids.device.type != "cuda" or weight.device != input_ids.device:
        raise ValueError("embedding requires indices and weights on one CUDA device")
    if scale.device != input_ids.device:
        raise ValueError("embedding scale must share the CUDA device")
    if weight.ndim != 2 or input_ids.ndim < 1:
        raise ValueError("embedding geometry is incompatible")
    if not input_ids.is_contiguous() or not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("embedding requires contiguous tensors")
    embedding_dim = weight.shape[1]
    output = torch.empty(
        (*input_ids.shape, embedding_dim),
        dtype=torch.float32,
        device=input_ids.device,
    )
    total_elements = output.numel()
    block_size = 256
    _nanoquant_bfloat16_embedding[(triton.cdiv(total_elements, block_size),)](
        input_ids,
        weight,
        scale,
        output,
        EMBEDDING_DIM=embedding_dim,
        TOTAL_ELEMENTS=total_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def launch_bfloat16_output_projection(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Project F32 states through a contiguous BF16 vocabulary matrix."""

    if value.dtype != torch.float32 or weight.dtype != torch.bfloat16:
        raise ValueError("output projection requires F32 inputs and BF16 weights")
    if value.device.type != "cuda" or weight.device != value.device:
        raise ValueError("output projection requires inputs on one CUDA device")
    if value.ndim < 2 or weight.ndim != 2 or value.shape[-1] != weight.shape[1]:
        raise ValueError("output projection geometry is incompatible")
    if not value.is_contiguous() or not weight.is_contiguous():
        raise ValueError("output projection requires contiguous tensors")
    n_in = value.shape[-1]
    n_out = weight.shape[0]
    token_count = value.numel() // n_in
    output = torch.empty((token_count, n_out), dtype=torch.float32, device=value.device)
    _nanoquant_bfloat16_output_projection[(triton.cdiv(n_out, 32), token_count)](
        value.view(token_count, n_in),
        weight,
        output,
        N_IN=n_in,
        N_OUT=n_out,
        BLOCK_IN=256,
        BLOCK_OUT=32,
        num_warps=8,
    )
    return output.view(*value.shape[:-1], n_out)


def launch_cache_prefix_update(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update an F16 KV prefix and materialize its exact F32 attention views."""

    if key_states.dtype != torch.float32 or value_states.dtype != torch.float32:
        raise ValueError("cache prefix update requires F32 key and value states")
    if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("cache prefix update requires F16 backing caches")
    if key_states.device.type != "cuda" or value_states.device != key_states.device:
        raise ValueError("cache prefix update requires key and value states on one CUDA device")
    if key_cache.device != key_states.device or value_cache.device != key_states.device:
        raise ValueError("cache prefix update inputs must share one CUDA device")
    if not all(
        tensor.is_contiguous()
        for tensor in (key_states, value_states, key_cache, value_cache)
    ):
        raise ValueError("cache prefix update requires contiguous tensors")
    if key_states.ndim != 4 or key_cache.ndim != 4:
        raise ValueError("cache prefix update requires four-dimensional tensors")
    if key_states.shape != value_states.shape or key_cache.shape != value_cache.shape:
        raise ValueError("cache prefix update requires matching key/value shapes")
    if (
        key_states.shape[0] != key_cache.shape[0]
        or key_states.shape[1] != key_cache.shape[1]
        or key_states.shape[3] != key_cache.shape[3]
    ):
        raise ValueError("cache prefix update state and cache geometry is incompatible")
    token_count = key_states.shape[2]
    cache_length = key_cache.shape[2]
    if position_start < 0 or position_start + token_count >= cache_length:
        raise ValueError("cache prefix update must be strictly before cache rollover")

    key_output = torch.empty(key_cache.shape, dtype=torch.float32, device=key_cache.device)
    value_output = torch.empty(value_cache.shape, dtype=torch.float32, device=value_cache.device)
    total_elements = key_cache.numel()
    block_size = 256
    _nanoquant_cache_prefix_update[(triton.cdiv(total_elements, block_size),)](
        key_states,
        value_states,
        key_cache,
        value_cache,
        key_output,
        value_output,
        position_start,
        TOKEN_COUNT=token_count,
        CACHE_LENGTH=cache_length,
        HEAD_DIM=key_cache.shape[3],
        TOTAL_ELEMENTS=total_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return key_output, value_output


def launch_decode_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the exact pinned Gemma one-token F32 rotary geometry."""

    expected_query = (1, 4, 1, 256)
    expected_key = (1, 1, 1, 256)
    if tuple(query.shape) != expected_query or tuple(key.shape) != expected_key:
        raise ValueError("decode RoPE requires the pinned batch/head/token geometry")
    if query.dtype != torch.float32 or key.dtype != torch.float32:
        raise ValueError("decode RoPE requires F32 query and key tensors")
    if query.device.type != "cuda" or key.device != query.device:
        raise ValueError("decode RoPE requires query and key on one CUDA device")
    if not query.is_contiguous() or not key.is_contiguous():
        raise ValueError("decode RoPE requires contiguous query and key tensors")
    if cosine.dtype != torch.float32 or sine.dtype != torch.float32:
        raise ValueError("decode RoPE requires F32 cosine and sine tensors")
    if tuple(cosine.shape) != (1, 1, 256) or tuple(sine.shape) != (1, 1, 256):
        raise ValueError("decode RoPE requires the pinned cosine and sine geometry")
    if cosine.device != query.device or sine.device != query.device:
        raise ValueError("decode RoPE inputs must share one CUDA device")
    if not cosine.is_contiguous() or not sine.is_contiguous():
        raise ValueError("decode RoPE requires contiguous cosine and sine tensors")
    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)
    _nanoquant_decode_rope[(4,)](
        query,
        key,
        cosine,
        sine,
        query_output,
        key_output,
        HEAD_DIM=256,
        BLOCK_DIM=256,
        num_warps=4,
    )
    return query_output, key_output


def launch_packed_linear(
    value: torch.Tensor,
    left_words: torch.Tensor,
    right_words: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    bias: torch.Tensor | None,
    salient_indices: torch.Tensor | None,
    salient_values: torch.Tensor | None,
    salient_scales: torch.Tensor | None,
) -> torch.Tensor:
    """Launch the two-stage packed operation on the current PyTorch CUDA stream."""

    n_in = value.shape[-1]
    n_rank = scale_mid.numel()
    n_out = scale_post.numel()
    token_count = value.numel() // n_in
    flattened = value.view(token_count, n_in)
    latent = torch.empty((token_count, n_rank), dtype=torch.float32, device=value.device)
    output = torch.empty((token_count, n_out), dtype=torch.float32, device=value.device)
    _nanoquant_stage1[(triton.cdiv(n_rank, 8), token_count)](
        flattened,
        right_words,
        scale_pre,
        scale_mid,
        latent,
        N_IN=n_in,
        N_RANK=n_rank,
        WORDS_PER_ROW=right_words.shape[1],
        BLOCK_IN=256,
        BLOCK_RANK=8,
        num_warps=4,
    )
    n_salient = 0 if salient_indices is None else salient_indices.numel()
    dummy = scale_pre
    _nanoquant_stage2[(triton.cdiv(n_out, 8), token_count)](
        flattened,
        latent,
        left_words,
        scale_post,
        dummy if salient_indices is None else salient_indices,
        dummy if salient_values is None else salient_values,
        dummy if salient_scales is None else salient_scales,
        dummy if bias is None else bias,
        output,
        N_IN=n_in,
        N_OUT=n_out,
        N_RANK=n_rank,
        N_SALIENT=n_salient,
        WORDS_PER_ROW=left_words.shape[1],
        HAS_SALIENT_SCALES=salient_scales is not None,
        HAS_BIAS=bias is not None,
        BLOCK_OUT=8,
        BLOCK_RANK=256,
        BLOCK_SALIENT=32,
        num_warps=4,
    )
    return output.view(*value.shape[:-1], n_out)
