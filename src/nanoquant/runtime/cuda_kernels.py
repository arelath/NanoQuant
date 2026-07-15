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
