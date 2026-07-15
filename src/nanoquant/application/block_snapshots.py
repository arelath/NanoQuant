"""Versioned model-level pre/post-KD block snapshot contracts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from nanoquant.application.loss_snapshots import compare_global_tuning_losses
from nanoquant.config.codec import canonical_json
from nanoquant.domain.models import BlockId, GlobalTuningBlockMetrics


@dataclass(frozen=True, slots=True)
class BlockSnapshotProtocol:
    version: str
    token_hash: str
    sample_count: int
    sequence_length: int
    loss_kind: str
    reference_dtype: str
    accumulation_dtype: str
    pad_token_id: int | None
    denominator_floor: float

    @property
    def semantic_key(self) -> str:
        return "sha256:" + hashlib.sha256(canonical_json(self).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class BlockSnapshotSelection:
    token_ids: torch.Tensor
    protocol: BlockSnapshotProtocol


def select_block_snapshot_tokens(
    token_ids: torch.Tensor,
    *,
    maximum_samples: int,
    maximum_tokens: int,
    pad_token_id: int | None = None,
    denominator_floor: float = 1e-12,
) -> BlockSnapshotSelection:
    if token_ids.ndim != 2 or token_ids.shape[0] <= 0 or token_ids.shape[1] <= 0:
        raise ValueError("block snapshot tokens must be a non-empty rank-two tensor")
    if maximum_samples <= 0 or maximum_tokens <= 0:
        raise ValueError("block snapshot sample and token limits must be positive")
    if pad_token_id is not None and pad_token_id < 0:
        raise ValueError("block snapshot pad token ID must be non-negative when provided")
    if not math.isfinite(denominator_floor) or denominator_floor < 0:
        raise ValueError("block snapshot denominator floor must be finite and non-negative")
    selected = token_ids[:maximum_samples, :maximum_tokens].detach().cpu().long().contiguous()
    token_hash = "sha256:" + hashlib.sha256(selected.view(torch.uint8).numpy().tobytes()).hexdigest()
    protocol = BlockSnapshotProtocol(
        "block-hidden-unweighted-mse-v1",
        token_hash,
        selected.shape[0],
        selected.shape[1],
        "unweighted_hidden_state_mse",
        "bfloat16",
        "float32",
        pad_token_id,
        denominator_floor,
    )
    return BlockSnapshotSelection(selected, protocol)


def compare_block_snapshots(
    blocks: tuple[BlockId, ...],
    final_frozen_pre_kd: tuple[float, ...],
    final_post_kd: tuple[float, ...],
    protocol: BlockSnapshotProtocol,
) -> tuple[GlobalTuningBlockMetrics, ...]:
    if not blocks or len(blocks) != len(final_frozen_pre_kd) or len(blocks) != len(final_post_kd):
        raise ValueError("block snapshot losses require equal non-empty block, pre-KD, and post-KD values")
    if tuple(block.index for block in blocks) != tuple(range(len(blocks))):
        raise ValueError("block snapshot identities must be contiguous and zero-based")
    if any(not math.isfinite(value) or value < 0 for value in (*final_frozen_pre_kd, *final_post_kd)):
        raise ValueError("block snapshot losses must be finite and non-negative")
    return tuple(
        compare_global_tuning_losses(block, before, after, protocol.denominator_floor)
        for block, before, after in zip(
            blocks,
            final_frozen_pre_kd,
            final_post_kd,
            strict=True,
        )
    )
