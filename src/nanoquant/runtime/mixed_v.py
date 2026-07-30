"""Compact mixed free/codebook representation of a packed right factor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nanoquant.runtime.backend import QuantizedLinearSpec
from nanoquant.runtime.packed import (
    PACKED_LAYOUT_VERSION,
    PACKED_WORD_BITS,
    PackedLayerState,
    packed_word_count,
)

MIXED_V_FORMAT_VERSION = "mixed-v-free-k10-pair9-v1"
MIXED_V_INDEX_BITS = 10
MIXED_V_CORRECTION_BITS = 9
MIXED_V_RECORD_BITS = MIXED_V_INDEX_BITS + MIXED_V_CORRECTION_BITS
MIXED_V_CODEBOOK_SIZE = 1 << MIXED_V_INDEX_BITS
MIXED_V_CORRECTION_PAIR_COUNT = math.comb(PACKED_WORD_BITS, 2)


def mixed_v_payload_word_count(record_count: int) -> int:
    if record_count <= 0:
        raise ValueError("mixed-V record count must be positive")
    return (record_count * MIXED_V_RECORD_BITS + PACKED_WORD_BITS - 1) // PACKED_WORD_BITS


def mixed_v_correction_pair_table(
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return the fixed pair-id table; it is algorithmic and is never persisted."""

    pairs = tuple(
        (first, second)
        for first in range(PACKED_WORD_BITS)
        for second in range(first + 1, PACKED_WORD_BITS)
    )
    table = torch.zeros(1 << MIXED_V_CORRECTION_BITS, dtype=torch.int32, device=device)
    for index, (first, second) in enumerate(pairs):
        table[index] = first | (second << 8)
    return table


def _validate_record_fields(
    codebook_indices: torch.Tensor,
    correction_pair_ids: torch.Tensor,
) -> None:
    if (
        codebook_indices.ndim != 2
        or codebook_indices.shape != correction_pair_ids.shape
        or not codebook_indices.is_contiguous()
        or not correction_pair_ids.is_contiguous()
        or codebook_indices.numel() == 0
    ):
        raise ValueError("mixed-V record fields must be matching non-empty contiguous matrices")
    if codebook_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("mixed-V codebook indices must use an integer dtype")
    if correction_pair_ids.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("mixed-V correction pair ids must use an integer dtype")
    if bool(torch.any(codebook_indices < 0)) or bool(
        torch.any(codebook_indices >= MIXED_V_CODEBOOK_SIZE)
    ):
        raise ValueError("mixed-V codebook index is outside the 10-bit codebook")
    if bool(torch.any(correction_pair_ids < 0)) or bool(
        torch.any(correction_pair_ids >= MIXED_V_CORRECTION_PAIR_COUNT)
    ):
        raise ValueError("mixed-V correction pair id is outside the 496 valid pairs")


def pack_mixed_v_records(
    codebook_indices: torch.Tensor,
    correction_pair_ids: torch.Tensor,
) -> torch.Tensor:
    """Pack row-major `(10-bit index, 9-bit pair)` records into minimal I32 words."""

    _validate_record_fields(codebook_indices, correction_pair_ids)
    indices = codebook_indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    pairs = correction_pair_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    records = indices | (pairs << MIXED_V_INDEX_BITS)
    positions = torch.arange(records.numel(), dtype=torch.int64)
    bit_offsets = positions * MIXED_V_RECORD_BITS
    word_indices = bit_offsets // PACKED_WORD_BITS
    shifts = bit_offsets % PACKED_WORD_BITS
    packed = torch.zeros(mixed_v_payload_word_count(records.numel()), dtype=torch.int64)
    packed.index_add_(0, word_indices, (records << shifts) & 0xFFFF_FFFF)
    crosses = shifts > PACKED_WORD_BITS - MIXED_V_RECORD_BITS
    packed.index_add_(
        0,
        word_indices[crosses] + 1,
        records[crosses] >> (PACKED_WORD_BITS - shifts[crosses]),
    )
    if bool(torch.any(packed > 0xFFFF_FFFF)):
        raise AssertionError("mixed-V record fields overlap")
    return packed.to(torch.int32).contiguous()


def unpack_mixed_v_records(
    payload: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode and validate the exact minimal 19-bit payload."""

    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("mixed-V record shape must contain two positive dimensions")
    count = shape[0] * shape[1]
    expected_words = mixed_v_payload_word_count(count)
    if (
        payload.dtype != torch.int32
        or tuple(payload.shape) != (expected_words,)
        or not payload.is_contiguous()
    ):
        raise ValueError(
            f"mixed-V payload must be contiguous int32 with shape ({expected_words},)"
        )
    used_tail_bits = (count * MIXED_V_RECORD_BITS) % PACKED_WORD_BITS
    unsigned = torch.bitwise_and(payload.to(torch.int64), 0xFFFF_FFFF)
    if used_tail_bits:
        padding_mask = 0xFFFF_FFFF ^ ((1 << used_tail_bits) - 1)
        if int(torch.bitwise_and(unsigned[-1], padding_mask).item()) != 0:
            raise ValueError("mixed-V payload has a non-zero padding bit")
    positions = torch.arange(count, dtype=torch.int64, device=payload.device)
    bit_offsets = positions * MIXED_V_RECORD_BITS
    word_indices = bit_offsets // PACKED_WORD_BITS
    shifts = bit_offsets % PACKED_WORD_BITS
    records = unsigned.index_select(0, word_indices) >> shifts
    crosses = shifts > PACKED_WORD_BITS - MIXED_V_RECORD_BITS
    records[crosses] |= (
        unsigned.index_select(0, word_indices[crosses] + 1)
        << (PACKED_WORD_BITS - shifts[crosses])
    )
    records &= (1 << MIXED_V_RECORD_BITS) - 1
    indices = (records & (MIXED_V_CODEBOOK_SIZE - 1)).reshape(shape).to(torch.int32)
    pairs = (records >> MIXED_V_INDEX_BITS).reshape(shape).to(torch.int32)
    _validate_record_fields(indices.contiguous(), pairs.contiguous())
    return indices.contiguous(), pairs.contiguous()


def decode_mixed_v_right_words(
    free_right_words: torch.Tensor,
    coded_payload: torch.Tensor,
    codebook_words: torch.Tensor,
    *,
    rank: int,
    in_features: int,
) -> torch.Tensor:
    """Expand one compact right factor to the canonical packed-word matrix."""

    free_rows = int(free_right_words.shape[0])
    words_per_row = packed_word_count(in_features)
    coded_shape = (rank - free_rows, words_per_row)
    indices, pair_ids = unpack_mixed_v_records(coded_payload, coded_shape)
    table = mixed_v_correction_pair_table(device=coded_payload.device)
    pairs = table.index_select(0, pair_ids.reshape(-1).long()).to(torch.int64)
    first = pairs & 0xFF
    second = (pairs >> 8) & 0xFF
    masks = (torch.ones_like(first) << first) | (torch.ones_like(second) << second)
    bases = codebook_words.index_select(0, indices.reshape(-1).long()).to(torch.int64)
    bases &= 0xFFFF_FFFF
    coded = (bases ^ masks).to(torch.int32).reshape(coded_shape)
    return torch.cat((free_right_words, coded), dim=0).contiguous()


@dataclass(frozen=True, slots=True)
class MixedVLayerState:
    """A packed layer whose V tail uses one 19-bit record per canonical word."""

    spec: QuantizedLinearSpec
    format: str
    free_rows: int
    left_words: torch.Tensor
    free_right_words: torch.Tensor
    coded_payload: torch.Tensor
    codebook_words: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    bias: torch.Tensor | None = None
    outlier_indices: torch.Tensor | None = None
    outlier_values: torch.Tensor | None = None
    outlier_scales: torch.Tensor | None = None
    patch_left: torch.Tensor | None = None
    patch_right: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.format != MIXED_V_FORMAT_VERSION:
            raise ValueError(f"unsupported mixed-V format: {self.format}")
        if self.free_rows <= 0 or self.free_rows >= self.spec.rank:
            raise ValueError("mixed-V free row count must be between zero and rank")
        words_per_row = packed_word_count(self.spec.in_features)
        if tuple(self.free_right_words.shape) != (self.free_rows, words_per_row):
            raise ValueError("mixed-V free right-word shape differs")
        if self.free_right_words.dtype != torch.int32 or not self.free_right_words.is_contiguous():
            raise ValueError("mixed-V free right words must be contiguous int32")
        if tuple(self.codebook_words.shape) != (MIXED_V_CODEBOOK_SIZE,):
            raise ValueError("mixed-V codebook must contain exactly 1024 words")
        if self.codebook_words.dtype != torch.int32 or not self.codebook_words.is_contiguous():
            raise ValueError("mixed-V codebook words must be contiguous int32")
        # This performs exact payload, correction-id, sign padding, scale, and sidecar validation.
        self.to_packed()

    @property
    def coded_rows(self) -> int:
        return self.spec.rank - self.free_rows

    @property
    def record_count(self) -> int:
        return self.coded_rows * packed_word_count(self.spec.in_features)

    def to_packed(self) -> PackedLayerState:
        right_words = decode_mixed_v_right_words(
            self.free_right_words,
            self.coded_payload,
            self.codebook_words,
            rank=self.spec.rank,
            in_features=self.spec.in_features,
        )
        return PackedLayerState(
            self.spec,
            PACKED_LAYOUT_VERSION,
            self.left_words,
            right_words,
            self.scale_pre,
            self.scale_mid,
            self.scale_post,
            self.bias,
            self.outlier_indices,
            self.outlier_values,
            self.outlier_scales,
            self.patch_left,
            self.patch_right,
        )

    def to_logical(self):  # type: ignore[no-untyped-def]
        return self.to_packed().to_logical()


def mixed_v_from_packed_words(
    packed: PackedLayerState,
    *,
    free_rows: int,
    codebook_words: torch.Tensor,
    codebook_indices: torch.Tensor,
    correction_pair_ids: torch.Tensor,
) -> MixedVLayerState:
    """Build a compact state after a factorizer has selected exact coded-tail records."""

    words_per_row = packed_word_count(packed.spec.in_features)
    expected = (packed.spec.rank - free_rows, words_per_row)
    if tuple(codebook_indices.shape) != expected or tuple(correction_pair_ids.shape) != expected:
        raise ValueError(f"mixed-V coded record matrices must have shape {expected}")
    state = MixedVLayerState(
        packed.spec,
        MIXED_V_FORMAT_VERSION,
        free_rows,
        packed.left_words,
        packed.right_words[:free_rows].contiguous(),
        pack_mixed_v_records(codebook_indices, correction_pair_ids),
        codebook_words.detach().cpu().to(torch.int32).contiguous(),
        packed.scale_pre,
        packed.scale_mid,
        packed.scale_post,
        packed.bias,
        packed.outlier_indices,
        packed.outlier_values,
        packed.outlier_scales,
        packed.patch_left,
        packed.patch_right,
    )
    if not torch.equal(state.to_packed().right_words, packed.right_words):
        raise ValueError("mixed-V records do not reconstruct the supplied packed right factor")
    return state


__all__ = [
    "MIXED_V_CODEBOOK_SIZE",
    "MIXED_V_CORRECTION_BITS",
    "MIXED_V_CORRECTION_PAIR_COUNT",
    "MIXED_V_FORMAT_VERSION",
    "MIXED_V_INDEX_BITS",
    "MIXED_V_RECORD_BITS",
    "MixedVLayerState",
    "decode_mixed_v_right_words",
    "mixed_v_correction_pair_table",
    "mixed_v_from_packed_words",
    "mixed_v_payload_word_count",
    "pack_mixed_v_records",
    "unpack_mixed_v_records",
]
