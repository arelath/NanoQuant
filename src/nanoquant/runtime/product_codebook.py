"""Compact mixed-free product-codebook factors for packed replay.

The research factorizer stores a free left factor and a right-factor prefix of
free sign words.  Every remaining right-factor word is represented by one
16-bit Cartesian-product index into two learned 256 x 16 sign tables.  Tall
projections may be fitted in transposed orientation; preparation decodes and
transposes once into the canonical packed NanoQuant layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nanoquant.runtime.backend import QuantizedLinearSpec
from nanoquant.runtime.packed import (
    PACKED_LAYOUT_VERSION,
    PACKED_WORD_BITS,
    PackedLayerState,
    pack_sign_matrix,
    packed_word_count,
    unpack_sign_matrix,
)

PRODUCT_CODEBOOK_FORMAT_VERSION = "product-codebook-free-k16-v1"
PRODUCT_CODEBOOK_INDEX_BITS = 16
PRODUCT_CODEBOOK_HALF_BITS = 16
PRODUCT_CODEBOOK_HALF_INDEX_BITS = 8
PRODUCT_CODEBOOK_TABLE_SIZE = 1 << PRODUCT_CODEBOOK_HALF_INDEX_BITS


def product_codebook_payload_word_count(record_count: int) -> int:
    if record_count <= 0:
        raise ValueError("product-codebook record count must be positive")
    return math.ceil(record_count * PRODUCT_CODEBOOK_INDEX_BITS / PACKED_WORD_BITS)


def pack_product_codebook_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack row-major unsigned 16-bit indices into minimal I32 words."""

    if (
        indices.ndim != 2
        or indices.numel() == 0
        or not indices.is_contiguous()
        or indices.dtype not in (torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("product-codebook indices must be a non-empty contiguous matrix")
    values = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    if bool(torch.any(values < 0)) or bool(torch.any(values >= (1 << 16))):
        raise ValueError("product-codebook index is outside the 16-bit range")
    if values.numel() % 2:
        values = torch.cat((values, torch.zeros(1, dtype=torch.int64)))
    packed = values[0::2] | (values[1::2] << 16)
    return packed.to(torch.int32).contiguous()


def unpack_product_codebook_indices(
    payload: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Decode the exact minimal 16-bit record stream."""

    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("product-codebook index shape must be positive")
    count = shape[0] * shape[1]
    expected = product_codebook_payload_word_count(count)
    if (
        payload.dtype != torch.int32
        or tuple(payload.shape) != (expected,)
        or not payload.is_contiguous()
    ):
        raise ValueError(
            f"product-codebook payload must be contiguous int32 with shape ({expected},)"
        )
    unsigned = payload.to(torch.int64) & 0xFFFF_FFFF
    if count % 2 and int(unsigned[-1].item()) >> 16:
        raise ValueError("product-codebook payload has a non-zero padding record")
    values = torch.empty(expected * 2, dtype=torch.int64, device=payload.device)
    values[0::2] = unsigned & 0xFFFF
    values[1::2] = unsigned >> 16
    return values[:count].reshape(shape).to(torch.int32).contiguous()


def pack_product_half_table(signs: torch.Tensor) -> torch.Tensor:
    """Pack a 256 x 16 sign table into exactly 256 I16 half-words."""

    expected = (PRODUCT_CODEBOOK_TABLE_SIZE, PRODUCT_CODEBOOK_HALF_BITS)
    if tuple(signs.shape) != expected or not signs.is_contiguous():
        raise ValueError(f"product-codebook half table must have shape {expected}")
    if not bool(torch.all((signs == 1) | (signs == -1))):
        raise ValueError("product-codebook half table contains a non-sign")
    negative = signs.detach().to(device="cpu") < 0
    shifts = torch.arange(PRODUCT_CODEBOOK_HALF_BITS, dtype=torch.int64)
    words = (negative.to(torch.int64) * (torch.ones_like(shifts) << shifts)).sum(dim=1)
    return words.to(torch.int16).contiguous()


def unpack_product_half_table(words: torch.Tensor) -> torch.Tensor:
    expected = (PRODUCT_CODEBOOK_TABLE_SIZE,)
    if words.dtype != torch.int16 or tuple(words.shape) != expected or not words.is_contiguous():
        raise ValueError("product-codebook half table must be contiguous int16[256]")
    unsigned = words.to(torch.int64) & 0xFFFF
    shifts = torch.arange(PRODUCT_CODEBOOK_HALF_BITS, dtype=torch.int64)
    bits = (unsigned[:, None] >> shifts[None, :]) & 1
    return (1 - 2 * bits).to(torch.float32).contiguous()


def decode_product_codebook_right_words(
    free_right_words: torch.Tensor,
    coded_payload: torch.Tensor,
    first_half_words: torch.Tensor,
    second_half_words: torch.Tensor,
    *,
    rank: int,
    columns: int,
) -> torch.Tensor:
    """Expand the factorization-space right factor into canonical sign words."""

    free_rows = int(free_right_words.shape[0])
    words_per_row = packed_word_count(columns)
    if (
        free_rows <= 0
        or free_rows >= rank
        or free_right_words.dtype != torch.int32
        or tuple(free_right_words.shape) != (free_rows, words_per_row)
        or not free_right_words.is_contiguous()
    ):
        raise ValueError("product-codebook free right-word prefix is invalid")
    coded_shape = (rank - free_rows, words_per_row)
    indices = unpack_product_codebook_indices(coded_payload, coded_shape).to(torch.int64)
    first = first_half_words.to(torch.int64) & 0xFFFF
    second = second_half_words.to(torch.int64) & 0xFFFF
    low = indices & 0xFF
    high = indices >> 8
    coded = first.index_select(0, low.reshape(-1))
    coded |= second.index_select(0, high.reshape(-1)) << 16
    coded = coded.to(torch.int32).reshape(coded_shape).contiguous()
    result = torch.cat((free_right_words, coded), dim=0).contiguous()
    # Canonical validation, including zero tail padding.
    unpack_sign_matrix(result, rank, columns)
    return result


@dataclass(frozen=True, slots=True)
class ProductCodebookLayerState:
    """One compact layer in factorization orientation."""

    spec: QuantizedLinearSpec
    format: str
    factorization_transposed: bool
    free_rows: int
    factor_left_words: torch.Tensor
    factor_right_free_words: torch.Tensor
    factor_right_coded_payload: torch.Tensor
    first_half_words: torch.Tensor
    second_half_words: torch.Tensor
    factor_scale_pre: torch.Tensor
    factor_scale_mid: torch.Tensor
    factor_scale_post: torch.Tensor
    outlier_indices: torch.Tensor | None = None
    outlier_values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.format != PRODUCT_CODEBOOK_FORMAT_VERSION:
            raise ValueError(f"unsupported product-codebook format: {self.format}")
        if self.spec.has_bias or self.spec.has_outlier_scales or self.spec.patch_rank:
            raise ValueError("product-codebook v1 does not carry optional bias, outlier scales, or patches")
        factor_out = self.spec.in_features if self.factorization_transposed else self.spec.out_features
        factor_in = self.spec.out_features if self.factorization_transposed else self.spec.in_features
        if self.free_rows <= 0 or self.free_rows >= self.spec.rank:
            raise ValueError("product-codebook free rows must be between zero and rank")
        expected_left = (factor_out, packed_word_count(self.spec.rank))
        if (
            self.factor_left_words.dtype != torch.int32
            or tuple(self.factor_left_words.shape) != expected_left
            or not self.factor_left_words.is_contiguous()
        ):
            raise ValueError("product-codebook factor-left words differ")
        unpack_sign_matrix(self.factor_left_words, factor_out, self.spec.rank)
        unpack_product_half_table(self.first_half_words)
        unpack_product_half_table(self.second_half_words)
        decode_product_codebook_right_words(
            self.factor_right_free_words,
            self.factor_right_coded_payload,
            self.first_half_words,
            self.second_half_words,
            rank=self.spec.rank,
            columns=factor_in,
        )
        expected_scales = (
            ("factor_scale_pre", self.factor_scale_pre, factor_in),
            ("factor_scale_mid", self.factor_scale_mid, self.spec.rank),
            ("factor_scale_post", self.factor_scale_post, factor_out),
        )
        for name, value, size in expected_scales:
            if (
                tuple(value.shape) != (size,)
                or not value.is_contiguous()
                or str(value.dtype).removeprefix("torch.") != self.spec.scale_dtype
                or not bool(torch.all(torch.isfinite(value)))
            ):
                raise ValueError(f"product-codebook {name} differs")
        if (self.outlier_indices is None) != (self.outlier_values is None):
            raise ValueError("product-codebook outlier tensors must be paired")
        if (self.outlier_indices is None) != (self.spec.outlier_count == 0):
            raise ValueError("product-codebook outlier presence differs")
        # Conversion validates physical outlier shapes, ordering, and zeroed pre-scales.
        self.to_packed()

    @property
    def factor_out_features(self) -> int:
        return self.spec.in_features if self.factorization_transposed else self.spec.out_features

    @property
    def factor_in_features(self) -> int:
        return self.spec.out_features if self.factorization_transposed else self.spec.in_features

    @property
    def coded_rows(self) -> int:
        return self.spec.rank - self.free_rows

    @property
    def record_count(self) -> int:
        return self.coded_rows * packed_word_count(self.factor_in_features)

    def to_packed(self) -> PackedLayerState:
        factor_right_words = decode_product_codebook_right_words(
            self.factor_right_free_words,
            self.factor_right_coded_payload,
            self.first_half_words,
            self.second_half_words,
            rank=self.spec.rank,
            columns=self.factor_in_features,
        )
        if self.factorization_transposed:
            factor_left = unpack_sign_matrix(
                self.factor_left_words,
                self.factor_out_features,
                self.spec.rank,
            )
            factor_right = unpack_sign_matrix(
                factor_right_words,
                self.spec.rank,
                self.factor_in_features,
            )
            left_words = pack_sign_matrix(factor_right.mT.contiguous())
            right_words = pack_sign_matrix(factor_left.mT.contiguous())
            scale_pre = self.factor_scale_post
            scale_post = self.factor_scale_pre
        else:
            left_words = self.factor_left_words
            right_words = factor_right_words
            scale_pre = self.factor_scale_pre
            scale_post = self.factor_scale_post
        return PackedLayerState(
            self.spec,
            PACKED_LAYOUT_VERSION,
            left_words,
            right_words,
            scale_pre,
            self.factor_scale_mid,
            scale_post,
            outlier_indices=self.outlier_indices,
            outlier_values=self.outlier_values,
        )

    def compact_logical_bits(self) -> int:
        index_bits = math.ceil(math.log2(self.spec.in_features)) * self.spec.outlier_count
        outlier_bits = self.spec.out_features * self.spec.outlier_count * 16
        return (
            self.factor_left_words.numel() * 32
            + self.factor_right_free_words.numel() * 32
            + self.record_count * PRODUCT_CODEBOOK_INDEX_BITS
            + 2 * PRODUCT_CODEBOOK_TABLE_SIZE * PRODUCT_CODEBOOK_HALF_BITS
            + 16  # persisted free-row count
            + (
                self.factor_scale_pre.numel()
                + self.factor_scale_mid.numel()
                + self.factor_scale_post.numel()
            )
            * 16
            + index_bits
            + outlier_bits
        )


__all__ = [
    "PRODUCT_CODEBOOK_FORMAT_VERSION",
    "PRODUCT_CODEBOOK_HALF_BITS",
    "PRODUCT_CODEBOOK_HALF_INDEX_BITS",
    "PRODUCT_CODEBOOK_INDEX_BITS",
    "PRODUCT_CODEBOOK_TABLE_SIZE",
    "ProductCodebookLayerState",
    "decode_product_codebook_right_words",
    "pack_product_codebook_indices",
    "pack_product_half_table",
    "product_codebook_payload_word_count",
    "unpack_product_codebook_indices",
    "unpack_product_half_table",
]
