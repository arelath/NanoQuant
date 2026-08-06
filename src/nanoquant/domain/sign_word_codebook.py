"""Analysis-grade sign-word-codebook factorization.

The production format stores one bit per sign.  This module supplies a bounded
research implementation of the fixed-width codebook alternative without
changing any persisted or runtime contract.  A 32-sign word is represented as
the Cartesian product of two independently fitted 16-sign half-codebooks.  The
two half indices pack into one fixed-width word index, so the decoded set is a
valid ``2**index_bits``-entry 32-sign codebook while assignment remains small
enough for a real Gemma matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .factorization import SCHEDULES, ADMMResult, ADMMTracePoint

CORRECTED_ASSIGNMENT_CANDIDATES = 16


@dataclass(frozen=True, slots=True)
class SignWordCodebookCost:
    """Exact fixed-width storage charged by the analysis probe."""

    index_bits: int
    scale_bits: int
    codebook_bits: int
    word_count: int

    @property
    def total(self) -> int:
        return self.index_bits + self.scale_bits + self.codebook_bits


@dataclass(frozen=True, slots=True)
class ProductSignCodebook:
    """Two half-word tables whose Cartesian product forms a 32-sign table."""

    index_bits: int
    first: torch.Tensor
    second: torch.Tensor

    def __post_init__(self) -> None:
        if self.index_bits <= 0 or self.index_bits % 2:
            raise ValueError("product codebook index width must be positive and even")
        expected = (1 << (self.index_bits // 2), 16)
        if tuple(self.first.shape) != expected or tuple(self.second.shape) != expected:
            raise ValueError(f"half-codebook shapes must both be {expected}")
        if self.first.device != self.second.device:
            raise ValueError("half-codebooks must share one device")
        for table in (self.first, self.second):
            if not torch.all((table == 1) | (table == -1)):
                raise ValueError("codebook entries must be signs")

    @property
    def entry_count(self) -> int:
        return 1 << self.index_bits


@dataclass(frozen=True, slots=True)
class FullSignCodebook:
    """An unconstrained ``2**k`` by 32 fitted sign table."""

    index_bits: int
    entries: torch.Tensor

    def __post_init__(self) -> None:
        if self.index_bits <= 0:
            raise ValueError("full codebook index width must be positive")
        expected = (1 << self.index_bits, 32)
        if tuple(self.entries.shape) != expected:
            raise ValueError(f"full codebook shape must be {expected}")
        if not torch.all((self.entries == 1) | (self.entries == -1)):
            raise ValueError("codebook entries must be signs")

    @property
    def entry_count(self) -> int:
        return 1 << self.index_bits


@dataclass(frozen=True, slots=True)
class BankedFullSignCodebook:
    """Full sign tables selected implicitly by word position or component row."""

    index_bits: int
    entries: torch.Tensor
    bank_axis: str = "word"

    def __post_init__(self) -> None:
        if self.index_bits <= 0:
            raise ValueError("banked codebook index width must be positive")
        if self.entries.ndim != 3:
            raise ValueError("banked codebook entries must have three dimensions")
        expected_tail = (1 << self.index_bits, 32)
        if self.entries.shape[0] <= 1 or tuple(self.entries.shape[1:]) != expected_tail:
            raise ValueError(
                f"banked codebook must contain multiple {expected_tail} tables"
            )
        if self.entries.shape[0] & (self.entries.shape[0] - 1):
            raise ValueError("banked codebook count must be a power of two")
        if self.bank_axis not in {"word", "row"}:
            raise ValueError("codebook bank axis must be 'word' or 'row'")
        if not torch.all((self.entries == 1) | (self.entries == -1)):
            raise ValueError("codebook entries must be signs")

    @property
    def entry_count(self) -> int:
        return self.entries.shape[0] * (1 << self.index_bits)

    @property
    def bank_count(self) -> int:
        return self.entries.shape[0]


SignCodebook = ProductSignCodebook | FullSignCodebook | BankedFullSignCodebook


@dataclass(frozen=True, slots=True)
class SignWordCodebookADMMResult:
    """Constrained factors plus their exact codebook representation."""

    factors: ADMMResult
    left_codebook: SignCodebook | None
    right_codebook: SignCodebook | None
    left_indices: torch.Tensor | None
    right_indices: torch.Tensor | None
    left_flip_positions: torch.Tensor | None
    right_flip_positions: torch.Tensor | None
    left_free_rows: int
    right_free_rows: int


def sign_word_codebook_bit_cost(
    out_features: int,
    in_features: int,
    rank: int,
    *,
    index_width: int,
    scale_width: int = 16,
    word_width: int = 32,
    codebook_count: int = 2,
) -> SignWordCodebookCost:
    """Charge fixed-width indices, all three scales, and full decode tables."""

    if min(out_features, in_features, rank) <= 0:
        raise ValueError("codebook cost dimensions and rank must be positive")
    if index_width <= 0 or scale_width < 0 or word_width <= 0 or codebook_count <= 0:
        raise ValueError("codebook cost widths/count are invalid")
    left_words = out_features * math.ceil(rank / word_width)
    right_words = rank * math.ceil(in_features / word_width)
    words = left_words + right_words
    return SignWordCodebookCost(
        index_bits=words * index_width,
        scale_bits=scale_width * (out_features + in_features + rank),
        codebook_bits=codebook_count * (1 << index_width) * word_width,
        word_count=words,
    )


def maximum_codebook_rank_for_budget(
    out_features: int,
    in_features: int,
    target_bits: int,
    *,
    index_width: int,
    rank_multiple: int = 32,
    scale_width: int = 16,
) -> int:
    """Return the largest aligned codebook rank within ``target_bits``."""

    if target_bits <= 0 or rank_multiple <= 0:
        raise ValueError("codebook rank budget and multiple must be positive")
    rank = rank_multiple
    accepted = 0
    while True:
        cost = sign_word_codebook_bit_cost(
            out_features,
            in_features,
            rank,
            index_width=index_width,
            scale_width=scale_width,
        )
        if cost.total > target_bits:
            break
        accepted = rank
        rank += rank_multiple
    if accepted <= 0:
        raise ValueError("target budget cannot fund one aligned codebook rank")
    return accepted


def asymmetric_sign_word_codebook_bit_cost(
    out_features: int,
    in_features: int,
    rank: int,
    *,
    left_index_width: int | None,
    right_index_width: int | None,
    scale_width: int = 16,
    word_width: int = 32,
) -> SignWordCodebookCost:
    """Charge independently compressed factors; ``None`` stores free words."""

    if min(out_features, in_features, rank) <= 0:
        raise ValueError("codebook cost dimensions and rank must be positive")
    if scale_width < 0 or word_width <= 0:
        raise ValueError("codebook cost widths are invalid")
    for width in (left_index_width, right_index_width):
        if width is not None and width <= 0:
            raise ValueError("codebook index widths must be positive")
    left_words = out_features * math.ceil(rank / word_width)
    right_words = rank * math.ceil(in_features / word_width)
    left_width = word_width if left_index_width is None else left_index_width
    right_width = word_width if right_index_width is None else right_index_width
    table_bits = sum(
        (1 << width) * word_width
        for width in (left_index_width, right_index_width)
        if width is not None
    )
    return SignWordCodebookCost(
        index_bits=left_words * left_width + right_words * right_width,
        scale_bits=scale_width * (out_features + in_features + rank),
        codebook_bits=table_bits,
        word_count=left_words + right_words,
    )


def maximum_asymmetric_codebook_rank_for_budget(
    out_features: int,
    in_features: int,
    target_bits: int,
    *,
    left_index_width: int | None,
    right_index_width: int | None,
    rank_multiple: int = 32,
    scale_width: int = 16,
) -> int:
    """Return the largest aligned asymmetric-codebook rank in the budget."""

    if target_bits <= 0 or rank_multiple <= 0:
        raise ValueError("codebook rank budget and multiple must be positive")
    rank = rank_multiple
    accepted = 0
    while True:
        cost = asymmetric_sign_word_codebook_bit_cost(
            out_features,
            in_features,
            rank,
            left_index_width=left_index_width,
            right_index_width=right_index_width,
            scale_width=scale_width,
        )
        if cost.total > target_bits:
            break
        accepted = rank
        rank += rank_multiple
    if accepted <= 0:
        raise ValueError("target budget cannot fund one aligned asymmetric rank")
    return accepted


def corrected_asymmetric_codebook_bit_cost(
    out_features: int,
    in_features: int,
    rank: int,
    *,
    left_index_width: int | None,
    right_index_width: int | None,
    left_flip_bits: int = 0,
    right_flip_bits: int = 0,
    scale_width: int = 16,
    word_width: int = 32,
    right_codebook_count: int = 1,
) -> SignWordCodebookCost:
    """Charge asymmetric codebooks plus fixed-width correction positions."""

    base = asymmetric_sign_word_codebook_bit_cost(
        out_features,
        in_features,
        rank,
        left_index_width=left_index_width,
        right_index_width=right_index_width,
        scale_width=scale_width,
        word_width=word_width,
    )
    if (
        left_flip_bits < 0
        or right_flip_bits < 0
        or right_codebook_count <= 0
    ):
        raise ValueError("correction widths must not be negative")
    if (left_flip_bits and left_index_width is None) or (
        right_flip_bits and right_index_width is None
    ):
        raise ValueError("correction streams require a codebook on that factor")
    left_words = out_features * math.ceil(rank / word_width)
    right_words = rank * math.ceil(in_features / word_width)
    return SignWordCodebookCost(
        index_bits=(
            base.index_bits
            + left_words * left_flip_bits
            + right_words * right_flip_bits
        ),
        scale_bits=base.scale_bits,
        codebook_bits=(
            base.codebook_bits
            + (
                (right_codebook_count - 1)
                * (1 << right_index_width)
                * word_width
                if right_index_width is not None
                else 0
            )
        ),
        word_count=base.word_count,
    )


def maximum_corrected_asymmetric_rank_for_budget(
    out_features: int,
    in_features: int,
    target_bits: int,
    *,
    left_index_width: int | None,
    right_index_width: int | None,
    left_flip_bits: int = 0,
    right_flip_bits: int = 0,
    rank_multiple: int = 32,
    scale_width: int = 16,
) -> int:
    """Return the largest corrected-codebook rank within ``target_bits``."""

    if target_bits <= 0 or rank_multiple <= 0:
        raise ValueError("codebook rank budget and multiple must be positive")
    rank = rank_multiple
    accepted = 0
    while True:
        cost = corrected_asymmetric_codebook_bit_cost(
            out_features,
            in_features,
            rank,
            left_index_width=left_index_width,
            right_index_width=right_index_width,
            left_flip_bits=left_flip_bits,
            right_flip_bits=right_flip_bits,
            scale_width=scale_width,
        )
        if cost.total > target_bits:
            break
        accepted = rank
        rank += rank_multiple
    if accepted <= 0:
        raise ValueError("target budget cannot fund one corrected aligned rank")
    return accepted


def mixed_right_corrected_codebook_bit_cost(
    out_features: int,
    in_features: int,
    rank: int,
    *,
    right_free_rows: int,
    right_index_width: int,
    right_flip_bits: int,
    scale_width: int = 16,
    word_width: int = 32,
    free_row_count_bits: int = 16,
    right_codebook_count: int = 1,
    right_corrected_rows: int | None = None,
) -> SignWordCodebookCost:
    """Charge free U, a free prefix of V rows, and corrected coded V rows."""

    if not 0 <= right_free_rows < rank:
        raise ValueError("right free rows must leave at least one coded row")
    if (
        right_index_width <= 0
        or right_flip_bits < 0
        or free_row_count_bits < 0
        or right_codebook_count <= 0
    ):
        raise ValueError("mixed right-code widths are invalid")
    left_words = out_features * math.ceil(rank / word_width)
    right_words_per_row = math.ceil(in_features / word_width)
    coded_rows = rank - right_free_rows
    corrected_rows = (
        coded_rows if right_corrected_rows is None else right_corrected_rows
    )
    if not 0 <= corrected_rows <= coded_rows:
        raise ValueError("corrected rows must lie within the coded-row suffix")
    right_words = rank * right_words_per_row
    payload_bits = (
        left_words * word_width
        + right_free_rows * right_words_per_row * word_width
        + coded_rows * right_words_per_row * right_index_width
        + corrected_rows * right_words_per_row * right_flip_bits
    )
    return SignWordCodebookCost(
        index_bits=payload_bits,
        scale_bits=scale_width * (out_features + in_features + rank),
        codebook_bits=(
            right_codebook_count * (1 << right_index_width) * word_width
            + free_row_count_bits
        ),
        word_count=left_words + right_words,
    )


def maximum_mixed_right_free_rows_for_budget(
    out_features: int,
    in_features: int,
    rank: int,
    target_bits: int,
    *,
    right_index_width: int,
    right_flip_bits: int,
    free_row_multiple: int = 32,
    scale_width: int = 16,
    right_codebook_count: int = 1,
) -> int:
    """Return the largest aligned free V prefix that fits the target budget."""

    if target_bits <= 0 or free_row_multiple <= 0:
        raise ValueError("mixed right-code budget and alignment must be positive")
    accepted = 0
    for free_rows in range(0, rank, free_row_multiple):
        cost = mixed_right_corrected_codebook_bit_cost(
            out_features,
            in_features,
            rank,
            right_free_rows=free_rows,
            right_index_width=right_index_width,
            right_flip_bits=right_flip_bits,
            scale_width=scale_width,
            right_codebook_count=right_codebook_count,
        )
        if cost.total > target_bits:
            break
        accepted = free_rows
    return accepted


def mixed_right_product_codebook_bit_cost(
    out_features: int,
    in_features: int,
    rank: int,
    *,
    right_free_rows: int,
    right_index_width: int,
    scale_width: int = 16,
    word_width: int = 32,
    free_row_count_bits: int = 16,
) -> SignWordCodebookCost:
    """Charge free U and a mixed V encoded by two learned half-word tables."""

    if not 0 <= right_free_rows < rank:
        raise ValueError("right free rows must leave at least one coded row")
    if (
        right_index_width <= 0
        or right_index_width % 2
        or scale_width < 0
        or word_width <= 0
        or word_width % 2
        or free_row_count_bits < 0
    ):
        raise ValueError("mixed product-code widths are invalid")
    left_words = out_features * math.ceil(rank / word_width)
    right_words_per_row = math.ceil(in_features / word_width)
    coded_rows = rank - right_free_rows
    right_words = rank * right_words_per_row
    payload_bits = (
        left_words * word_width
        + right_free_rows * right_words_per_row * word_width
        + coded_rows * right_words_per_row * right_index_width
    )
    half_width = word_width // 2
    table_bits = 2 * (1 << (right_index_width // 2)) * half_width
    return SignWordCodebookCost(
        index_bits=payload_bits,
        scale_bits=scale_width * (out_features + in_features + rank),
        codebook_bits=table_bits + free_row_count_bits,
        word_count=left_words + right_words,
    )


def maximum_mixed_right_product_free_rows_for_budget(
    out_features: int,
    in_features: int,
    rank: int,
    target_bits: int,
    *,
    right_index_width: int,
    free_row_multiple: int = 32,
    scale_width: int = 16,
) -> int:
    """Return the largest aligned free prefix for a compact product code."""

    if target_bits <= 0 or free_row_multiple <= 0:
        raise ValueError("mixed product-code budget and alignment must be positive")
    accepted = 0
    for free_rows in range(0, rank, free_row_multiple):
        cost = mixed_right_product_codebook_bit_cost(
            out_features,
            in_features,
            rank,
            right_free_rows=free_rows,
            right_index_width=right_index_width,
            scale_width=scale_width,
        )
        if cost.total > target_bits:
            break
        accepted = free_rows
    return accepted


def decode_product_codebook(
    indices: torch.Tensor,
    codebook: ProductSignCodebook,
    columns: int,
) -> torch.Tensor:
    """Decode row-major fixed-width word indices to a sign matrix."""

    if indices.ndim != 2 or columns <= 0:
        raise ValueError("codebook indices must be a matrix and columns positive")
    expected_words = math.ceil(columns / 32)
    if indices.shape[1] != expected_words:
        raise ValueError("codebook index word count does not match columns")
    half_bits = codebook.index_bits // 2
    mask = (1 << half_bits) - 1
    values = indices.to(dtype=torch.int64)
    first = codebook.first[values.bitwise_and(mask)]
    second = codebook.second[values.bitwise_right_shift(half_bits)]
    decoded = torch.cat((first, second), dim=-1).reshape(indices.shape[0], expected_words * 32)
    return decoded[:, :columns].contiguous()


def decode_sign_codebook(
    indices: torch.Tensor,
    codebook: SignCodebook,
    columns: int,
) -> torch.Tensor:
    """Decode either supported fixed-width sign-word table."""

    if isinstance(codebook, ProductSignCodebook):
        return decode_product_codebook(indices, codebook, columns)
    if indices.ndim != 2 or columns <= 0:
        raise ValueError("codebook indices must be a matrix and columns positive")
    expected_words = math.ceil(columns / 32)
    if indices.shape[1] != expected_words:
        raise ValueError("codebook index word count does not match columns")
    if isinstance(codebook, BankedFullSignCodebook):
        banks = (
            _word_bank_indices(
                expected_words,
                codebook.bank_count,
                indices.device,
            ).reshape(1, -1)
            if codebook.bank_axis == "word"
            else _word_bank_indices(
                indices.shape[0],
                codebook.bank_count,
                indices.device,
            ).reshape(-1, 1)
        )
        decoded = codebook.entries[
            banks,
            indices.to(torch.int64),
        ].reshape(indices.shape[0], expected_words * 32)
    else:
        decoded = codebook.entries[indices.to(torch.int64)].reshape(
            indices.shape[0],
            expected_words * 32,
        )
    return decoded[:, :columns].contiguous()


def _word_bank_indices(
    words: int,
    bank_count: int,
    device: torch.device,
) -> torch.Tensor:
    if words <= 0 or bank_count <= 0 or bank_count > words:
        raise ValueError("word-bank dimensions are invalid")
    return torch.div(
        torch.arange(words, device=device) * bank_count,
        words,
        rounding_mode="floor",
    )


def _sign(value: torch.Tensor) -> torch.Tensor:
    return (value >= 0).to(dtype=value.dtype).mul_(2).sub_(1)


def _power_iteration(
    value: torch.Tensor,
    iterations: int,
    generator: torch.Generator,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vector = torch.randn(
        value.shape[1],
        dtype=value.dtype,
        device=value.device,
        generator=generator,
    )
    vector = vector / vector.norm().clamp_min(epsilon)
    for _ in range(iterations):
        left = value @ vector
        left = left / left.norm().clamp_min(epsilon)
        vector = value.mT @ left
        vector = vector / vector.norm().clamp_min(epsilon)
    unnormalized = value @ vector
    singular = unnormalized.norm().clamp_min(epsilon)
    return unnormalized / singular, singular, vector


def _rank_one_magnitudes(
    value: torch.Tensor,
    iterations: int,
    generator: torch.Generator,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, singular, right = _power_iteration(
        value.abs(),
        iterations,
        generator,
        epsilon,
    )
    # Perron vectors are defined only up to a joint sign.  Absolute values
    # select the non-negative representative required by sign decoding.
    return (left * singular).abs(), right.abs()


def _random_codebook(
    index_bits: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    mode: str,
    bank_count: int = 1,
    bank_axis: str = "word",
) -> SignCodebook:
    if bank_count <= 0 or bank_count & (bank_count - 1):
        raise ValueError("codebook bank count must be a positive power of two")
    if mode == "full":
        shape = (
            (1 << index_bits, 32)
            if bank_count == 1
            else (bank_count, 1 << index_bits, 32)
        )
        entries = (
            torch.randint(
                0,
                2,
                shape,
                device=device,
                generator=generator,
                dtype=torch.int8,
            )
            .to(dtype)
            .mul_(2)
            .sub_(1)
        )
        return (
            FullSignCodebook(index_bits, entries)
            if bank_count == 1
            else BankedFullSignCodebook(index_bits, entries, bank_axis)
        )
    if bank_count != 1:
        raise ValueError("only full codebooks support word banks")
    if mode != "product":
        raise ValueError(f"unsupported codebook mode: {mode}")
    half_entries = 1 << (index_bits // 2)

    def table() -> torch.Tensor:
        return (
            torch.randint(
                0,
                2,
                (half_entries, 16),
                device=device,
                generator=generator,
                dtype=torch.int8,
            )
            .to(dtype)
            .mul_(2)
            .sub_(1)
        )

    return ProductSignCodebook(index_bits, table(), table())


def _assign_half_words(
    values: torch.Tensor,
    table: torch.Tensor,
    *,
    update: bool,
    batch_words: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or values.shape[1] != table.shape[1]:
        raise ValueError("word values and codebook entry widths must match")
    if batch_words <= 0:
        raise ValueError("assignment batch size must be positive")
    assignments = torch.empty(values.shape[0], dtype=torch.int64, device=values.device)
    sums = torch.zeros_like(table, dtype=torch.float32) if update else None
    counts = torch.zeros(table.shape[0], dtype=torch.int64, device=values.device) if update else None
    table32 = table.float()
    for start in range(0, values.shape[0], batch_words):
        stop = min(values.shape[0], start + batch_words)
        batch = values[start:stop].float()
        selected = (batch @ table32.mT).argmax(dim=1)
        assignments[start:stop] = selected
        if sums is not None and counts is not None:
            sums.index_add_(0, selected, batch)
            counts.index_add_(
                0,
                selected,
                torch.ones_like(selected, dtype=torch.int64),
            )
    if sums is None or counts is None:
        return assignments, table
    replacement = _sign(sums).to(table.dtype)
    populated = counts > 0
    updated = table.clone()
    updated[populated] = replacement[populated]
    return assignments, updated


def _assign_product_words(
    weighted_value: torch.Tensor,
    codebook: ProductSignCodebook,
    *,
    update: bool,
    batch_words: int,
) -> tuple[torch.Tensor, torch.Tensor, ProductSignCodebook]:
    rows, columns = weighted_value.shape
    words = math.ceil(columns / 32)
    padded_columns = words * 32
    if padded_columns != columns:
        padded = torch.zeros(
            (rows, padded_columns),
            dtype=weighted_value.dtype,
            device=weighted_value.device,
        )
        padded[:, :columns] = weighted_value
    else:
        padded = weighted_value
    word_values = padded.reshape(rows, words, 2, 16)
    first_values = word_values[:, :, 0, :].reshape(-1, 16).contiguous()
    second_values = word_values[:, :, 1, :].reshape(-1, 16).contiguous()
    first_indices, first_table = _assign_half_words(
        first_values,
        codebook.first,
        update=update,
        batch_words=batch_words,
    )
    second_indices, second_table = _assign_half_words(
        second_values,
        codebook.second,
        update=update,
        batch_words=batch_words,
    )
    updated = ProductSignCodebook(codebook.index_bits, first_table, second_table)
    if update:
        # The centroids moved, so persist assignments to the updated table.
        first_indices, _ = _assign_half_words(
            first_values,
            updated.first,
            update=False,
            batch_words=batch_words,
        )
        second_indices, _ = _assign_half_words(
            second_values,
            updated.second,
            update=False,
            batch_words=batch_words,
        )
    half_bits = codebook.index_bits // 2
    indices = (
        first_indices.bitwise_or(second_indices.bitwise_left_shift(half_bits))
        .reshape(rows, words)
        .to(torch.int32)
    )
    decoded = decode_product_codebook(indices, updated, padded_columns)[:, :columns]
    return decoded, indices, updated


def _assign_full_words(
    weighted_value: torch.Tensor,
    codebook: FullSignCodebook,
    *,
    update: bool,
    batch_words: int,
) -> tuple[torch.Tensor, torch.Tensor, FullSignCodebook]:
    rows, columns = weighted_value.shape
    words = math.ceil(columns / 32)
    padded_columns = words * 32
    if padded_columns != columns:
        padded = torch.zeros(
            (rows, padded_columns),
            dtype=weighted_value.dtype,
            device=weighted_value.device,
        )
        padded[:, :columns] = weighted_value
    else:
        padded = weighted_value
    word_values = padded.reshape(-1, 32).contiguous()
    assignments, entries = _assign_half_words(
        word_values,
        codebook.entries,
        update=update,
        batch_words=batch_words,
    )
    updated = FullSignCodebook(codebook.index_bits, entries)
    if update:
        assignments, _ = _assign_half_words(
            word_values,
            updated.entries,
            update=False,
            batch_words=batch_words,
        )
    indices = assignments.reshape(rows, words).to(torch.int32)
    decoded = decode_sign_codebook(indices, updated, padded_columns)[:, :columns]
    return decoded, indices, updated


def _assign_corrected_flat_words(
    values: torch.Tensor,
    table: torch.Tensor,
    *,
    flips_per_word: int,
    update: bool,
    batch_words: int,
    candidate_count: int = CORRECTED_ASSIGNMENT_CANDIDATES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jointly choose a nearby codeword and its best fixed-count corrections."""

    if values.ndim != 2 or values.shape[1] != 32:
        raise ValueError("corrected assignment requires flat 32-sign words")
    if not 1 <= flips_per_word <= 3:
        raise ValueError("corrected assignment supports one to three flips")
    candidates = min(candidate_count, table.shape[0])
    assignments = torch.empty(
        values.shape[0],
        dtype=torch.int64,
        device=values.device,
    )
    flip_positions = torch.empty(
        (values.shape[0], flips_per_word),
        dtype=torch.int8,
        device=values.device,
    )
    sums = torch.zeros_like(table, dtype=torch.float32) if update else None
    counts = (
        torch.zeros(table.shape[0], dtype=torch.int64, device=values.device)
        if update
        else None
    )
    table32 = table.float()
    for start in range(0, values.shape[0], batch_words):
        stop = min(values.shape[0], start + batch_words)
        batch = values[start:stop].float()
        scores = batch @ table32.mT
        candidate_scores, candidate_indices = scores.topk(candidates, dim=1)
        candidate_entries = table32[candidate_indices]
        gains = -2 * batch.unsqueeze(1) * candidate_entries
        flip_gains, candidate_flips = gains.topk(flips_per_word, dim=2)
        corrected_scores = candidate_scores + flip_gains.sum(dim=2)
        choices = corrected_scores.argmax(dim=1)
        rows = torch.arange(batch.shape[0], device=values.device)
        selected = candidate_indices[rows, choices]
        selected_flips = candidate_flips[rows, choices]
        assignments[start:stop] = selected
        flip_positions[start:stop] = selected_flips.to(torch.int8)
        if sums is not None and counts is not None:
            transformed = batch.clone()
            for offset in range(flips_per_word):
                transformed[rows, selected_flips[:, offset]] *= -1
            sums.index_add_(0, selected, transformed)
            counts.index_add_(
                0,
                selected,
                torch.ones_like(selected, dtype=torch.int64),
            )
    if sums is None or counts is None:
        return assignments, flip_positions, table
    updated = table.clone()
    populated = counts > 0
    updated[populated] = _sign(sums[populated]).to(table.dtype)
    final_assignments, final_positions, _ = _assign_corrected_flat_words(
        values,
        updated,
        flips_per_word=flips_per_word,
        update=False,
        batch_words=batch_words,
        candidate_count=candidate_count,
    )
    return final_assignments, final_positions, updated


def _assign_corrected_full_words(
    weighted_value: torch.Tensor,
    codebook: FullSignCodebook,
    *,
    flips_per_word: int,
    update: bool,
    batch_words: int,
    candidate_count: int = CORRECTED_ASSIGNMENT_CANDIDATES,
) -> tuple[torch.Tensor, torch.Tensor, FullSignCodebook, torch.Tensor]:
    rows, columns = weighted_value.shape
    words = math.ceil(columns / 32)
    padded_columns = words * 32
    padded = torch.zeros(
        (rows, padded_columns),
        dtype=weighted_value.dtype,
        device=weighted_value.device,
    )
    padded[:, :columns] = weighted_value
    word_values = padded.reshape(-1, 32).contiguous()
    assignments, positions, entries = _assign_corrected_flat_words(
        word_values,
        codebook.entries,
        flips_per_word=flips_per_word,
        update=update,
        batch_words=batch_words,
        candidate_count=candidate_count,
    )
    updated = FullSignCodebook(codebook.index_bits, entries)
    indices = assignments.reshape(rows, words).to(torch.int32)
    shaped_positions = positions.reshape(rows, words, flips_per_word)
    if flips_per_word == 1:
        shaped_positions = shaped_positions.squeeze(-1)
    decoded = decode_sign_codebook(indices, updated, padded_columns)[:, :columns]
    corrected = apply_word_flips(decoded, shaped_positions)
    return corrected, indices, updated, shaped_positions


def _assign_banked_full_words(
    weighted_value: torch.Tensor,
    codebook: BankedFullSignCodebook,
    *,
    flips_per_word: int,
    update: bool,
    batch_words: int,
    candidate_count: int = CORRECTED_ASSIGNMENT_CANDIDATES,
    corrected_bank_count: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    BankedFullSignCodebook,
    torch.Tensor | None,
]:
    """Assign each word against the table implied by its column position."""

    rows, columns = weighted_value.shape
    words = math.ceil(columns / 32)
    padded_columns = words * 32
    padded = torch.zeros(
        (rows, padded_columns),
        dtype=weighted_value.dtype,
        device=weighted_value.device,
    )
    padded[:, :columns] = weighted_value
    word_values = padded.reshape(rows, words, 32)
    indices = torch.empty((rows, words), dtype=torch.int32, device=weighted_value.device)
    corrected_banks = (
        codebook.bank_count
        if corrected_bank_count is None
        else corrected_bank_count
    )
    if not 1 <= corrected_banks <= codebook.bank_count:
        raise ValueError("corrected bank count is outside the codebook")
    if (
        codebook.bank_axis != "row"
        and corrected_banks not in {0, codebook.bank_count}
    ):
        raise ValueError("partial correction banking requires row banks")
    corrected_rows = (
        math.ceil(rows * corrected_banks / codebook.bank_count)
        if codebook.bank_axis == "row"
        else rows
    )
    positions = (
        torch.empty(
            (corrected_rows, words, flips_per_word),
            dtype=torch.int8,
            device=weighted_value.device,
        )
        if flips_per_word
        else None
    )
    updated_entries = []
    bank_ids = _word_bank_indices(
        words if codebook.bank_axis == "word" else rows,
        codebook.bank_count,
        weighted_value.device,
    )
    for bank in range(codebook.bank_count):
        bank_mask = bank_ids == bank
        values = (
            word_values[:, bank_mask, :]
            if codebook.bank_axis == "word"
            else word_values[bank_mask, :, :]
        ).reshape(-1, 32).contiguous()
        bank_flips = flips_per_word if bank < corrected_banks else 0
        if bank_flips:
            assigned, corrected_positions, entries = _assign_corrected_flat_words(
                values,
                codebook.entries[bank],
                flips_per_word=bank_flips,
                update=update,
                batch_words=batch_words,
                candidate_count=candidate_count,
            )
            assert positions is not None
            if codebook.bank_axis == "word":
                positions[:, bank_mask, :] = corrected_positions.reshape(
                    rows,
                    int(bank_mask.sum()),
                    bank_flips,
                )
            else:
                positions[bank_mask[:corrected_rows], :, :] = (
                    corrected_positions.reshape(
                    int(bank_mask.sum()),
                    words,
                    bank_flips,
                )
                )
        else:
            assigned, entries = _assign_half_words(
                values,
                codebook.entries[bank],
                update=update,
                batch_words=batch_words,
            )
            if update:
                assigned, _ = _assign_half_words(
                    values,
                    entries,
                    update=False,
                    batch_words=batch_words,
                )
        if codebook.bank_axis == "word":
            indices[:, bank_mask] = assigned.reshape(
                rows,
                int(bank_mask.sum()),
            ).to(torch.int32)
        else:
            indices[bank_mask, :] = assigned.reshape(
                int(bank_mask.sum()),
                words,
            ).to(torch.int32)
        updated_entries.append(entries)
    updated = BankedFullSignCodebook(
        codebook.index_bits,
        torch.stack(updated_entries),
        codebook.bank_axis,
    )
    shaped_positions = positions
    if shaped_positions is not None and flips_per_word == 1:
        shaped_positions = shaped_positions.squeeze(-1)
    decoded = decode_sign_codebook(indices, updated, padded_columns)[:, :columns]
    corrected = (
        torch.cat(
            (
                apply_word_flips(decoded[:corrected_rows], shaped_positions),
                decoded[corrected_rows:],
            )
        )
        if shaped_positions is not None
        else decoded
    )
    return corrected, indices, updated, shaped_positions


def apply_word_flips(
    decoded: torch.Tensor,
    flip_positions: torch.Tensor,
) -> torch.Tensor:
    """Flip indexed signs in every row-major 32-sign word."""

    rows, columns = decoded.shape
    words = math.ceil(columns / 32)
    if flip_positions.ndim not in {2, 3}:
        raise ValueError("flip positions must have one or more positions per word")
    if tuple(flip_positions.shape[:2]) != (rows, words):
        raise ValueError("flip-position shape does not match decoded words")
    positions_per_word = 1 if flip_positions.ndim == 2 else flip_positions.shape[2]
    if positions_per_word <= 0:
        raise ValueError("flip positions must have one or more positions per word")
    if torch.any((flip_positions < 0) | (flip_positions >= 32)):
        raise ValueError("flip positions must lie in [0, 32)")
    if flip_positions.ndim == 3:
        ordered = flip_positions.sort(dim=2).values
        if torch.any(ordered[:, :, 1:] == ordered[:, :, :-1]):
            raise ValueError("flip positions within a word must be distinct")
    padded_columns = words * 32
    if padded_columns != columns:
        padded = torch.ones(
            (rows, padded_columns),
            dtype=decoded.dtype,
            device=decoded.device,
        )
        padded[:, :columns] = decoded
    else:
        padded = decoded.clone()
    word_values = padded.reshape(rows * words, 32)
    positions = flip_positions.reshape(rows * words, positions_per_word).to(
        torch.int64
    )
    row_indices = torch.arange(word_values.shape[0], device=decoded.device)
    for offset in range(positions_per_word):
        word_values[row_indices, positions[:, offset]] *= -1
    return word_values.reshape(rows, padded_columns)[:, :columns].contiguous()


def apply_single_word_flip(
    decoded: torch.Tensor,
    flip_positions: torch.Tensor,
) -> torch.Tensor:
    """Compatibility wrapper for the one-correction representation."""

    return apply_word_flips(decoded, flip_positions)


def _apply_best_word_flips(
    weighted_value: torch.Tensor,
    decoded: torch.Tensor,
    flips_per_word: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = decoded.shape
    words = math.ceil(columns / 32)
    padded_columns = words * 32
    padded_value = torch.zeros(
        (rows, padded_columns),
        dtype=weighted_value.dtype,
        device=weighted_value.device,
    )
    padded_value[:, :columns] = weighted_value
    padded_decoded = torch.ones(
        (rows, padded_columns),
        dtype=decoded.dtype,
        device=decoded.device,
    )
    padded_decoded[:, :columns] = decoded
    gain = (
        -2
        * padded_value.reshape(rows, words, 32)
        * padded_decoded.reshape(rows, words, 32)
    )
    if padded_columns != columns:
        valid = torch.arange(padded_columns, device=decoded.device) < columns
        gain = gain.masked_fill(~valid.reshape(words, 32), -torch.inf)
    positions = gain.topk(flips_per_word, dim=-1).indices.to(torch.int8)
    if flips_per_word == 1:
        positions = positions.squeeze(-1)
    return apply_word_flips(decoded, positions), positions


def _project(
    value: torch.Tensor,
    codebook: SignCodebook | None,
    *,
    update_codebook: bool,
    inner_iterations: int,
    generator: torch.Generator,
    epsilon: float,
    assignment_batch_words: int,
    corrected_assignment_candidates: int,
    flips_per_word: int,
    free_rows: int,
    corrected_codebook_banks: int | None,
) -> tuple[
    torch.Tensor,
    SignCodebook | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    row_magnitude, column_magnitude = _rank_one_magnitudes(
        value,
        inner_iterations,
        generator,
        epsilon,
    )
    magnitude = torch.outer(row_magnitude, column_magnitude)
    if codebook is None:
        if flips_per_word or free_rows:
            raise ValueError("word corrections require a codebook")
        return magnitude * _sign(value), None, None, None
    if not 0 <= free_rows < value.shape[0]:
        raise ValueError("free rows must leave at least one codebook row")
    weighted = value.float() * magnitude.float()
    coded_weighted = weighted[free_rows:]
    flip_positions = None
    if isinstance(codebook, ProductSignCodebook):
        decoded, indices, product_updated = _assign_product_words(
            coded_weighted,
            codebook,
            update=update_codebook,
            batch_words=assignment_batch_words,
        )
        updated: SignCodebook = product_updated
        if 1 <= flips_per_word <= 3:
            decoded, flip_positions = _apply_best_word_flips(
                coded_weighted,
                decoded,
                flips_per_word,
            )
    elif isinstance(codebook, FullSignCodebook):
        if flips_per_word:
            (
                decoded,
                indices,
                full_updated,
                flip_positions,
            ) = _assign_corrected_full_words(
                coded_weighted,
                codebook,
                flips_per_word=flips_per_word,
                update=update_codebook,
                batch_words=assignment_batch_words,
                candidate_count=corrected_assignment_candidates,
            )
        else:
            decoded, indices, full_updated = _assign_full_words(
                coded_weighted,
                codebook,
                update=update_codebook,
                batch_words=assignment_batch_words,
            )
        updated = full_updated
    else:
        decoded, indices, updated, flip_positions = _assign_banked_full_words(
            coded_weighted,
            codebook,
            flips_per_word=flips_per_word,
            update=update_codebook,
            batch_words=assignment_batch_words,
            candidate_count=corrected_assignment_candidates,
            corrected_bank_count=corrected_codebook_banks,
        )
    if free_rows:
        decoded = torch.cat((_sign(value[:free_rows]), decoded), dim=0)
    if flips_per_word < 0 or flips_per_word > 3:
        raise ValueError("only zero to three corrections per word are supported")
    return magnitude * decoded.to(value.dtype), updated, indices, flip_positions


def _decode_factor(
    latent: torch.Tensor,
    indices: torch.Tensor | None,
    codebook: SignCodebook | None,
    flip_positions: torch.Tensor | None,
    *,
    free_rows: int,
) -> torch.Tensor:
    if indices is None or codebook is None:
        if free_rows or flip_positions is not None:
            raise ValueError("free-row metadata requires a codebook")
        return _sign(latent)
    coded = decode_sign_codebook(indices, codebook, latent.shape[1]).to(
        latent.dtype
    )
    if flip_positions is not None:
        corrected_rows = flip_positions.shape[0]
        coded = torch.cat(
            (
                apply_word_flips(coded[:corrected_rows], flip_positions),
                coded[corrected_rows:],
            )
        )
    if not free_rows:
        return coded
    return torch.cat((_sign(latent[:free_rows]), coded), dim=0)


def _solve(
    design: torch.Tensor,
    target: torch.Tensor,
    projected: torch.Tensor,
    dual: torch.Tensor,
    rho: float,
    regularization: float,
    epsilon: float,
) -> torch.Tensor:
    """Ridge solve using the smaller of the primal and dual systems."""

    design32 = design.float()
    diagonal_mean = design32.square().sum(dim=0).mean().abs()
    stabilizer = (rho * diagonal_mean + regularization).clamp_min(epsilon)
    target32 = target.float()
    if design32.shape[1] <= design32.shape[0]:
        system = design32.mT @ design32
        system = (system + system.mT).mul_(0.5)
        system.diagonal().add_(stabilizer)
        rhs = design32.mT @ target32
        rhs.add_(projected, alpha=rho)
        rhs.add_(dual, alpha=-rho)
        factor, info = torch.linalg.cholesky_ex(system)
        solution = (
            torch.cholesky_solve(rhs, factor)
            if int(info.max()) == 0
            else torch.linalg.solve(system, rhs)
        )
        return solution.to(design.dtype)

    # (D^T D + lambda I)^-1 with a non-zero ridge prior, evaluated through
    # the smaller D D^T system.  This is what makes rank > min(m, n)
    # practical for the over-complete codebook arms.
    prior = projected.float()
    prior.sub_(dual.float()).mul_(rho / float(stabilizer))
    residual = target32 - design32 @ prior
    system = design32 @ design32.mT
    system = (system + system.mT).mul_(0.5)
    system.diagonal().add_(stabilizer)
    factor, info = torch.linalg.cholesky_ex(system)
    correction = (
        torch.cholesky_solve(residual, factor)
        if int(info.max()) == 0
        else torch.linalg.solve(system, residual)
    )
    return (prior + design32.mT @ correction).to(design.dtype)


def _index_metrics(indices: torch.Tensor, index_bits: int) -> dict[str, float | int]:
    counts = torch.bincount(indices.reshape(-1).to(torch.int64), minlength=1 << index_bits).float()
    used = int((counts > 0).sum())
    probabilities = counts[counts > 0] / counts.sum().clamp_min(1)
    entropy = float(-(probabilities * probabilities.log2()).sum())
    return {
        "word_count": indices.numel(),
        "used_entries": used,
        "entry_count": 1 << index_bits,
        "empirical_entropy_bits": entropy,
        "maximum_frequency": float(probabilities.max()) if probabilities.numel() else 0.0,
    }


def codebook_index_metrics(
    result: SignWordCodebookADMMResult,
) -> dict[str, dict[str, float | int | bool]]:
    """Summarize actual fixed-width codebook utilization."""

    metrics: dict[str, dict[str, float | int | bool]] = {}
    for side, indices, codebook, free_rows in (
        (
            "left",
            result.left_indices,
            result.left_codebook,
            result.left_free_rows,
        ),
        (
            "right",
            result.right_indices,
            result.right_codebook,
            result.right_free_rows,
        ),
    ):
        metrics[side] = (
            {"free_words": True}
            if indices is None or codebook is None
            else (
                _index_metrics(
                    indices
                    + (
                        _word_bank_indices(
                            indices.shape[1],
                            codebook.bank_count,
                            indices.device,
                        ).reshape(1, -1)
                        if codebook.bank_axis == "word"
                        else _word_bank_indices(
                            indices.shape[0],
                            codebook.bank_count,
                            indices.device,
                        ).reshape(-1, 1)
                    )
                    * (1 << codebook.index_bits),
                    codebook.index_bits
                    + int(math.log2(codebook.bank_count)),
                )
                if isinstance(codebook, BankedFullSignCodebook)
                else _index_metrics(indices, codebook.index_bits)
            )
        )
        metrics[side]["free_row_count"] = free_rows
        if isinstance(codebook, BankedFullSignCodebook):
            metrics[side]["implicit_codebook_banks"] = codebook.bank_count
            metrics[side]["codebook_banks_by_row"] = codebook.bank_axis == "row"
    return metrics


def factorize_sign_word_codebook_admm(
    weight: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    rank: int,
    generator: torch.Generator,
    *,
    index_bits: int,
    outer_iterations: int = 400,
    inner_iterations: int = 5,
    regularization: float = 3e-2,
    penalty_schedule: str = "cubic",
    convergence_check_interval: int = 100,
    codebook_update_interval: int = 10,
    codebook_freeze_fraction: float = 0.5,
    codebook_warmup_fraction: float = 0.0,
    assignment_batch_words: int = 65_536,
    corrected_assignment_candidates: int = CORRECTED_ASSIGNMENT_CANDIDATES,
    codebook_mode: str = "product",
    constrain_left: bool = True,
    constrain_right: bool = True,
    left_flips_per_word: int = 0,
    right_flips_per_word: int = 0,
    left_free_rows: int = 0,
    right_free_rows: int = 0,
    left_codebook_banks: int = 1,
    right_codebook_banks: int = 1,
    left_codebook_bank_axis: str = "word",
    right_codebook_bank_axis: str = "word",
    left_corrected_codebook_banks: int | None = None,
    right_corrected_codebook_banks: int | None = None,
    epsilon: float = 1e-12,
) -> SignWordCodebookADMMResult:
    """Jointly fit over-complete factors constrained to fixed-width codebooks."""

    if weight.ndim != 2 or rank <= 0:
        raise ValueError("weight must be a matrix and rank positive")
    if input_importance.numel() != weight.shape[1] or output_importance.numel() != weight.shape[0]:
        raise ValueError("importance dimensions do not match weight")
    if codebook_mode not in {"product", "full"}:
        raise ValueError("codebook mode must be 'product' or 'full'")
    if index_bits <= 0 or (codebook_mode == "product" and index_bits % 2):
        raise ValueError(
            "index bits must be positive and even for product codebooks"
        )
    for banks in (left_codebook_banks, right_codebook_banks):
        if banks <= 0 or banks & (banks - 1):
            raise ValueError("codebook bank counts must be positive powers of two")
    if codebook_mode != "full" and (
        left_codebook_banks != 1 or right_codebook_banks != 1
    ):
        raise ValueError("only full codebooks support word banks")
    if left_codebook_bank_axis not in {"word", "row"} or (
        right_codebook_bank_axis not in {"word", "row"}
    ):
        raise ValueError("codebook bank axes must be 'word' or 'row'")
    if (
        left_codebook_banks
        > (
            math.ceil(rank / 32)
            if left_codebook_bank_axis == "word"
            else weight.shape[0] - left_free_rows
        )
        or right_codebook_banks
        > (
            math.ceil(weight.shape[1] / 32)
            if right_codebook_bank_axis == "word"
            else rank - right_free_rows
        )
    ):
        raise ValueError("codebook bank count exceeds the factor word count")
    if (not constrain_left and left_codebook_banks != 1) or (
        not constrain_right and right_codebook_banks != 1
    ):
        raise ValueError("free factors cannot declare codebook banks")
    for corrected_banks, banks, axis in (
        (
            left_corrected_codebook_banks,
            left_codebook_banks,
            left_codebook_bank_axis,
        ),
        (
            right_corrected_codebook_banks,
            right_codebook_banks,
            right_codebook_bank_axis,
        ),
    ):
        if corrected_banks is not None and (
            corrected_banks <= 0
            or corrected_banks > banks
            or (corrected_banks != banks and axis != "row")
        ):
            raise ValueError(
                "partial corrected banks require a valid row-banked prefix"
            )
    if not constrain_left and not constrain_right:
        raise ValueError("at least one factor must use a codebook")
    if left_flips_per_word not in {0, 1, 2, 3} or right_flips_per_word not in {
        0,
        1,
        2,
        3,
    }:
        raise ValueError("only zero to three corrections per word are supported")
    if (left_flips_per_word and not constrain_left) or (
        right_flips_per_word and not constrain_right
    ):
        raise ValueError("word corrections require a constrained factor")
    if (
        left_free_rows < 0
        or right_free_rows < 0
        or left_free_rows >= weight.shape[0]
        or right_free_rows >= rank
    ):
        raise ValueError("free-row prefixes must leave constrained rows")
    if (left_free_rows and not constrain_left) or (
        right_free_rows and not constrain_right
    ):
        raise ValueError("free-row prefixes require a codebook on that factor")
    if (
        outer_iterations <= 0
        or inner_iterations <= 0
        or convergence_check_interval <= 0
        or codebook_update_interval <= 0
        or corrected_assignment_candidates <= 0
    ):
        raise ValueError("iteration settings must be positive")
    if (
        not 0 <= codebook_warmup_fraction <= codebook_freeze_fraction <= 1
        or codebook_warmup_fraction >= 1
    ):
        raise ValueError("codebook warmup/freeze fractions are invalid")
    try:
        schedule = SCHEDULES[penalty_schedule]
    except KeyError as exc:
        raise ValueError(f"unknown penalty schedule: {penalty_schedule}") from exc

    # Over-complete constrained solves carry much larger dual states than the
    # production in-cap solve.  Keep the research optimizer in FP32 so a
    # codebook arm is not rejected because BF16 multipliers overflow; exported
    # signs remain exact and the caller applies the declared scale dtype.
    dtype = torch.float32
    target = weight.detach().to(dtype=dtype)
    input_scale = input_importance.detach().float().sqrt().clamp_min(epsilon)
    output_scale = output_importance.detach().float().sqrt().clamp_min(epsilon).reshape(-1, 1)
    normalized = target * input_scale.reshape(1, -1) * output_scale
    left = torch.randn(
        (weight.shape[0], rank),
        dtype=dtype,
        device=weight.device,
        generator=generator,
    )
    right = torch.randn(
        (rank, weight.shape[1]),
        dtype=dtype,
        device=weight.device,
        generator=generator,
    )
    left_template = (
        _random_codebook(
            index_bits,
            weight.device,
            dtype,
            generator,
            codebook_mode,
            left_codebook_banks,
            left_codebook_bank_axis,
        )
        if constrain_left
        else None
    )
    right_template = (
        _random_codebook(
            index_bits,
            weight.device,
            dtype,
            generator,
            codebook_mode,
            right_codebook_banks,
            right_codebook_bank_axis,
        )
        if constrain_right
        else None
    )
    left_codebook = left_template if codebook_warmup_fraction == 0 else None
    right_codebook = right_template if codebook_warmup_fraction == 0 else None
    (
        left_projected,
        left_codebook_value,
        left_indices_value,
        left_flip_positions,
    ) = _project(
        left,
        left_codebook,
        update_codebook=True,
        inner_iterations=inner_iterations,
        generator=generator,
        epsilon=epsilon,
        assignment_batch_words=assignment_batch_words,
        corrected_assignment_candidates=corrected_assignment_candidates,
        flips_per_word=left_flips_per_word,
        free_rows=left_free_rows,
        corrected_codebook_banks=left_corrected_codebook_banks,
    )
    (
        right_projected,
        right_codebook_value,
        right_indices_value,
        right_flip_positions,
    ) = _project(
        right,
        right_codebook,
        update_codebook=True,
        inner_iterations=inner_iterations,
        generator=generator,
        epsilon=epsilon,
        assignment_batch_words=assignment_batch_words,
        corrected_assignment_candidates=corrected_assignment_candidates,
        flips_per_word=right_flips_per_word,
        free_rows=right_free_rows,
        corrected_codebook_banks=right_corrected_codebook_banks,
    )
    left_codebook = left_codebook_value
    right_codebook = right_codebook_value
    left_indices = left_indices_value
    right_indices = right_indices_value
    left_dual = left - left_projected
    right_dual = right - right_projected
    trace: list[ADMMTracePoint] = []
    activation_iteration = max(
        1,
        math.floor(outer_iterations * codebook_warmup_fraction),
    )
    for iteration in range(outer_iterations):
        rho = schedule(iteration / max(1, outer_iterations))
        right_norm = right_projected.norm(dim=1).clamp_min(epsilon)
        left = _solve(
            right_projected.mT / right_norm,
            normalized.mT,
            left_projected.mT,
            left_dual.mT,
            rho,
            regularization,
            epsilon,
        ).mT
        left_norm = left_projected.norm(dim=0).clamp_min(epsilon)
        right = _solve(
            left_projected / left_norm,
            normalized,
            right_projected,
            right_dual,
            rho,
            regularization,
            epsilon,
        )
        previous_left = left_projected
        previous_right = right_projected
        completed = iteration + 1
        update_ceiling = math.floor(outer_iterations * codebook_freeze_fraction)
        activate = (
            codebook_warmup_fraction > 0
            and completed == activation_iteration
        )
        if activate:
            left_codebook = left_template
            right_codebook = right_template
        update = activate or (
            completed % codebook_update_interval == 0
            and completed <= update_ceiling
            and completed >= activation_iteration
        )
        (
            left_projected,
            left_codebook_value,
            left_indices_value,
            left_flip_positions,
        ) = _project(
            left + left_dual,
            left_codebook,
            update_codebook=update,
            inner_iterations=inner_iterations,
            generator=generator,
            epsilon=epsilon,
            assignment_batch_words=assignment_batch_words,
            corrected_assignment_candidates=corrected_assignment_candidates,
            flips_per_word=left_flips_per_word,
            free_rows=left_free_rows,
            corrected_codebook_banks=left_corrected_codebook_banks,
        )
        (
            right_projected,
            right_codebook_value,
            right_indices_value,
            right_flip_positions,
        ) = _project(
            right + right_dual,
            right_codebook,
            update_codebook=update,
            inner_iterations=inner_iterations,
            generator=generator,
            epsilon=epsilon,
            assignment_batch_words=assignment_batch_words,
            corrected_assignment_candidates=corrected_assignment_candidates,
            flips_per_word=right_flips_per_word,
            free_rows=right_free_rows,
            corrected_codebook_banks=right_corrected_codebook_banks,
        )
        left_codebook = left_codebook_value
        right_codebook = right_codebook_value
        left_indices = left_indices_value
        right_indices = right_indices_value
        if update:
            # Updating the discrete feasible set invalidates the accumulated
            # multiplier.  Restart the dual residual at the new projection;
            # otherwise stale multipliers can explosively oppose a moved
            # codeword late in the solve.
            left_dual = left - left_projected
            right_dual = right - right_projected
        else:
            left_dual.add_(left - left_projected)
            right_dual.add_(right - right_projected)
        if iteration == 0 or completed % convergence_check_interval == 0 or completed == outer_iterations:
            primal = float((left - left_projected).norm() + (right - right_projected).norm())
            dual_residual = float(
                rho * ((left_projected - previous_left).norm() + (right_projected - previous_right).norm())
            )
            trace.append(ADMMTracePoint(completed, rho, primal, dual_residual))

    left_unbalanced = left_projected / output_scale
    right_unbalanced = right_projected / input_scale
    balance = (right_unbalanced.norm().clamp_min(epsilon) / left_unbalanced.norm().clamp_min(epsilon)).sqrt()
    left_export = left_unbalanced * balance
    right_export = right_unbalanced / balance
    left_latent = ((left + left_dual) / output_scale) * balance
    right_latent = ((right + right_dual) / input_scale) / balance
    scale_factor = left_projected.norm(dim=0).clamp_min(epsilon).reciprocal()
    left_export = left_export * scale_factor

    right_u, scale_pre = _rank_one_magnitudes(
        right_export.float(),
        inner_iterations,
        generator,
        epsilon,
    )
    left_u, scale_post = _rank_one_magnitudes(
        left_export.mT.float(),
        inner_iterations,
        generator,
        epsilon,
    )
    left_binary = _decode_factor(
        left_export,
        left_indices,
        left_codebook,
        left_flip_positions,
        free_rows=left_free_rows,
    )
    right_binary = _decode_factor(
        right_export,
        right_indices,
        right_codebook,
        right_flip_positions,
        free_rows=right_free_rows,
    )
    scale_mid = (right_u * left_u).to(dtype)
    scale_pre = scale_pre.to(dtype)
    scale_post = scale_post.to(dtype)
    reconstruction = (left_binary * scale_post.reshape(-1, 1)) @ (
        right_binary * scale_mid.reshape(-1, 1) * scale_pre.reshape(1, -1)
    )
    factors = ADMMResult(
        left_latent.clone().contiguous(),
        right_latent.clone().contiguous(),
        left_binary.contiguous(),
        right_binary.contiguous(),
        scale_pre.contiguous(),
        scale_mid.contiguous(),
        scale_post.contiguous(),
        reconstruction.contiguous(),
        outer_iterations,
        False,
        tuple(trace),
    )
    return SignWordCodebookADMMResult(
        factors,
        left_codebook,
        right_codebook,
        left_indices.contiguous() if left_indices is not None else None,
        right_indices.contiguous() if right_indices is not None else None,
        (
            left_flip_positions.contiguous()
            if left_flip_positions is not None
            else None
        ),
        (
            right_flip_positions.contiguous()
            if right_flip_positions is not None
            else None
        ),
        left_free_rows,
        right_free_rows,
    )
