"""Exact-objective coordinate search over free and codebook sign words."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .scale_fit import fit_scales, reconstruct
from .sign_word_codebook import (
    FullSignCodebook,
    ProductSignCodebook,
    apply_word_flips,
    decode_product_codebook,
    decode_sign_codebook,
)


@dataclass(frozen=True, slots=True)
class SignWordPayloadSearchConfig:
    """Bounded analysis settings for exact word-coordinate proposals."""

    enabled: bool = False
    outer_passes: int = 2
    max_words_per_pass: int = 2_048
    scale_passes: int = 64
    candidate_batch_words: int = 2_048
    table_chunk_size: int = 128
    acceptance_tolerance: float = 1e-10
    functional_candidate_words_per_pass: int = 0
    functional_table_bit_passes: int = 0
    functional_table_bit_candidates_per_pass: int = 4


@dataclass(frozen=True, slots=True)
class SignWordPayloadSearchResult:
    right_binary: torch.Tensor
    right_indices: torch.Tensor | None
    right_flip_positions: torch.Tensor | None
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    reconstruction: torch.Tensor
    before_error: float
    after_error: float
    accepted_outer_passes: int
    candidate_words_evaluated: int
    codebook_patterns_evaluated: int
    selected_words: int
    accepted_words: int
    sign_updates: int
    functional_candidates_ranked: int = 0
    functional_fit_error_before: float | None = None
    functional_fit_error_after: float | None = None
    functional_held_out_error_before: float | None = None
    functional_held_out_error_after: float | None = None
    right_codebook: FullSignCodebook | ProductSignCodebook | None = None
    table_bit_candidates_evaluated: int = 0
    accepted_table_bit_flips: int = 0
    table_bit_sign_updates: int = 0


def _weighted_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> float:
    return float(
        (
            (target.float() - prediction.float()).square()
            * output_importance.float().reshape(-1, 1)
            * input_importance.float().reshape(1, -1)
        ).sum()
    )


def _functional_residual(
    inputs: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    return inputs.float() @ (target.float() - prediction.float()).mT


def _functional_change(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    component_vector: torch.Tensor,
    delta_word: torch.Tensor,
    scale_pre_word: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    latent_delta = inputs.float() @ (delta_word.float() * scale_pre_word.float())
    output_delta = latent_delta[:, None] * component_vector.float()[None, :]
    change = float(
        (-2.0 * residual.float() * output_delta + output_delta.square()).sum()
    )
    return change, output_delta


def _pad_words(value: torch.Tensor, padded_columns: int, fill: float) -> torch.Tensor:
    if value.shape[1] == padded_columns:
        return value
    padded = torch.full(
        (value.shape[0], padded_columns),
        fill,
        dtype=value.dtype,
        device=value.device,
    )
    padded[:, : value.shape[1]] = value
    return padded


def _best_corrected_payload_candidates(
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    current: torch.Tensor,
    table: torch.Tensor,
    *,
    corrections_per_word: int,
    table_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact best table index and corrections for each word.

    With every other factor sign and all scales fixed, columns are separable.
    ``flip_cost`` is therefore the exact objective change from toggling one
    current sign. A table entry pays the sum for its mismatches; each stored
    correction toggles one mismatch decision.
    """

    if linear.shape != quadratic.shape or linear.shape != current.shape:
        raise ValueError("payload candidate tensors must share one shape")
    if linear.ndim != 2 or linear.shape[1] != 32:
        raise ValueError("payload candidates require batches of 32-sign words")
    if not 1 <= corrections_per_word <= 3:
        raise ValueError("payload search supports one to three corrections")
    if table.ndim != 2 or table.shape[1] != 32 or table_chunk_size <= 0:
        raise ValueError("payload search table settings are invalid")

    linear32 = linear.float()
    quadratic32 = quadratic.float()
    current32 = current.float()
    table32 = table.float()
    flip_cost = 4.0 * (linear32 * current32 + quadratic32)
    weighted_current = current32 * flip_cost
    summed = flip_cost.sum(dim=1)
    best_cost = torch.full(
        (linear.shape[0],),
        torch.inf,
        dtype=torch.float32,
        device=linear.device,
    )
    best_index = torch.zeros(
        linear.shape[0], dtype=torch.int64, device=linear.device
    )
    best_positions = torch.zeros(
        (linear.shape[0], corrections_per_word),
        dtype=torch.int64,
        device=linear.device,
    )
    rows = torch.arange(linear.shape[0], device=linear.device)
    for start in range(0, table.shape[0], table_chunk_size):
        stop = min(table.shape[0], start + table_chunk_size)
        entries = table32[start:stop]
        base_cost = 0.5 * (summed[:, None] - weighted_current @ entries.mT)
        correction_delta = (
            current32[:, None, :] * entries[None, :, :] * flip_cost[:, None, :]
        )
        correction_cost, positions = correction_delta.topk(
            corrections_per_word,
            dim=2,
            largest=False,
        )
        total = base_cost + correction_cost.sum(dim=2)
        local_cost, local_index = total.min(dim=1)
        improved = local_cost < best_cost
        if bool(improved.any()):
            best_cost[improved] = local_cost[improved]
            best_index[improved] = local_index[improved] + start
            best_positions[improved] = positions[rows, local_index][improved]
    return best_cost, best_index, best_positions.to(torch.int8)


def _best_product_payload_candidates(
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    current: torch.Tensor,
    codebook: ProductSignCodebook,
    *,
    table_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact best Cartesian-product codeword for each word.

    The weighted coordinate objective is separable across columns.  The best
    32-sign Cartesian product is therefore exactly the independently best
    entry from each 16-sign half table; no 65,536-pair materialization is
    needed for a k16 product code.
    """

    if linear.shape != quadratic.shape or linear.shape != current.shape:
        raise ValueError("payload candidate tensors must share one shape")
    if linear.ndim != 2 or linear.shape[1] != 32:
        raise ValueError("payload candidates require batches of 32-sign words")
    if table_chunk_size <= 0:
        raise ValueError("payload search table settings are invalid")

    def best_half(
        half_linear: torch.Tensor,
        half_quadratic: torch.Tensor,
        half_current: torch.Tensor,
        table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flip_cost = 4.0 * (
            half_linear.float() * half_current.float() + half_quadratic.float()
        )
        weighted_current = half_current.float() * flip_cost
        summed = flip_cost.sum(dim=1)
        best_cost = torch.full(
            (half_linear.shape[0],),
            torch.inf,
            dtype=torch.float32,
            device=linear.device,
        )
        best_index = torch.zeros(
            half_linear.shape[0], dtype=torch.int64, device=linear.device
        )
        for start in range(0, table.shape[0], table_chunk_size):
            stop = min(table.shape[0], start + table_chunk_size)
            costs = 0.5 * (
                summed[:, None]
                - weighted_current @ table[start:stop].float().mT
            )
            local_cost, local_index = costs.min(dim=1)
            improved = local_cost < best_cost
            best_cost[improved] = local_cost[improved]
            best_index[improved] = local_index[improved] + start
        return best_cost, best_index

    first_cost, first_index = best_half(
        linear[:, :16], quadratic[:, :16], current[:, :16], codebook.first
    )
    second_cost, second_index = best_half(
        linear[:, 16:], quadratic[:, 16:], current[:, 16:], codebook.second
    )
    half_bits = codebook.index_bits // 2
    indices = first_index.bitwise_or(second_index.bitwise_left_shift(half_bits))
    return first_cost + second_cost, indices


def _decode_candidate_word(
    codebook: FullSignCodebook | ProductSignCodebook,
    index: int,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    if isinstance(codebook, ProductSignCodebook):
        half_bits = codebook.index_bits // 2
        mask = (1 << half_bits) - 1
        return torch.cat(
            (
                codebook.first[index & mask],
                codebook.second[index >> half_bits],
            )
        )
    if positions is None:
        raise ValueError("corrected-code candidate requires correction positions")
    candidate = codebook.entries[index].clone()
    candidate[positions.to(torch.int64)] *= -1
    return candidate


def _validate_payload(
    right: torch.Tensor,
    codebook: FullSignCodebook | ProductSignCodebook,
    indices: torch.Tensor,
    flip_positions: torch.Tensor | None,
    free_rows: int,
) -> None:
    columns = right.shape[1]
    padded_columns = math.ceil(columns / 32) * 32
    if isinstance(codebook, ProductSignCodebook):
        if flip_positions is not None:
            raise ValueError("product-code payload must not store correction positions")
        decoded = decode_product_codebook(indices, codebook, padded_columns)[:, :columns]
    else:
        if flip_positions is None:
            raise ValueError("corrected-code payload requires correction positions")
        decoded = decode_sign_codebook(indices, codebook, padded_columns)[:, :columns]
        decoded = apply_word_flips(decoded, flip_positions)
    if not torch.equal(decoded.to(right.dtype), right[free_rows:]):
        raise ValueError("codebook payload does not decode to the supplied factor")


def _functional_flip_changes(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    scaled_left: torch.Tensor,
    right: torch.Tensor,
    scale_pre: torch.Tensor,
) -> torch.Tensor:
    """Diagonal Gauss--Newton score for flipping each right-factor sign."""

    projected_residual = residual.float() @ scaled_left.float()
    cross = (inputs.float().mT @ projected_residual).mT
    component_norm = scaled_left.float().square().sum(dim=0)
    input_norm = inputs.float().square().sum(dim=0)
    return 4.0 * (
        right.float() * scale_pre.float()[None, :] * cross
        + component_norm[:, None]
        * input_norm[None, :]
        * scale_pre.float().square()[None, :]
    )


def _product_table_bit_scores(
    flip_changes: torch.Tensor,
    indices: torch.Tensor,
    *,
    free_rows: int,
    index_bits: int,
) -> torch.Tensor:
    """Aggregate per-sign flip scores for both shared half-word tables."""

    coded_rows, words = indices.shape
    if flip_changes.shape[0] - free_rows != coded_rows:
        raise ValueError("table-bit score rows do not match product assignments")
    half_bits = index_bits // 2
    entry_count = 1 << half_bits
    mask = entry_count - 1
    values = indices.to(torch.int64)
    table_indices = (
        values.bitwise_and(mask),
        values.bitwise_right_shift(half_bits),
    )
    scores = torch.zeros(
        (2, entry_count, 16),
        dtype=torch.float32,
        device=flip_changes.device,
    )
    word_ids = torch.arange(words, device=flip_changes.device)
    for side in range(2):
        for bit in range(16):
            columns = word_ids * 32 + side * 16 + bit
            valid = columns < flip_changes.shape[1]
            if not bool(valid.any()):
                continue
            selected = flip_changes[
                free_rows:,
                columns[valid],
            ].reshape(-1)
            entries = table_indices[side][:, valid].reshape(-1)
            scores[side, :, bit].scatter_add_(0, entries, selected)
    return scores


def _refine_product_table_bits(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    codebook: ProductSignCodebook,
    indices: torch.Tensor,
    *,
    free_rows: int,
    fit_inputs: torch.Tensor,
    held_inputs: torch.Tensor,
    config: SignWordPayloadSearchConfig,
) -> tuple[
    torch.Tensor,
    ProductSignCodebook,
    torch.Tensor,
    float,
    torch.Tensor,
    float,
    torch.Tensor,
    float,
    int,
    int,
    int,
]:
    """Flip shared table bits only when exact fit and validation losses improve."""

    prediction = reconstruct(left, right, pre, mid, post).float()
    best_error = _weighted_error(target, prediction, input_weight, output_weight)
    fit_residual = _functional_residual(fit_inputs, target, prediction)
    held_residual = _functional_residual(held_inputs, target, prediction)
    fit_error = float(fit_residual.square().sum())
    held_error = float(held_residual.square().sum())
    candidates_evaluated = 0
    accepted_flips = 0
    sign_updates = 0
    for _ in range(config.functional_table_bit_passes):
        scaled_left = left * (post[:, None] * mid[None, :])
        fit_scores = _product_table_bit_scores(
            _functional_flip_changes(
                fit_inputs,
                fit_residual,
                scaled_left,
                right,
                pre,
            ),
            indices,
            free_rows=free_rows,
            index_bits=codebook.index_bits,
        )
        held_scores = _product_table_bit_scores(
            _functional_flip_changes(
                held_inputs,
                held_residual,
                scaled_left,
                right,
                pre,
            ),
            indices,
            free_rows=free_rows,
            index_bits=codebook.index_bits,
        )
        viable = torch.nonzero(
            (fit_scores < 0) & (held_scores < 0),
            as_tuple=False,
        )
        if viable.numel() == 0:
            break
        order = held_scores[
            viable[:, 0], viable[:, 1], viable[:, 2]
        ].argsort()
        accepted = False
        for candidate_index in order[
            : config.functional_table_bit_candidates_per_pass
        ].tolist():
            side, entry, bit = (
                int(value) for value in viable[candidate_index].tolist()
            )
            half_bits = codebook.index_bits // 2
            mask = (1 << half_bits) - 1
            assigned = (
                indices.to(torch.int64).bitwise_and(mask)
                if side == 0
                else indices.to(torch.int64).bitwise_right_shift(half_bits)
            )
            word_ids = torch.arange(indices.shape[1], device=indices.device)
            columns = word_ids * 32 + side * 16 + bit
            valid = columns < right.shape[1]
            occurrences = torch.nonzero(
                (assigned == entry) & valid.reshape(1, -1),
                as_tuple=False,
            )
            if occurrences.numel() == 0:
                continue
            candidate_right = right.clone()
            components = occurrences[:, 0] + free_rows
            occurrence_columns = columns[occurrences[:, 1]]
            candidate_right[components, occurrence_columns] *= -1
            candidate_prediction = reconstruct(
                left,
                candidate_right,
                pre,
                mid,
                post,
            ).float()
            candidate_fit_residual = _functional_residual(
                fit_inputs,
                target,
                candidate_prediction,
            )
            candidate_held_residual = _functional_residual(
                held_inputs,
                target,
                candidate_prediction,
            )
            candidate_fit_error = float(candidate_fit_residual.square().sum())
            candidate_held_error = float(candidate_held_residual.square().sum())
            candidates_evaluated += 1
            fit_threshold = config.acceptance_tolerance * max(abs(fit_error), 1.0)
            held_threshold = config.acceptance_tolerance * max(abs(held_error), 1.0)
            if (
                not math.isfinite(candidate_fit_error)
                or not math.isfinite(candidate_held_error)
                or candidate_fit_error >= fit_error - fit_threshold
                or candidate_held_error >= held_error - held_threshold
            ):
                continue
            first = codebook.first.clone()
            second = codebook.second.clone()
            (first if side == 0 else second)[entry, bit] *= -1
            codebook = ProductSignCodebook(codebook.index_bits, first, second)
            right = candidate_right
            prediction = candidate_prediction
            best_error = _weighted_error(
                target,
                prediction,
                input_weight,
                output_weight,
            )
            fit_residual = candidate_fit_residual
            held_residual = candidate_held_residual
            fit_error = candidate_fit_error
            held_error = candidate_held_error
            accepted_flips += 1
            sign_updates += int(occurrences.shape[0])
            accepted = True
            break
        if not accepted:
            break
    return (
        right,
        codebook,
        prediction,
        best_error,
        fit_residual,
        fit_error,
        held_residual,
        held_error,
        candidates_evaluated,
        accepted_flips,
        sign_updates,
    )


def refine_sign_word_payloads(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    free_rows: int,
    codebook: FullSignCodebook | ProductSignCodebook | None,
    right_indices: torch.Tensor | None,
    right_flip_positions: torch.Tensor | None,
    config: SignWordPayloadSearchConfig,
    functional_fit_inputs: torch.Tensor | None = None,
    functional_held_out_inputs: torch.Tensor | None = None,
) -> SignWordPayloadSearchResult:
    """Refine right-factor words without changing their stored representation.

    Free rows receive their exact best arbitrary 32-sign coordinate proposal.
    Coded rows evaluate every full-table entry plus fixed-count corrections,
    or both halves of a Cartesian-product table. The highest-gain bounded proposals are rescored sequentially,
    followed by a common full scale refit and exact outer-pass rollback.
    """

    if not config.enabled:
        raise ValueError("sign-word payload search is not enabled")
    if (
        config.outer_passes < 0
        or config.max_words_per_pass < 0
        or config.scale_passes < 0
        or config.candidate_batch_words <= 0
        or config.table_chunk_size <= 0
        or config.acceptance_tolerance < 0
        or config.functional_candidate_words_per_pass < 0
        or config.functional_table_bit_passes < 0
        or config.functional_table_bit_candidates_per_pass <= 0
    ):
        raise ValueError("sign-word payload search settings are invalid")
    if target.ndim != 2 or left_binary.ndim != 2 or right_binary.ndim != 2:
        raise ValueError("sign-word payload search requires matrices")
    rank = left_binary.shape[1]
    if (
        left_binary.shape[0] != target.shape[0]
        or right_binary.shape != (rank, target.shape[1])
        or scale_pre.numel() != target.shape[1]
        or scale_mid.numel() != rank
        or scale_post.numel() != target.shape[0]
        or input_importance.numel() != target.shape[1]
        or output_importance.numel() != target.shape[0]
        or not 0 <= free_rows <= rank
    ):
        raise ValueError("sign-word payload search dimensions do not match")
    functional_enabled = (
        config.functional_candidate_words_per_pass > 0
        or config.functional_table_bit_passes > 0
    )
    if functional_enabled:
        if (
            functional_fit_inputs is None
            or functional_held_out_inputs is None
            or functional_fit_inputs.ndim != 2
            or functional_held_out_inputs.ndim != 2
            or functional_fit_inputs.shape[1] != target.shape[1]
            or functional_held_out_inputs.shape[1] != target.shape[1]
            or functional_fit_inputs.shape[0] == 0
            or functional_held_out_inputs.shape[0] == 0
        ):
            raise ValueError(
                "functional payload acceptance requires non-empty matching inputs"
            )
    elif functional_fit_inputs is not None or functional_held_out_inputs is not None:
        raise ValueError(
            "functional payload inputs require a positive functional candidate budget"
        )
    if codebook is None:
        if free_rows != rank or right_indices is not None or right_flip_positions is not None:
            raise ValueError("free-word search must expose every row and omit payload metadata")
        corrections_per_word = 0
    elif isinstance(codebook, FullSignCodebook):
        if (
            free_rows >= rank
            or right_indices is None
            or right_flip_positions is None
            or right_flip_positions.ndim != 3
            or right_flip_positions.shape[2] not in {1, 2, 3}
        ):
            raise ValueError("corrected-code search requires complete payload metadata")
        words = math.ceil(target.shape[1] / 32)
        expected_rows = rank - free_rows
        if tuple(right_indices.shape) != (expected_rows, words) or tuple(
            right_flip_positions.shape[:2]
        ) != (expected_rows, words):
            raise ValueError("corrected-code payload metadata has the wrong shape")
        corrections_per_word = right_flip_positions.shape[2]
    else:
        if free_rows >= rank or right_indices is None or right_flip_positions is not None:
            raise ValueError("product-code search requires indices and no corrections")
        words = math.ceil(target.shape[1] / 32)
        expected_rows = rank - free_rows
        if tuple(right_indices.shape) != (expected_rows, words):
            raise ValueError("product-code payload metadata has the wrong shape")
        corrections_per_word = 0
    if config.functional_table_bit_passes and not isinstance(
        codebook, ProductSignCodebook
    ):
        raise ValueError(
            "functional table-bit search requires a product codebook"
        )

    target32 = target.detach().float()
    left = left_binary.detach().float()
    right = right_binary.detach().float().clone()
    pre = scale_pre.detach().float().reshape(-1).clone()
    mid = scale_mid.detach().float().reshape(-1).clone()
    post = scale_post.detach().float().reshape(-1).clone()
    input_weight = input_importance.detach().float().reshape(-1).clamp_min(1e-8)
    output_weight = output_importance.detach().float().reshape(-1).clamp_min(1e-8)
    indices = None if right_indices is None else right_indices.detach().clone()
    positions = (
        None if right_flip_positions is None else right_flip_positions.detach().clone()
    )
    if isinstance(codebook, ProductSignCodebook):
        codebook = ProductSignCodebook(
            codebook.index_bits,
            codebook.first.detach().clone(),
            codebook.second.detach().clone(),
        )
    if codebook is not None:
        assert indices is not None
        _validate_payload(right, codebook, indices, positions, free_rows)

    prediction = reconstruct(left, right, pre, mid, post).float()
    best_error = _weighted_error(
        target32, prediction, input_weight, output_weight
    )
    before_error = best_error
    fit_inputs = (
        None
        if functional_fit_inputs is None
        else functional_fit_inputs.detach().to(device=target.device, dtype=torch.float32)
    )
    held_inputs = (
        None
        if functional_held_out_inputs is None
        else functional_held_out_inputs.detach().to(
            device=target.device, dtype=torch.float32
        )
    )
    fit_residual = (
        None if fit_inputs is None else _functional_residual(fit_inputs, target32, prediction)
    )
    held_residual = (
        None
        if held_inputs is None
        else _functional_residual(held_inputs, target32, prediction)
    )
    fit_error = None if fit_residual is None else float(fit_residual.square().sum())
    held_error = (
        None if held_residual is None else float(held_residual.square().sum())
    )
    fit_error_before = fit_error
    held_error_before = held_error
    columns = target.shape[1]
    words = math.ceil(columns / 32)
    padded_columns = words * 32
    candidate_words_evaluated = 0
    codebook_patterns_evaluated = 0
    selected_words = 0
    accepted_words = 0
    sign_updates = 0
    accepted_outer_passes = 0
    functional_candidates_ranked = 0
    table_bit_candidates_evaluated = 0
    accepted_table_bit_flips = 0
    table_bit_sign_updates = 0

    for _ in range(config.outer_passes):
        previous = (
            right.clone(),
            None if indices is None else indices.clone(),
            None if positions is None else positions.clone(),
            pre.clone(),
            mid.clone(),
            post.clone(),
            prediction.clone(),
            best_error,
            None if fit_residual is None else fit_residual.clone(),
            None if held_residual is None else held_residual.clone(),
            fit_error,
            held_error,
        )
        residual = target32 - prediction
        scaled_left = left * (post[:, None] * mid[None, :])
        linear = scaled_left.mT @ (residual * output_weight[:, None])
        linear *= (input_weight * pre)[None, :]
        component_norm = (scaled_left.square() * output_weight[:, None]).sum(dim=0)
        quadratic = component_norm[:, None] * (
            input_weight * pre.square()
        )[None, :]
        padded_linear = _pad_words(linear, padded_columns, 0.0).reshape(
            rank, words, 32
        )
        padded_quadratic = _pad_words(quadratic, padded_columns, 0.0).reshape(
            rank, words, 32
        )
        padded_right = _pad_words(right, padded_columns, 1.0).reshape(
            rank, words, 32
        )
        costs = torch.full(
            (rank, words),
            torch.inf,
            dtype=torch.float32,
            device=target.device,
        )
        free_flip_mask = torch.zeros(
            (free_rows, words, 32), dtype=torch.bool, device=target.device
        )
        if free_rows:
            free_flip_cost = 4.0 * (
                padded_linear[:free_rows] * padded_right[:free_rows]
                + padded_quadratic[:free_rows]
            )
            free_flip_mask = free_flip_cost < 0
            costs[:free_rows] = torch.where(
                free_flip_mask, free_flip_cost, 0.0
            ).sum(dim=2)

        coded_indices = None
        coded_positions = None
        if codebook is not None:
            coded_rows = rank - free_rows
            coded_words = coded_rows * words
            coded_indices = torch.empty(
                coded_words, dtype=torch.int64, device=target.device
            )
            if isinstance(codebook, FullSignCodebook):
                coded_positions = torch.empty(
                    (coded_words, corrections_per_word),
                    dtype=torch.int8,
                    device=target.device,
                )
            flat_linear = padded_linear[free_rows:].reshape(-1, 32)
            flat_quadratic = padded_quadratic[free_rows:].reshape(-1, 32)
            flat_right = padded_right[free_rows:].reshape(-1, 32)
            flat_costs = torch.empty(
                coded_words, dtype=torch.float32, device=target.device
            )
            for start in range(0, coded_words, config.candidate_batch_words):
                stop = min(coded_words, start + config.candidate_batch_words)
                if isinstance(codebook, ProductSignCodebook):
                    batch_cost, batch_indices = _best_product_payload_candidates(
                        flat_linear[start:stop],
                        flat_quadratic[start:stop],
                        flat_right[start:stop],
                        codebook,
                        table_chunk_size=config.table_chunk_size,
                    )
                else:
                    batch_cost, batch_indices, batch_positions = _best_corrected_payload_candidates(
                        flat_linear[start:stop],
                        flat_quadratic[start:stop],
                        flat_right[start:stop],
                        codebook.entries,
                        corrections_per_word=corrections_per_word,
                        table_chunk_size=config.table_chunk_size,
                    )
                    assert coded_positions is not None
                    coded_positions[start:stop] = batch_positions
                flat_costs[start:stop] = batch_cost
                coded_indices[start:stop] = batch_indices
            costs[free_rows:] = flat_costs.reshape(coded_rows, words)
            codebook_patterns_evaluated += coded_words * (
                codebook.first.shape[0] + codebook.second.shape[0]
                if isinstance(codebook, ProductSignCodebook)
                else codebook.entry_count
            )

        candidate_words_evaluated += rank * words
        available = min(config.max_words_per_pass, costs.numel())
        if available == 0:
            break
        improvements, selected = (-costs).reshape(-1).topk(available)
        threshold = config.acceptance_tolerance * max(abs(best_error), 1.0)
        useful = improvements > threshold
        selected = selected[useful]
        if selected.numel() == 0:
            break
        if config.functional_candidate_words_per_pass > 0:
            assert fit_inputs is not None and held_inputs is not None
            assert fit_residual is not None and held_residual is not None
            assert fit_error is not None and held_error is not None
            selected = selected[
                : config.functional_candidate_words_per_pass
            ]
            ranked: list[tuple[float, int]] = []
            fit_threshold = config.acceptance_tolerance * max(abs(fit_error), 1.0)
            held_threshold = config.acceptance_tolerance * max(abs(held_error), 1.0)
            for flat_index in selected.tolist():
                component = flat_index // words
                word = flat_index % words
                start = word * 32
                stop = min(start + 32, columns)
                current = right[component, start:stop]
                if component < free_rows:
                    candidate = padded_right[component, word].clone()
                    candidate[free_flip_mask[component, word]] *= -1
                else:
                    assert codebook is not None
                    assert coded_indices is not None
                    coded_flat = (component - free_rows) * words + word
                    candidate = _decode_candidate_word(
                        codebook,
                        int(coded_indices[coded_flat]),
                        None if coded_positions is None else coded_positions[coded_flat],
                    )
                delta = candidate[: stop - start] - current
                if not bool(delta.any()):
                    continue
                component_vector = scaled_left[:, component]
                fit_change, _fit_delta = _functional_change(
                    fit_inputs[:, start:stop],
                    fit_residual,
                    component_vector,
                    delta,
                    pre[start:stop],
                )
                held_change, _held_delta = _functional_change(
                    held_inputs[:, start:stop],
                    held_residual,
                    component_vector,
                    delta,
                    pre[start:stop],
                )
                functional_candidates_ranked += 1
                if (
                    math.isfinite(fit_change)
                    and math.isfinite(held_change)
                    and fit_change < -fit_threshold
                    and held_change < -held_threshold
                ):
                    ranked.append((held_change, flat_index))
            ranked.sort()
            selected = torch.tensor(
                [flat_index for _change, flat_index in ranked],
                dtype=torch.int64,
                device=target.device,
            )
            if selected.numel() == 0:
                break
        selected_words += int(selected.numel())

        pass_accepted = 0
        pass_sign_updates = 0
        for flat_index in selected.tolist():
            component = flat_index // words
            word = flat_index % words
            start = word * 32
            stop = min(start + 32, columns)
            current = right[component, start:stop]
            if component < free_rows:
                candidate = padded_right[component, word].clone()
                candidate[free_flip_mask[component, word]] *= -1
            else:
                assert codebook is not None
                assert coded_indices is not None
                coded_flat = (component - free_rows) * words + word
                candidate = _decode_candidate_word(
                    codebook,
                    int(coded_indices[coded_flat]),
                    None if coded_positions is None else coded_positions[coded_flat],
                )
            candidate = candidate[: stop - start]
            delta = candidate - current
            if not bool(delta.any()):
                continue
            component_vector = scaled_left[:, component]
            local_residual = target32[:, start:stop] - prediction[:, start:stop]
            local_linear = (
                component_vector[:, None]
                * local_residual
                * output_weight[:, None]
            ).sum(dim=0)
            local_linear *= input_weight[start:stop] * pre[start:stop]
            local_quadratic = (
                component_vector.square() * output_weight
            ).sum() * input_weight[start:stop] * pre[start:stop].square()
            change = float(
                (-2.0 * local_linear * delta + local_quadratic * delta.square()).sum()
            )
            if not math.isfinite(change) or change >= -threshold:
                continue
            fit_delta = None
            held_delta = None
            if config.functional_candidate_words_per_pass > 0:
                assert fit_inputs is not None and held_inputs is not None
                assert fit_residual is not None and held_residual is not None
                assert fit_error is not None and held_error is not None
                fit_change, fit_delta = _functional_change(
                    fit_inputs[:, start:stop],
                    fit_residual,
                    component_vector,
                    delta,
                    pre[start:stop],
                )
                held_change, held_delta = _functional_change(
                    held_inputs[:, start:stop],
                    held_residual,
                    component_vector,
                    delta,
                    pre[start:stop],
                )
                fit_threshold = config.acceptance_tolerance * max(
                    abs(fit_error), 1.0
                )
                held_threshold = config.acceptance_tolerance * max(
                    abs(held_error), 1.0
                )
                if (
                    not math.isfinite(fit_change)
                    or not math.isfinite(held_change)
                    or fit_change >= -fit_threshold
                    or held_change >= -held_threshold
                ):
                    continue
            right[component, start:stop] = candidate
            prediction[:, start:stop] += component_vector[:, None] * (
                delta * pre[start:stop]
            )[None, :]
            if fit_delta is not None and held_delta is not None:
                assert fit_residual is not None and held_residual is not None
                assert fit_error is not None and held_error is not None
                fit_residual -= fit_delta
                held_residual -= held_delta
                fit_error = float(fit_residual.square().sum())
                held_error = float(held_residual.square().sum())
            if component >= free_rows:
                assert indices is not None and coded_indices is not None
                coded_flat = (component - free_rows) * words + word
                indices[component - free_rows, word] = coded_indices[coded_flat]
                if positions is not None:
                    assert coded_positions is not None
                    positions[component - free_rows, word] = coded_positions[coded_flat]
            pass_accepted += 1
            pass_sign_updates += int((delta != 0).sum())

        if pass_accepted == 0:
            break
        fitted = fit_scales(
            target32,
            left,
            right,
            pre,
            mid,
            post,
            input_weight,
            output_weight,
            alternating_passes=config.scale_passes,
        )
        improvement = best_error - fitted.after_error
        fitted_fit_residual = None
        fitted_held_residual = None
        fitted_fit_error = None
        fitted_held_error = None
        functional_pass_accepted = True
        if config.functional_candidate_words_per_pass > 0:
            assert fit_inputs is not None and held_inputs is not None
            previous_fit_error = previous[-2]
            previous_held_error = previous[-1]
            assert previous_fit_error is not None and previous_held_error is not None
            fitted_fit_residual = _functional_residual(
                fit_inputs, target32, fitted.reconstruction
            )
            fitted_held_residual = _functional_residual(
                held_inputs, target32, fitted.reconstruction
            )
            fitted_fit_error = float(fitted_fit_residual.square().sum())
            fitted_held_error = float(fitted_held_residual.square().sum())
            functional_pass_accepted = (
                math.isfinite(fitted_fit_error)
                and math.isfinite(fitted_held_error)
                and fitted_fit_error
                < previous_fit_error
                - config.acceptance_tolerance * max(abs(previous_fit_error), 1.0)
                and fitted_held_error
                < previous_held_error
                - config.acceptance_tolerance * max(abs(previous_held_error), 1.0)
            )
        if (
            not math.isfinite(fitted.after_error)
            or improvement <= threshold
            or not functional_pass_accepted
        ):
            (
                right,
                indices,
                positions,
                pre,
                mid,
                post,
                prediction,
                best_error,
                fit_residual,
                held_residual,
                fit_error,
                held_error,
            ) = previous
            break
        pre = fitted.scale_pre
        mid = fitted.scale_mid
        post = fitted.scale_post
        prediction = fitted.reconstruction.float()
        best_error = fitted.after_error
        if config.functional_candidate_words_per_pass > 0:
            fit_residual = fitted_fit_residual
            held_residual = fitted_held_residual
            fit_error = fitted_fit_error
            held_error = fitted_held_error
        accepted_words += pass_accepted
        sign_updates += pass_sign_updates
        accepted_outer_passes += 1

    if config.functional_table_bit_passes:
        assert isinstance(codebook, ProductSignCodebook)
        assert indices is not None
        assert fit_inputs is not None and held_inputs is not None
        (
            right,
            codebook,
            prediction,
            best_error,
            fit_residual,
            fit_error,
            held_residual,
            held_error,
            table_bit_candidates_evaluated,
            accepted_table_bit_flips,
            table_bit_sign_updates,
        ) = _refine_product_table_bits(
            target32,
            left,
            right,
            pre,
            mid,
            post,
            input_weight,
            output_weight,
            codebook,
            indices,
            free_rows=free_rows,
            fit_inputs=fit_inputs,
            held_inputs=held_inputs,
            config=config,
        )
    if codebook is not None:
        assert indices is not None
        _validate_payload(right, codebook, indices, positions, free_rows)
    return SignWordPayloadSearchResult(
        right.to(right_binary.dtype),
        indices,
        positions,
        pre,
        mid,
        post,
        prediction,
        before_error,
        best_error,
        accepted_outer_passes,
        candidate_words_evaluated,
        codebook_patterns_evaluated,
        selected_words,
        accepted_words,
        sign_updates,
        functional_candidates_ranked,
        fit_error_before,
        fit_error,
        held_error_before,
        held_error,
        codebook,
        table_bit_candidates_evaluated,
        accepted_table_bit_flips,
        table_bit_sign_updates,
    )
