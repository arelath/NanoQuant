"""Exact-objective coordinate search over free and corrected-code sign words."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .scale_fit import fit_scales, reconstruct
from .sign_word_codebook import (
    FullSignCodebook,
    apply_word_flips,
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


def _decode_candidate_word(
    table: torch.Tensor,
    index: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    candidate = table[index].clone()
    candidate[positions.to(torch.int64)] *= -1
    return candidate


def _validate_payload(
    right: torch.Tensor,
    codebook: FullSignCodebook,
    indices: torch.Tensor,
    flip_positions: torch.Tensor,
    free_rows: int,
) -> None:
    columns = right.shape[1]
    padded_columns = math.ceil(columns / 32) * 32
    decoded = decode_sign_codebook(indices, codebook, padded_columns)[:, :columns]
    decoded = apply_word_flips(decoded, flip_positions)
    if not torch.equal(decoded.to(right.dtype), right[free_rows:]):
        raise ValueError("codebook payload does not decode to the supplied factor")


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
    codebook: FullSignCodebook | None,
    right_indices: torch.Tensor | None,
    right_flip_positions: torch.Tensor | None,
    config: SignWordPayloadSearchConfig,
) -> SignWordPayloadSearchResult:
    """Refine right-factor words without changing their stored representation.

    Free rows receive their exact best arbitrary 32-sign coordinate proposal.
    Coded rows evaluate every table entry and the best fixed-count correction
    positions. The highest-gain bounded proposals are rescored sequentially,
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
    if codebook is None:
        if free_rows != rank or right_indices is not None or right_flip_positions is not None:
            raise ValueError("free-word search must expose every row and omit payload metadata")
        corrections_per_word = 0
    else:
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
    if codebook is not None:
        assert indices is not None and positions is not None
        _validate_payload(right, codebook, indices, positions, free_rows)

    prediction = reconstruct(left, right, pre, mid, post).float()
    best_error = _weighted_error(
        target32, prediction, input_weight, output_weight
    )
    before_error = best_error
    columns = target.shape[1]
    words = math.ceil(columns / 32)
    padded_columns = words * 32
    candidate_words_evaluated = 0
    codebook_patterns_evaluated = 0
    selected_words = 0
    accepted_words = 0
    sign_updates = 0
    accepted_outer_passes = 0

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
                batch_cost, batch_indices, batch_positions = (
                    _best_corrected_payload_candidates(
                        flat_linear[start:stop],
                        flat_quadratic[start:stop],
                        flat_right[start:stop],
                        codebook.entries,
                        corrections_per_word=corrections_per_word,
                        table_chunk_size=config.table_chunk_size,
                    )
                )
                flat_costs[start:stop] = batch_cost
                coded_indices[start:stop] = batch_indices
                coded_positions[start:stop] = batch_positions
            costs[free_rows:] = flat_costs.reshape(coded_rows, words)
            codebook_patterns_evaluated += coded_words * codebook.entry_count

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
                assert coded_indices is not None and coded_positions is not None
                coded_flat = (component - free_rows) * words + word
                candidate = _decode_candidate_word(
                    codebook.entries,
                    int(coded_indices[coded_flat]),
                    coded_positions[coded_flat],
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
            right[component, start:stop] = candidate
            prediction[:, start:stop] += component_vector[:, None] * (
                delta * pre[start:stop]
            )[None, :]
            if component >= free_rows:
                assert indices is not None and positions is not None
                assert coded_indices is not None and coded_positions is not None
                coded_flat = (component - free_rows) * words + word
                indices[component - free_rows, word] = coded_indices[coded_flat]
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
        if not math.isfinite(fitted.after_error) or improvement <= threshold:
            (
                right,
                indices,
                positions,
                pre,
                mid,
                post,
                prediction,
                best_error,
            ) = previous
            break
        pre = fitted.scale_pre
        mid = fitted.scale_mid
        post = fitted.scale_post
        prediction = fitted.reconstruction.float()
        best_error = fitted.after_error
        accepted_words += pass_accepted
        sign_updates += pass_sign_updates
        accepted_outer_passes += 1

    if codebook is not None:
        assert indices is not None and positions is not None
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
    )
