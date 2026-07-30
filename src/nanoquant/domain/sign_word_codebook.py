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
from typing import cast

import torch

from .factorization import SCHEDULES, ADMMResult, ADMMTracePoint


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


SignCodebook = ProductSignCodebook | FullSignCodebook


@dataclass(frozen=True, slots=True)
class SignWordCodebookADMMResult:
    """Constrained factors plus their exact codebook representation."""

    factors: ADMMResult
    left_codebook: SignCodebook
    right_codebook: SignCodebook
    left_indices: torch.Tensor
    right_indices: torch.Tensor


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
    decoded = codebook.entries[indices.to(torch.int64)].reshape(
        indices.shape[0],
        expected_words * 32,
    )
    return decoded[:, :columns].contiguous()


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
) -> SignCodebook:
    if mode == "full":
        entries = (
            torch.randint(
                0,
                2,
                (1 << index_bits, 32),
                device=device,
                generator=generator,
                dtype=torch.int8,
            )
            .to(dtype)
            .mul_(2)
            .sub_(1)
        )
        return FullSignCodebook(index_bits, entries)
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


def _project(
    value: torch.Tensor,
    codebook: SignCodebook | None,
    *,
    update_codebook: bool,
    inner_iterations: int,
    generator: torch.Generator,
    epsilon: float,
    assignment_batch_words: int,
) -> tuple[
    torch.Tensor,
    SignCodebook | None,
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
        return magnitude * _sign(value), None, None
    weighted = value.float() * magnitude.float()
    if isinstance(codebook, ProductSignCodebook):
        decoded, indices, product_updated = _assign_product_words(
            weighted,
            codebook,
            update=update_codebook,
            batch_words=assignment_batch_words,
        )
        updated: SignCodebook = product_updated
    else:
        decoded, indices, full_updated = _assign_full_words(
            weighted,
            codebook,
            update=update_codebook,
            batch_words=assignment_batch_words,
        )
        updated = full_updated
    return magnitude * decoded.to(value.dtype), updated, indices


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


def codebook_index_metrics(result: SignWordCodebookADMMResult) -> dict[str, dict[str, float | int]]:
    """Summarize actual fixed-width codebook utilization."""

    return {
        "left": _index_metrics(result.left_indices, result.left_codebook.index_bits),
        "right": _index_metrics(result.right_indices, result.right_codebook.index_bits),
    }


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
    assignment_batch_words: int = 65_536,
    codebook_mode: str = "product",
    epsilon: float = 1e-12,
) -> SignWordCodebookADMMResult:
    """Jointly fit over-complete factors constrained to fixed-width codebooks."""

    if weight.ndim != 2 or rank <= 0:
        raise ValueError("weight must be a matrix and rank positive")
    if input_importance.numel() != weight.shape[1] or output_importance.numel() != weight.shape[0]:
        raise ValueError("importance dimensions do not match weight")
    if index_bits <= 0 or index_bits % 2:
        raise ValueError("index bits must be positive and even")
    if codebook_mode not in {"product", "full"}:
        raise ValueError("codebook mode must be 'product' or 'full'")
    if (
        outer_iterations <= 0
        or inner_iterations <= 0
        or convergence_check_interval <= 0
        or codebook_update_interval <= 0
    ):
        raise ValueError("iteration settings must be positive")
    if not 0 <= codebook_freeze_fraction <= 1:
        raise ValueError("codebook freeze fraction must lie in [0, 1]")
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
    left_codebook = _random_codebook(
        index_bits,
        weight.device,
        dtype,
        generator,
        codebook_mode,
    )
    right_codebook = _random_codebook(
        index_bits,
        weight.device,
        dtype,
        generator,
        codebook_mode,
    )
    left_projected, left_codebook_value, left_indices_value = _project(
        left,
        left_codebook,
        update_codebook=True,
        inner_iterations=inner_iterations,
        generator=generator,
        epsilon=epsilon,
        assignment_batch_words=assignment_batch_words,
    )
    right_projected, right_codebook_value, right_indices_value = _project(
        right,
        right_codebook,
        update_codebook=True,
        inner_iterations=inner_iterations,
        generator=generator,
        epsilon=epsilon,
        assignment_batch_words=assignment_batch_words,
    )
    left_codebook = cast(SignCodebook, left_codebook_value)
    right_codebook = cast(SignCodebook, right_codebook_value)
    left_indices = cast(torch.Tensor, left_indices_value)
    right_indices = cast(torch.Tensor, right_indices_value)
    left_dual = left - left_projected
    right_dual = right - right_projected
    trace: list[ADMMTracePoint] = []
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
        update_ceiling = math.floor(outer_iterations * codebook_freeze_fraction)
        update = (
            (iteration + 1) % codebook_update_interval == 0
            and iteration + 1 <= update_ceiling
        )
        left_projected, left_codebook_value, left_indices_value = _project(
            left + left_dual,
            left_codebook,
            update_codebook=update,
            inner_iterations=inner_iterations,
            generator=generator,
            epsilon=epsilon,
            assignment_batch_words=assignment_batch_words,
        )
        right_projected, right_codebook_value, right_indices_value = _project(
            right + right_dual,
            right_codebook,
            update_codebook=update,
            inner_iterations=inner_iterations,
            generator=generator,
            epsilon=epsilon,
            assignment_batch_words=assignment_batch_words,
        )
        left_codebook = cast(SignCodebook, left_codebook_value)
        right_codebook = cast(SignCodebook, right_codebook_value)
        left_indices = cast(torch.Tensor, left_indices_value)
        right_indices = cast(torch.Tensor, right_indices_value)
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
        completed = iteration + 1
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
    left_binary = decode_sign_codebook(left_indices, left_codebook, rank).to(dtype)
    right_binary = decode_sign_codebook(right_indices, right_codebook, weight.shape[1]).to(dtype)
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
        left_indices.contiguous(),
        right_indices.contiguous(),
    )
