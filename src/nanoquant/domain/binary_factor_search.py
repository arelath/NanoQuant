"""Bounded direct search over NanoQuant's final binary factors."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

from nanoquant.domain.scale_fit import fit_scales, reconstruct

_JOINT_SCALE_SCREEN_BATCH_ELEMENTS = 16_000_000


@dataclass(frozen=True, slots=True)
class BinaryFactorSearchResult:
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    reconstruction: torch.Tensor
    before_error: float
    after_error: float
    accepted_outer_passes: int
    continuous_updates: int
    one_bit_updates: int
    codebook_updates: int
    variable_depth_updates: int
    pair_updates: int
    block_updates: int
    block_patterns_evaluated: int
    component_updates: int
    joint_updates: int
    joint_patterns_evaluated: int


@dataclass(slots=True)
class _VectorSearchStats:
    continuous_updates: int = 0
    one_bit_updates: int = 0
    codebook_updates: int = 0
    variable_depth_updates: int = 0
    pair_updates: int = 0
    block_updates: int = 0
    block_patterns_evaluated: int = 0
    component_updates: int = 0
    joint_updates: int = 0
    joint_patterns_evaluated: int = 0


def _sign(value: torch.Tensor) -> torch.Tensor:
    return (value >= 0).to(value.dtype).mul_(2).sub_(1)


def _scores(
    vectors: torch.Tensor,
    cross: torch.Tensor,
    gram: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha = (vectors * cross).sum(dim=1)
    gram_product = vectors @ gram
    beta = (vectors * gram_product).sum(dim=1).clamp_min(epsilon)
    return alpha.square() / beta, alpha, beta, gram_product


def _accept_vectors(
    vectors: torch.Tensor,
    scales: torch.Tensor,
    scores: torch.Tensor,
    candidate_vectors: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_alpha: torch.Tensor,
    candidate_beta: torch.Tensor,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    accepted = candidate_scores > scores + tolerance * scores.abs().clamp_min(1.0)
    if not bool(accepted.any()):
        return vectors, scales, scores, 0
    vectors = vectors.clone()
    scales = scales.clone()
    scores = scores.clone()
    vectors[accepted] = candidate_vectors[accepted]
    scales[accepted] = candidate_alpha[accepted] / candidate_beta[accepted]
    scores[accepted] = candidate_scores[accepted]
    return vectors, scales, scores, int(accepted.sum())


def _continuous_candidates(
    gram: torch.Tensor,
    cross: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    system = 0.5 * (gram + gram.mT)
    ridge = (system.diagonal().mean().abs() * 1e-6).clamp_min(epsilon)
    system = system.clone()
    system.diagonal().add_(ridge)
    factor, info = torch.linalg.cholesky_ex(system)
    solution = (
        torch.cholesky_solve(cross.mT, factor).mT
        if int(info.max()) == 0
        else torch.linalg.lstsq(system, cross.mT).solution.mT
    )
    return _sign(torch.nan_to_num(solution))


def _one_bit_pass(
    vectors: torch.Tensor,
    cross: torch.Tensor,
    gram: torch.Tensor,
    scales: torch.Tensor,
    scores: torch.Tensor,
    epsilon: float,
    tolerance: float,
    *,
    vector_weights: torch.Tensor | None = None,
    maximum_updates: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    _, alpha, beta, gram_product = _scores(vectors, cross, gram, epsilon)
    candidate_alpha = alpha[:, None] - 2.0 * vectors * cross
    candidate_beta = (
        beta[:, None]
        - 4.0 * vectors * gram_product
        + 4.0 * gram.diagonal()[None, :]
    ).clamp_min(epsilon)
    candidate_scores = candidate_alpha.square() / candidate_beta
    best_scores, indices = candidate_scores.max(dim=1)
    if maximum_updates is not None and maximum_updates < vectors.shape[0]:
        weights = (
            torch.ones_like(best_scores)
            if vector_weights is None
            else vector_weights.to(device=vectors.device, dtype=best_scores.dtype)
        )
        gains = (best_scores - scores) * weights
        selected = gains.topk(maximum_updates).indices
        active = torch.zeros(vectors.shape[0], dtype=torch.bool, device=vectors.device)
        active[selected] = True
        best_scores = torch.where(active, best_scores, scores)
    rows = torch.arange(vectors.shape[0], device=vectors.device)
    candidates = vectors.clone()
    candidates[rows, indices] *= -1
    return _accept_vectors(
        vectors,
        scales,
        scores,
        candidates,
        best_scores,
        candidate_alpha[rows, indices],
        candidate_beta[rows, indices],
        tolerance,
    )


def _pair_pass(
    vectors: torch.Tensor,
    cross: torch.Tensor,
    gram: torch.Tensor,
    scales: torch.Tensor,
    scores: torch.Tensor,
    pool_size: int,
    epsilon: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    rank = vectors.shape[1]
    pool_size = min(rank, pool_size)
    if pool_size < 2:
        return vectors, scales, scores, 0
    _, alpha, beta, gram_product = _scores(vectors, cross, gram, epsilon)
    single_alpha = alpha[:, None] - 2.0 * vectors * cross
    single_beta = (
        beta[:, None]
        - 4.0 * vectors * gram_product
        + 4.0 * gram.diagonal()[None, :]
    ).clamp_min(epsilon)
    single_scores = single_alpha.square() / single_beta
    pool = single_scores.topk(pool_size, dim=1).indices
    local_pairs = torch.tensor(
        tuple(itertools.combinations(range(pool_size), 2)),
        dtype=torch.long,
        device=vectors.device,
    )
    first = pool[:, local_pairs[:, 0]]
    second = pool[:, local_pairs[:, 1]]
    first_sign = vectors.gather(1, first)
    second_sign = vectors.gather(1, second)
    first_cross = cross.gather(1, first)
    second_cross = cross.gather(1, second)
    first_gradient = gram_product.gather(1, first)
    second_gradient = gram_product.gather(1, second)
    diagonal = gram.diagonal()
    first_diagonal = diagonal[first]
    second_diagonal = diagonal[second]
    interaction = gram[first, second]
    candidate_alpha = (
        alpha[:, None]
        - 2.0 * first_sign * first_cross
        - 2.0 * second_sign * second_cross
    )
    candidate_beta = (
        beta[:, None]
        - 4.0 * first_sign * first_gradient
        - 4.0 * second_sign * second_gradient
        + 4.0 * first_diagonal
        + 4.0 * second_diagonal
        + 8.0 * first_sign * second_sign * interaction
    ).clamp_min(epsilon)
    candidate_scores = candidate_alpha.square() / candidate_beta
    best_scores, choices = candidate_scores.max(dim=1)
    rows = torch.arange(vectors.shape[0], device=vectors.device)
    first_choice = first[rows, choices]
    second_choice = second[rows, choices]
    candidates = vectors.clone()
    candidates[rows, first_choice] *= -1
    candidates[rows, second_choice] *= -1
    return _accept_vectors(
        vectors,
        scales,
        scores,
        candidates,
        best_scores,
        candidate_alpha[rows, choices],
        candidate_beta[rows, choices],
        tolerance,
    )


def _codebook_pass(
    vectors: torch.Tensor,
    cross: torch.Tensor,
    gram: torch.Tensor,
    scales: torch.Tensor,
    scores: torch.Tensor,
    maximum_patterns: int,
    epsilon: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Transfer exact scale-profiled sign candidates between shared-Q rows."""

    if maximum_patterns == 0 or vectors.shape[0] < 2:
        return vectors, scales, scores, 0
    patterns, counts = torch.unique(vectors, dim=0, return_counts=True)
    if patterns.shape[0] > maximum_patterns:
        frequent_count = min(maximum_patterns // 2, patterns.shape[0])
        frequent = counts.topk(frequent_count).indices
        remaining = maximum_patterns - frequent_count
        stride = max(1, patterns.shape[0] // max(remaining, 1))
        diverse = torch.arange(0, patterns.shape[0], stride, device=vectors.device)[:remaining]
        selected = torch.unique(torch.cat((frequent, diverse)))
        if selected.numel() < maximum_patterns:
            mask = torch.ones(patterns.shape[0], dtype=torch.bool, device=vectors.device)
            mask[selected] = False
            supplement = torch.arange(patterns.shape[0], device=vectors.device)[mask][
                : maximum_patterns - selected.numel()
            ]
            selected = torch.cat((selected, supplement))
        patterns = patterns.index_select(0, selected[:maximum_patterns])
    denominator = ((patterns @ gram) * patterns).sum(dim=1).clamp_min(epsilon)
    numerator = cross @ patterns.mT
    candidate_scores = numerator.square() / denominator[None, :]
    best_scores, choices = candidate_scores.max(dim=1)
    rows = torch.arange(vectors.shape[0], device=vectors.device)
    candidates = patterns.index_select(0, choices)
    return _accept_vectors(
        vectors,
        scales,
        scores,
        candidates,
        best_scores,
        numerator[rows, choices],
        denominator.index_select(0, choices),
        tolerance,
    )


def _variable_depth_pass(
    vectors: torch.Tensor,
    cross: torch.Tensor,
    gram: torch.Tensor,
    scales: torch.Tensor,
    scores: torch.Tensor,
    chain_length: int,
    epsilon: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build uphill-capable locked-bit chains and commit the best prefix."""

    chain_length = min(chain_length, vectors.shape[1])
    if chain_length == 0:
        return vectors, scales, scores, 0
    state = vectors.clone()
    _, alpha, beta, gram_product = _scores(state, cross, gram, epsilon)
    locked = torch.zeros_like(state, dtype=torch.bool)
    best_vectors = vectors.clone()
    best_scores = scores.clone()
    best_alpha = alpha.clone()
    best_beta = beta.clone()
    rows = torch.arange(vectors.shape[0], device=vectors.device)
    diagonal = gram.diagonal()[None, :]
    for _ in range(chain_length):
        candidate_alpha = alpha[:, None] - 2.0 * state * cross
        candidate_beta = (
            beta[:, None] - 4.0 * state * gram_product + 4.0 * diagonal
        ).clamp_min(epsilon)
        candidate_scores = candidate_alpha.square() / candidate_beta
        candidate_scores.masked_fill_(locked, -torch.inf)
        choices = candidate_scores.argmax(dim=1)
        old_signs = state[rows, choices].clone()
        alpha = candidate_alpha[rows, choices]
        beta = candidate_beta[rows, choices]
        state[rows, choices] *= -1
        locked[rows, choices] = True
        gram_product -= 2.0 * old_signs[:, None] * gram.index_select(0, choices)
        current_scores = alpha.square() / beta
        improved = current_scores > best_scores
        if bool(improved.any()):
            best_vectors[improved] = state[improved]
            best_scores[improved] = current_scores[improved]
            best_alpha[improved] = alpha[improved]
            best_beta[improved] = beta[improved]
    return _accept_vectors(
        vectors,
        scales,
        scores,
        best_vectors,
        best_scores,
        best_alpha,
        best_beta,
        tolerance,
    )


def _patterns(bit_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(1 << bit_count, dtype=torch.int64, device=device)
    shifts = torch.arange(bit_count, dtype=torch.int64, device=device)
    return (((values[:, None] >> shifts) & 1).to(dtype) * 2.0) - 1.0


def _block_pass(
    vectors: torch.Tensor,
    cross: torch.Tensor,
    gram: torch.Tensor,
    scales: torch.Tensor,
    scores: torch.Tensor,
    vector_weights: torch.Tensor,
    target_energy: torch.Tensor,
    block_bits: int,
    hard_vectors: int,
    epsilon: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    block_bits = min(block_bits, vectors.shape[1])
    hard_vectors = min(hard_vectors, vectors.shape[0])
    if block_bits < 2 or hard_vectors == 0:
        return vectors, scales, scores, 0, 0
    current_scores, alpha, beta, gram_product = _scores(vectors, cross, gram, epsilon)
    one_alpha = alpha[:, None] - 2.0 * vectors * cross
    one_beta = (
        beta[:, None]
        - 4.0 * vectors * gram_product
        + 4.0 * gram.diagonal()[None, :]
    ).clamp_min(epsilon)
    one_scores = one_alpha.square() / one_beta
    hard = _hard_vector_indices(current_scores, vector_weights, target_energy, hard_vectors)
    signs = _patterns(block_bits, vectors.device, vectors.dtype)
    updated = vectors.clone()
    updated_scales = scales.clone()
    updated_scores = scores.clone()
    accepted = 0
    evaluated = 0
    for row in hard.tolist():
        selected = one_scores[row].topk(block_bits).indices
        candidates = vectors[row].expand(signs.shape[0], -1).clone()
        candidates[:, selected] = signs
        candidate_alpha = candidates @ cross[row]
        candidate_beta = ((candidates @ gram) * candidates).sum(dim=1).clamp_min(epsilon)
        candidate_scores = candidate_alpha.square() / candidate_beta
        choice = int(candidate_scores.argmax())
        evaluated += signs.shape[0]
        if float(candidate_scores[choice]) > float(scores[row]) + tolerance * max(abs(float(scores[row])), 1.0):
            updated[row] = candidates[choice]
            updated_scales[row] = candidate_alpha[choice] / candidate_beta[choice]
            updated_scores[row] = candidate_scores[choice]
            accepted += 1
    return updated, updated_scales, updated_scores, accepted, evaluated


def _hard_vector_indices(
    current_scores: torch.Tensor,
    vector_weights: torch.Tensor,
    target_energy: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Select vectors by their actual weighted residual after scale profiling."""

    residual = vector_weights * (target_energy - current_scores).clamp_min(0)
    return residual.topk(min(count, residual.numel())).indices


def _refine_vectors(
    vectors: torch.Tensor,
    cross: torch.Tensor,
    gram: torch.Tensor,
    scales: torch.Tensor,
    vector_weights: torch.Tensor,
    target_energy: torch.Tensor,
    *,
    continuous: bool,
    one_bit_passes: int,
    one_bit_vectors: int,
    codebook_passes: int,
    codebook_size: int,
    variable_depth_passes: int,
    variable_depth_length: int,
    pair_passes: int,
    pair_pool_size: int,
    block_bits: int,
    block_passes: int,
    hard_vectors: int,
    epsilon: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, _VectorSearchStats]:
    stats = _VectorSearchStats()
    scores, _, _, _ = _scores(vectors, cross, gram, epsilon)
    if continuous:
        candidates = _continuous_candidates(gram, cross, epsilon)
        candidate_scores, candidate_alpha, candidate_beta, _ = _scores(candidates, cross, gram, epsilon)
        vectors, scales, scores, count = _accept_vectors(
            vectors,
            scales,
            scores,
            candidates,
            candidate_scores,
            candidate_alpha,
            candidate_beta,
            tolerance,
        )
        stats.continuous_updates += count
    for _ in range(one_bit_passes):
        vectors, scales, scores, count = _one_bit_pass(
            vectors,
            cross,
            gram,
            scales,
            scores,
            epsilon,
            tolerance,
            vector_weights=vector_weights,
            maximum_updates=one_bit_vectors,
        )
        stats.one_bit_updates += count
        if count == 0:
            break
    for _ in range(codebook_passes):
        vectors, scales, scores, count = _codebook_pass(
            vectors,
            cross,
            gram,
            scales,
            scores,
            codebook_size,
            epsilon,
            tolerance,
        )
        stats.codebook_updates += count
        if count == 0:
            break
    for _ in range(variable_depth_passes):
        vectors, scales, scores, count = _variable_depth_pass(
            vectors,
            cross,
            gram,
            scales,
            scores,
            variable_depth_length,
            epsilon,
            tolerance,
        )
        stats.variable_depth_updates += count
        if count == 0:
            break
    for _ in range(pair_passes):
        vectors, scales, scores, count = _pair_pass(
            vectors,
            cross,
            gram,
            scales,
            scores,
            pair_pool_size,
            epsilon,
            tolerance,
        )
        stats.pair_updates += count
        if count == 0:
            break
    for _ in range(block_passes):
        vectors, scales, scores, count, evaluated = _block_pass(
            vectors,
            cross,
            gram,
            scales,
            scores,
            vector_weights,
            target_energy,
            block_bits,
            hard_vectors,
            epsilon,
            tolerance,
        )
        stats.block_updates += count
        stats.block_patterns_evaluated += evaluated
        if count == 0:
            break
    return vectors, scales, stats


def _component_replacement_sweep(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    component_limit: int,
    alternating_steps: int,
    epsilon: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Replace complete rank components while holding boundary scales fixed."""

    if component_limit == 0 or alternating_steps == 0:
        return left, right, mid, 0
    prediction = reconstruct(left, right, pre, mid, post).float()
    weighted_target = output_weight[:, None] * input_weight[None, :]
    error = float(((prediction - target).square() * weighted_target).sum())
    residual = target - prediction
    scaled_residual = residual * weighted_target * post[:, None] * pre[None, :]
    residual_responses = (left * (scaled_residual @ right.mT)).sum(dim=0) * mid
    component_energy = (
        mid.square()
        * (post.square() * output_weight).sum()
        * (pre.square() * input_weight).sum()
    )
    removal_cost = component_energy + 2.0 * residual_responses
    order = _component_candidate_order(
        mid,
        removal_cost,
        residual_responses.abs(),
        component_limit,
    )
    updates = 0
    for component in order.tolist():
        old_component = torch.outer(post * left[:, component], pre * right[component]) * mid[component]
        residual = target - (prediction - old_component)
        interaction = (
            residual
            * output_weight[:, None]
            * input_weight[None, :]
            * post[:, None]
            * pre[None, :]
        )
        if interaction.shape[1] <= 16:
            patterns = _patterns(interaction.shape[1], interaction.device, interaction.dtype)
            responses = interaction @ patterns.mT
            numerators = responses.abs().sum(dim=0)
            choice = int(numerators.argmax())
            best_right = patterns[choice]
            best_left = _sign(responses[:, choice])
            best_numerator = numerators[choice]
        elif interaction.shape[0] <= 16:
            patterns = _patterns(interaction.shape[0], interaction.device, interaction.dtype)
            responses = interaction.mT @ patterns.mT
            numerators = responses.abs().sum(dim=0)
            choice = int(numerators.argmax())
            best_left = patterns[choice]
            best_right = _sign(responses[:, choice])
            best_numerator = numerators[choice]
        else:
            row_strength = interaction.square().sum(dim=1)
            start_rows = row_strength.topk(min(8, interaction.shape[0])).indices
            starts = [
                right[component].clone(),
                _sign(interaction.sum(dim=0)),
                *[_sign(interaction[index]) for index in start_rows.tolist()],
            ]
            best_numerator = torch.zeros((), device=target.device)
            best_left = left[:, component]
            best_right = right[component]
            for start in starts:
                candidate_right = start
                candidate_left = left[:, component]
                for _ in range(alternating_steps):
                    candidate_left = _sign(interaction @ candidate_right)
                    candidate_right = _sign(interaction.mT @ candidate_left)
                numerator = candidate_left @ interaction @ candidate_right
                if float(numerator.abs()) > float(best_numerator.abs()):
                    best_numerator = numerator
                    best_left = candidate_left
                    best_right = candidate_right
        denominator = (
            (post.square() * output_weight).sum()
            * (pre.square() * input_weight).sum()
        ).clamp_min(epsilon)
        candidate_mid = best_numerator / denominator
        new_component = torch.outer(post * best_left, pre * best_right) * candidate_mid
        candidate_prediction = prediction - old_component + new_component
        candidate_error = float(((candidate_prediction - target).square() * weighted_target).sum())
        threshold = tolerance * max(abs(error), 1.0)
        if math.isfinite(candidate_error) and candidate_error < error - threshold:
            left = left.clone()
            right = right.clone()
            mid = mid.clone()
            left[:, component] = best_left
            right[component] = best_right
            mid[component] = candidate_mid
            prediction = candidate_prediction
            error = candidate_error
            updates += 1
    return left, right, mid, updates


def _component_candidate_order(
    mid: torch.Tensor,
    removal_cost: torch.Tensor,
    residual_alignment: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    """Interleave weak, strong, and residual-aligned component pools."""

    count = min(limit, mid.numel())
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=mid.device)
    pools = (
        removal_cost.argsort(),
        mid.abs().argsort(descending=True),
        residual_alignment.argsort(descending=True),
    )
    selected: list[int] = []
    seen: set[int] = set()
    for position in range(mid.numel()):
        for pool in pools:
            component = int(pool[position])
            if component in seen:
                continue
            selected.append(component)
            seen.add(component)
            if len(selected) == count:
                return torch.tensor(selected, dtype=torch.long, device=mid.device)
    return torch.tensor(selected, dtype=torch.long, device=mid.device)


def _canonicalize_sign_gauges(
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left = left.clone()
    right = right.clone()
    pre = pre.clone()
    post = post.clone()
    row_signs = left[:, 0].clone()
    left *= row_signs[:, None]
    post *= row_signs
    component_signs = left[0].clone()
    left *= component_signs[None, :]
    right *= component_signs[:, None]
    column_signs = right[0].clone()
    right *= column_signs[None, :]
    pre *= column_signs
    return left, right, pre, post


def _joint_scale_screen_batch(
    target: torch.Tensor,
    candidate_left: torch.Tensor,
    candidate_right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scale_passes: int,
    epsilon: float,
) -> torch.Tensor:
    """Rank a candidate batch after the same bounded pre/mid/post ALS fit."""

    count = candidate_left.shape[0]
    candidate_pre = pre[None, :].expand(count, -1).clone()
    candidate_mid = mid[None, :].expand(count, -1).clone()
    candidate_post = post[None, :].expand(count, -1).clone()
    candidate_prediction = torch.bmm(
        candidate_left * candidate_post[:, :, None],
        candidate_right * (candidate_mid[:, :, None] * candidate_pre[:, None, :]),
    )
    screen_errors = (
        (candidate_prediction - target[None]).square()
        * output_weight[None, :, None]
        * input_weight[None, None, :]
    ).sum(dim=(1, 2))
    for _screen_pass in range(scale_passes):
        base = torch.bmm(
            candidate_left,
            candidate_right * (candidate_mid[:, :, None] * candidate_pre[:, None, :]),
        )
        candidate_post = torch.nan_to_num(
            (base * target[None] * input_weight[None, None, :]).sum(dim=2)
            / (base.square() * input_weight[None, None, :]).sum(dim=2).clamp_min(epsilon)
        )
        base = torch.bmm(
            candidate_left * (candidate_post[:, :, None] * candidate_mid[:, None, :]),
            candidate_right,
        )
        candidate_pre = torch.nan_to_num(
            (base * target[None] * output_weight[None, :, None]).sum(dim=1)
            / (base.square() * output_weight[None, :, None]).sum(dim=1).clamp_min(epsilon)
        )
        scaled_left = candidate_left * candidate_post[:, :, None]
        scaled_right = candidate_right * candidate_pre[:, None, :]
        weighted_left = scaled_left * output_weight.sqrt()[None, :, None]
        weighted_right = scaled_right * input_weight.sqrt()[None, None, :]
        left_gram = torch.bmm(weighted_left.mT, weighted_left)
        right_gram = torch.bmm(weighted_right, weighted_right.mT)
        middle_system = left_gram * right_gram
        middle_system = 0.5 * (middle_system + middle_system.mT)
        middle_ridge = (
            middle_system.diagonal(dim1=-2, dim2=-1).abs().mean(dim=1) * 1e-6
        ).clamp_min(epsilon)
        middle_system.diagonal(dim1=-2, dim2=-1).add_(middle_ridge[:, None])
        weighted_target = target * output_weight[:, None] * input_weight[None, :]
        target_times_right = torch.bmm(
            weighted_target[None].expand(count, -1, -1),
            scaled_right.mT,
        )
        middle_rhs = (scaled_left * target_times_right).sum(dim=1)
        candidate_mid, info = torch.linalg.solve_ex(middle_system, middle_rhs[:, :, None])
        candidate_mid = torch.nan_to_num(candidate_mid.squeeze(2))
        candidate_mid[info != 0] = mid
        candidate_prediction = torch.bmm(
            candidate_left * candidate_post[:, :, None],
            candidate_right * (candidate_mid[:, :, None] * candidate_pre[:, None, :]),
        )
        current_errors = (
            (candidate_prediction - target[None]).square()
            * output_weight[None, :, None]
            * input_weight[None, None, :]
        ).sum(dim=(1, 2))
        screen_errors = torch.minimum(screen_errors, current_errors)
    return screen_errors


def _joint_bit_window(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    joint_bits: int,
    candidate_refits: int,
    scale_passes: int,
    screen_scale_passes: int,
    batch_size: int,
    selection_trial: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int, int]:
    """Enumerate a bounded window containing signs from both factors."""

    left, right, pre, post = _canonicalize_sign_gauges(left, right, pre, post)
    prediction = reconstruct(left, right, pre, mid, post).float()
    weighted_difference = (prediction - target).square() * output_weight[:, None] * input_weight[None, :]
    baseline_error = float(weighted_difference.sum())
    free_left = torch.cartesian_prod(
        torch.arange(1, left.shape[0], device=left.device),
        torch.arange(1, left.shape[1], device=left.device),
    )
    free_right = torch.cartesian_prod(
        torch.arange(1, right.shape[0], device=right.device),
        torch.arange(right.shape[1], device=right.device),
    )
    available = free_left.shape[0] + free_right.shape[0]
    if joint_bits == 0 or available == 0:
        return left, right, pre, mid, post, baseline_error, 0, 0
    joint_bits = min(joint_bits, available)

    scaled_right = right * (mid[:, None] * pre[None, :])
    left_gram = (scaled_right * input_weight[None, :]) @ scaled_right.mT
    left_cross = (target * input_weight[None, :]) @ scaled_right.mT
    left_scores, left_alpha, left_beta, left_product = _scores(left, left_cross, left_gram, epsilon)
    left_flip_scores = (
        (left_alpha[:, None] - 2.0 * left * left_cross).square()
        / (
            left_beta[:, None]
            - 4.0 * left * left_product
            + 4.0 * left_gram.diagonal()[None, :]
        ).clamp_min(epsilon)
    )
    left_margins = left_scores[:, None] - left_flip_scores

    scaled_left = left * (post[:, None] * mid[None, :])
    right_gram = scaled_left.mT @ (scaled_left * output_weight[:, None])
    right_cross = (scaled_left.mT @ (target * output_weight[:, None])).mT
    right_vectors = right.mT
    right_scores, right_alpha, right_beta, right_product = _scores(
        right_vectors, right_cross, right_gram, epsilon
    )
    right_flip_scores = (
        (right_alpha[:, None] - 2.0 * right_vectors * right_cross).square()
        / (
            right_beta[:, None]
            - 4.0 * right_vectors * right_product
            + 4.0 * right_gram.diagonal()[None, :]
        ).clamp_min(epsilon)
    )
    right_margins = right_scores[:, None] - right_flip_scores

    if available <= joint_bits:
        selected_left = free_left
        selected_right = free_right
    else:
        mode = selection_trial % 3
        if mode == 1:
            left_quota = min(free_left.shape[0], joint_bits - 1)
        elif mode == 2:
            left_quota = max(1, joint_bits - free_right.shape[0])
        else:
            left_quota = min(free_left.shape[0], joint_bits // 2)
        right_quota = min(free_right.shape[0], joint_bits - left_quota)
        left_quota = min(free_left.shape[0], joint_bits - right_quota)
        if selection_trial < 3:
            left_values = left_margins[free_left[:, 0], free_left[:, 1]]
            right_values = right_margins[free_right[:, 1], free_right[:, 0]]
        else:
            row_residual = weighted_difference.sum(dim=1)
            column_residual = weighted_difference.sum(dim=0)
            left_values = -row_residual[free_left[:, 0]]
            right_values = -column_residual[free_right[:, 1]]
        selected_left = free_left[left_values.topk(left_quota, largest=False).indices]
        selected_right = free_right[right_values.topk(right_quota, largest=False).indices]
    actual_bits = selected_left.shape[0] + selected_right.shape[0]
    patterns = _patterns(actual_bits, left.device, left.dtype)
    errors = torch.empty(patterns.shape[0], device=left.device)
    # Joint search is explicitly opt-in. Keep the scale-profiled screening
    # semantics at every matrix size and bound peak memory by reducing the
    # candidate batch, rather than switching large matrices to a weaker
    # fixed-scale objective. The old total-work threshold created a recall
    # cliff precisely for production-shaped owners.
    screen_batch_size = min(
        batch_size,
        max(1, _JOINT_SCALE_SCREEN_BATCH_ELEMENTS // max(1, target.numel())),
    )
    for start in range(0, patterns.shape[0], screen_batch_size):
        signs = patterns[start : start + screen_batch_size]
        count = signs.shape[0]
        candidate_left = left.expand(count, -1, -1).clone()
        candidate_right = right.expand(count, -1, -1).clone()
        split = selected_left.shape[0]
        if split > 0:
            candidate_left[:, selected_left[:, 0], selected_left[:, 1]] = signs[:, :split]
        if selected_right.shape[0] > 0:
            candidate_right[:, selected_right[:, 0], selected_right[:, 1]] = signs[:, split:]
        errors[start : start + count] = _joint_scale_screen_batch(
            target,
            candidate_left,
            candidate_right,
            pre,
            mid,
            post,
            input_weight,
            output_weight,
            max(screen_scale_passes, scale_passes),
            epsilon,
        )

    best_error = baseline_error
    best = (left, right, pre, mid, post)
    for choice in errors.topk(min(candidate_refits, errors.numel()), largest=False).indices.tolist():
        candidate_left = left.clone()
        candidate_right = right.clone()
        signs = patterns[choice]
        split = selected_left.shape[0]
        if split > 0:
            candidate_left[selected_left[:, 0], selected_left[:, 1]] = signs[:split]
        if selected_right.shape[0] > 0:
            candidate_right[selected_right[:, 0], selected_right[:, 1]] = signs[split:]
        fitted = fit_scales(
            target,
            candidate_left,
            candidate_right,
            pre,
            mid,
            post,
            input_weight,
            output_weight,
            alternating_passes=scale_passes,
        )
        if fitted.after_error < best_error:
            best_error = fitted.after_error
            best = (
                candidate_left,
                candidate_right,
                fitted.scale_pre,
                fitted.scale_mid,
                fitted.scale_post,
            )
    updates = int((best[0] != left).sum() + (best[1] != right).sum())
    return (*best, best_error, updates, patterns.shape[0])


def refine_binary_factors_separable(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    outer_passes: int = 4,
    scale_passes: int = 4,
    continuous_candidates: bool = True,
    one_bit_passes: int = 8,
    one_bit_fraction: float = 1.0,
    max_one_bit_vectors: int = 2**31 - 1,
    codebook_passes: int = 0,
    codebook_size: int = 512,
    variable_depth_passes: int = 0,
    variable_depth_length: int = 32,
    pair_passes: int = 2,
    pair_pool_size: int = 32,
    block_bits: int = 10,
    block_passes: int = 1,
    hard_fraction: float = 0.1,
    max_hard_vectors: int = 128,
    component_passes: int = 1,
    component_limit: int = 8,
    component_alternating_steps: int = 8,
    joint_passes: int = 0,
    joint_bits: int = 10,
    joint_candidate_refits: int = 4,
    joint_batch_size: int = 64,
    joint_screen_scale_passes: int = 4,
    epsilon: float = 1e-8,
    acceptance_tolerance: float = 1e-10,
) -> BinaryFactorSearchResult:
    """Improve final binary factors with scale-eliminated bounded neighborhoods.

    The representation and bit cost are unchanged. Every outer pass is checked
    against the exact separable weighted reconstruction objective and rolled
    back unless it improves that objective.
    """

    if target.ndim != 2 or left_binary.ndim != 2 or right_binary.ndim != 2:
        raise ValueError("binary-factor search target and factors must be matrices")
    rank = left_binary.shape[1]
    if (
        left_binary.shape[0] != target.shape[0]
        or right_binary.shape != (rank, target.shape[1])
        or scale_pre.numel() != target.shape[1]
        or scale_mid.numel() != rank
        or scale_post.numel() != target.shape[0]
        or input_importance.numel() != target.shape[1]
        or output_importance.numel() != target.shape[0]
    ):
        raise ValueError("binary-factor search dimensions do not match")
    if (
        outer_passes < 0
        or scale_passes < 0
        or one_bit_passes < 0
        or not 0.0 <= one_bit_fraction <= 1.0
        or max_one_bit_vectors < 0
        or codebook_passes < 0
        or codebook_size < 0
        or variable_depth_passes < 0
        or variable_depth_length < 0
        or pair_passes < 0
        or pair_pool_size < 0
        or block_bits < 0
        or block_bits > 16
        or block_passes < 0
        or not 0.0 <= hard_fraction <= 1.0
        or max_hard_vectors < 0
        or component_passes < 0
        or component_limit < 0
        or component_alternating_steps < 0
        or joint_passes < 0
        or joint_bits < 0
        or joint_bits > 20
        or joint_candidate_refits <= 0
        or joint_batch_size <= 0
        or joint_screen_scale_passes < 0
        or epsilon <= 0
        or acceptance_tolerance < 0
    ):
        raise ValueError("binary-factor search settings are invalid")

    target32 = target.detach().float()
    left = _sign(left_binary.detach().float()).clone()
    right = _sign(right_binary.detach().float()).clone()
    pre = scale_pre.detach().float().reshape(-1).clone()
    mid = scale_mid.detach().float().reshape(-1).clone()
    post = scale_post.detach().float().reshape(-1).clone()
    input_weight = input_importance.detach().float().reshape(-1).clamp_min(epsilon)
    output_weight = output_importance.detach().float().reshape(-1).clamp_min(epsilon)
    initial_prediction = reconstruct(left, right, pre, mid, post)
    initial_error = float(
        ((initial_prediction - target32).square() * output_weight[:, None] * input_weight[None, :]).sum()
    )
    fitted = fit_scales(
        target32,
        left,
        right,
        pre,
        mid,
        post,
        input_weight,
        output_weight,
        alternating_passes=scale_passes,
    )
    pre, mid, post = fitted.scale_pre, fitted.scale_mid, fitted.scale_post
    best_error = fitted.after_error
    best_prediction = fitted.reconstruction.float()
    totals = _VectorSearchStats()
    accepted_outer_passes = 0
    for _ in range(outer_passes):
        previous = (left.clone(), right.clone(), pre.clone(), mid.clone(), post.clone(), best_prediction.clone())
        scaled_right = right * (mid[:, None] * pre[None, :])
        left_gram = (scaled_right * input_weight[None, :]) @ scaled_right.mT
        left_cross = (target32 * input_weight[None, :]) @ scaled_right.mT
        hard_left = min(max_hard_vectors, math.ceil(left.shape[0] * hard_fraction))
        one_bit_left = min(max_one_bit_vectors, math.ceil(left.shape[0] * one_bit_fraction))
        left, post, left_stats = _refine_vectors(
            left,
            left_cross,
            left_gram,
            post,
            output_weight,
            (target32.square() * input_weight[None, :]).sum(dim=1),
            continuous=continuous_candidates,
            one_bit_passes=one_bit_passes,
            one_bit_vectors=one_bit_left,
            codebook_passes=codebook_passes,
            codebook_size=codebook_size,
            variable_depth_passes=variable_depth_passes,
            variable_depth_length=variable_depth_length,
            pair_passes=pair_passes,
            pair_pool_size=pair_pool_size,
            block_bits=block_bits,
            block_passes=block_passes,
            hard_vectors=hard_left,
            epsilon=epsilon,
            tolerance=acceptance_tolerance,
        )

        scaled_left = left * (post[:, None] * mid[None, :])
        right_gram = scaled_left.mT @ (scaled_left * output_weight[:, None])
        right_cross = (scaled_left.mT @ (target32 * output_weight[:, None])).mT
        hard_right = min(max_hard_vectors, math.ceil(right.shape[1] * hard_fraction))
        one_bit_right = min(max_one_bit_vectors, math.ceil(right.shape[1] * one_bit_fraction))
        right_vectors, pre, right_stats = _refine_vectors(
            right.mT.contiguous(),
            right_cross,
            right_gram,
            pre,
            input_weight,
            (target32.square() * output_weight[:, None]).sum(dim=0),
            continuous=continuous_candidates,
            one_bit_passes=one_bit_passes,
            one_bit_vectors=one_bit_right,
            codebook_passes=codebook_passes,
            codebook_size=codebook_size,
            variable_depth_passes=variable_depth_passes,
            variable_depth_length=variable_depth_length,
            pair_passes=pair_passes,
            pair_pool_size=pair_pool_size,
            block_bits=block_bits,
            block_passes=block_passes,
            hard_vectors=hard_right,
            epsilon=epsilon,
            tolerance=acceptance_tolerance,
        )
        right = right_vectors.mT.contiguous()
        component_updates = 0
        for _ in range(component_passes):
            left, right, mid, updates = _component_replacement_sweep(
                target32,
                left,
                right,
                pre,
                mid,
                post,
                input_weight,
                output_weight,
                component_limit,
                component_alternating_steps,
                epsilon,
                acceptance_tolerance,
            )
            component_updates += updates
            if updates == 0:
                break
        joint_updates = 0
        joint_patterns = 0
        for selection_trial in range(joint_passes):
            left, right, pre, mid, post, _joint_error, updates, evaluated = _joint_bit_window(
                target32,
                left,
                right,
                pre,
                mid,
                post,
                input_weight,
                output_weight,
                joint_bits,
                joint_candidate_refits,
                scale_passes,
                joint_screen_scale_passes,
                joint_batch_size,
                selection_trial,
                epsilon,
            )
            joint_updates += updates
            joint_patterns += evaluated
        candidate = fit_scales(
            target32,
            left,
            right,
            pre,
            mid,
            post,
            input_weight,
            output_weight,
            alternating_passes=scale_passes,
        )
        improvement = best_error - candidate.after_error
        threshold = acceptance_tolerance * max(abs(best_error), 1.0)
        if not math.isfinite(candidate.after_error) or improvement <= threshold:
            left, right, pre, mid, post, best_prediction = previous
            break
        pre, mid, post = candidate.scale_pre, candidate.scale_mid, candidate.scale_post
        best_prediction = candidate.reconstruction.float()
        best_error = candidate.after_error
        accepted_outer_passes += 1
        for stats in (left_stats, right_stats):
            totals.continuous_updates += stats.continuous_updates
            totals.one_bit_updates += stats.one_bit_updates
            totals.codebook_updates += stats.codebook_updates
            totals.variable_depth_updates += stats.variable_depth_updates
            totals.pair_updates += stats.pair_updates
            totals.block_updates += stats.block_updates
            totals.block_patterns_evaluated += stats.block_patterns_evaluated
        totals.component_updates += component_updates
        totals.joint_updates += joint_updates
        totals.joint_patterns_evaluated += joint_patterns

    return BinaryFactorSearchResult(
        left.to(left_binary.dtype),
        right.to(right_binary.dtype),
        pre,
        mid,
        post,
        best_prediction.to(target.dtype),
        initial_error,
        best_error,
        accepted_outer_passes,
        totals.continuous_updates,
        totals.one_bit_updates,
        totals.codebook_updates,
        totals.variable_depth_updates,
        totals.pair_updates,
        totals.block_updates,
        totals.block_patterns_evaluated,
        totals.component_updates,
        totals.joint_updates,
        totals.joint_patterns_evaluated,
    )
