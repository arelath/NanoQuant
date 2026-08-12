"""Analysis math for equal-rate grouped middle-core factor probes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GroupedMiddleCoreResult:
    component_indices: torch.Tensor
    scale_pre: torch.Tensor
    core: torch.Tensor
    scale_post: torch.Tensor
    reconstruction: torch.Tensor
    diagonal_bits: int
    grouped_bits: int
    before_error: float
    after_error: float
    accepted: bool


def grouped_rank_at_or_below_diagonal_rate(
    original_rank: int,
    out_features: int,
    in_features: int,
    *,
    group_size: int = 2,
    scale_bits: int = 16,
) -> tuple[int, int, int]:
    """Return the largest complete grouped rank no larger than diagonal rate."""

    if (
        original_rank <= 0
        or out_features <= 0
        or in_features <= 0
        or group_size <= 1
        or scale_bits <= 0
    ):
        raise ValueError("grouped middle-core rate dimensions must be positive")
    factor_extent = out_features + in_features
    diagonal_bits = original_rank * (factor_extent + scale_bits)
    grouped_component_bits = factor_extent + group_size * scale_bits
    grouped_rank = diagonal_bits // grouped_component_bits
    grouped_rank -= grouped_rank % group_size
    if grouped_rank < group_size:
        raise ValueError("diagonal rate cannot fund one complete middle-core group")
    grouped_bits = grouped_rank * grouped_component_bits
    return grouped_rank, diagonal_bits, grouped_bits


def grouped_dense_reconstruction(
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    core: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    if core.ndim != 2 or core.shape != (left.shape[1], right.shape[0]):
        raise ValueError("middle core differs from the factor rank")
    return ((left.float() * post.float()[:, None]) @ core.float()) @ (
        right.float() * pre.float()[None, :]
    )


def _weighted_error(
    target: torch.Tensor,
    candidate: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> float:
    return float(
        (
            (candidate.float() - target.float()).square()
            * output_importance.float()[:, None]
            * input_importance.float()[None, :]
        ).sum()
    )


def _group_pairs(rank: int, group_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    left_indexes = []
    right_indexes = []
    for start in range(0, rank, group_size):
        for left in range(start, start + group_size):
            for right in range(start, start + group_size):
                left_indexes.append(left)
                right_indexes.append(right)
    return (
        torch.tensor(left_indexes, device=device, dtype=torch.long),
        torch.tensor(right_indexes, device=device, dtype=torch.long),
    )


def _greedy_pair_order(score: torch.Tensor) -> torch.Tensor:
    if score.ndim != 2 or score.shape[0] != score.shape[1] or score.shape[0] % 2:
        raise ValueError("group pairing score must be an even square matrix")
    rank = score.shape[0]
    rows, columns = torch.triu_indices(rank, rank, offset=1, device=score.device)
    order = torch.argsort(score[rows, columns], descending=True, stable=True)
    used = torch.zeros(rank, dtype=torch.bool, device=score.device)
    paired: list[int] = []
    for position in order.tolist():
        left = int(rows[position])
        right = int(columns[position])
        if not bool(used[left]) and not bool(used[right]):
            paired.extend((left, right))
            used[left] = True
            used[right] = True
            if len(paired) == rank:
                break
    if len(paired) != rank:
        remaining = torch.nonzero(~used, as_tuple=False).flatten().tolist()
        paired.extend(int(value) for value in remaining)
    return torch.tensor(paired, device=score.device, dtype=torch.long)


def _ordered_components(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    grouped_rank: int,
    pairing: str,
    epsilon: float,
) -> torch.Tensor:
    strongest = torch.topk(mid.abs(), grouped_rank).indices
    if pairing == "magnitude":
        return strongest[torch.argsort(mid.abs()[strongest], descending=True, stable=True)]
    if pairing == "fixed":
        return torch.sort(strongest).values
    selected_left = left[:, strongest]
    selected_right = right[strongest]
    scaled_left = selected_left * post[:, None]
    scaled_right = selected_right * pre[None, :]
    left_gram = scaled_left.mT @ (scaled_left * output_importance[:, None])
    weighted_right = scaled_right * input_importance.sqrt()[None, :]
    right_gram = weighted_right @ weighted_right.mT
    if pairing == "coupling":
        score = (left_gram * right_gram).abs()
    elif pairing == "residual":
        diagonal = (left * post[:, None]) @ (right * mid[:, None] * pre[None, :])
        residual = target - diagonal
        cross = scaled_left.mT @ (
            residual * input_importance[None, :] * output_importance[:, None]
        )
        rhs = cross @ scaled_right.mT
        denominator = (
            left_gram.diagonal()[:, None] * right_gram.diagonal()[None, :]
        ).clamp_min(epsilon)
        directed = rhs.square() / denominator
        score = directed + directed.mT
    else:
        raise ValueError(f"unsupported grouped middle-core pairing: {pairing}")
    score.diagonal().fill_(-torch.inf)
    return strongest[_greedy_pair_order(score)]


def _fit_core(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    group_size: int,
    epsilon: float,
) -> torch.Tensor:
    scaled_left = left * post[:, None]
    scaled_right = right * pre[None, :]
    left_gram = scaled_left.mT @ (scaled_left * output_importance[:, None])
    weighted_right = scaled_right * input_importance.sqrt()[None, :]
    right_gram = weighted_right @ weighted_right.mT
    left_indexes, right_indexes = _group_pairs(left.shape[1], group_size, target.device)
    system = left_gram[left_indexes[:, None], left_indexes[None, :]] * right_gram[
        right_indexes[:, None], right_indexes[None, :]
    ]
    system = 0.5 * (system + system.mT)
    cross = scaled_left.mT @ (
        target * input_importance[None, :] * output_importance[:, None]
    )
    rhs = (cross[left_indexes] * scaled_right[right_indexes]).sum(dim=1)
    ridge = torch.clamp(system.diagonal().mean().abs() * 1e-6, min=epsilon)
    system.diagonal().add_(ridge)
    cholesky, info = torch.linalg.cholesky_ex(system, upper=False)
    coefficients = (
        torch.cholesky_solve(rhs[:, None], cholesky, upper=False).squeeze(1)
        if int(info.item()) == 0
        else torch.linalg.lstsq(system, rhs[:, None]).solution.squeeze(1)
    )
    core = torch.zeros(
        (left.shape[1], left.shape[1]),
        device=target.device,
        dtype=torch.float32,
    )
    core[left_indexes, right_indexes] = torch.nan_to_num(coefficients)
    return core


def _fit_post(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    core: torch.Tensor,
    input_importance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    base = (left @ core) @ (right * pre[None, :])
    numerator = (base * target * input_importance[None, :]).sum(dim=1)
    denominator = (base.square() * input_importance[None, :]).sum(dim=1).clamp_min(epsilon)
    return torch.nan_to_num(numerator / denominator)


def _fit_pre(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    core: torch.Tensor,
    post: torch.Tensor,
    output_importance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    base = ((left * post[:, None]) @ core) @ right
    numerator = (base * target * output_importance[:, None]).sum(dim=0)
    denominator = (base.square() * output_importance[:, None]).sum(dim=0).clamp_min(epsilon)
    return torch.nan_to_num(numerator / denominator)


def fit_equal_rate_grouped_middle_core(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    group_size: int = 2,
    alternating_passes: int = 2,
    scale_bits: int = 16,
    storage_dtype: torch.dtype = torch.bfloat16,
    protected_columns: torch.Tensor | None = None,
    pairing: str = "magnitude",
    epsilon: float = 1e-8,
) -> GroupedMiddleCoreResult:
    """Fit a block-diagonal core after reducing rank to the diagonal bit rate."""

    if alternating_passes < 0 or epsilon <= 0:
        raise ValueError("grouped middle-core fit settings are invalid")
    if target.ndim != 2 or left_binary.shape[0] != target.shape[0] or right_binary.shape[1] != target.shape[1]:
        raise ValueError("grouped middle-core factors differ from the target")
    original_rank = left_binary.shape[1]
    if right_binary.shape[0] != original_rank or scale_mid.numel() != original_rank:
        raise ValueError("grouped middle-core factor ranks differ")
    grouped_rank, diagonal_bits, grouped_bits = grouped_rank_at_or_below_diagonal_rate(
        original_rank,
        target.shape[0],
        target.shape[1],
        group_size=group_size,
        scale_bits=scale_bits,
    )
    # Component permutation is representation-preserving. Retain the strongest
    # diagonal components when the grouped rate requires dropping a few ranks.
    pre = scale_pre.detach().float().reshape(-1).clone()
    post = scale_post.detach().float().reshape(-1).clone()
    full_left = torch.sign(left_binary.detach().float())
    full_right = torch.sign(right_binary.detach().float())
    full_mid = scale_mid.detach().float().reshape(-1)
    target32 = target.detach().float()
    input_weight = input_importance.detach().float().reshape(-1).clamp_min(epsilon)
    output_weight = output_importance.detach().float().reshape(-1).clamp_min(epsilon)
    selected = _ordered_components(
        target32,
        full_left,
        full_right,
        pre,
        full_mid,
        post,
        input_weight,
        output_weight,
        grouped_rank,
        pairing,
        epsilon,
    )
    left = full_left[:, selected].contiguous()
    right = full_right[selected].contiguous()
    mid = full_mid[selected]
    core = torch.diag(mid)
    protected = None if protected_columns is None else protected_columns.detach().long().reshape(-1)
    if protected is not None:
        pre[protected] = 0

    diagonal_reconstruction = (
        (full_left * scale_post.detach().float().reshape(-1, 1))
        @ (
            full_right
            * full_mid.reshape(-1, 1)
            * scale_pre.detach().float().reshape(1, -1)
        )
    )
    before_error = _weighted_error(
        target32,
        diagonal_reconstruction.to(storage_dtype).float(),
        input_weight,
        output_weight,
    )
    best_pre = pre.clone()
    best_post = post.clone()
    best_core = core.clone()
    best_reconstruction = grouped_dense_reconstruction(left, right, pre, core, post).to(storage_dtype).float()
    best_error = _weighted_error(target32, best_reconstruction, input_weight, output_weight)
    for _ in range(alternating_passes):
        core = _fit_core(
            target32,
            left,
            right,
            pre,
            post,
            input_weight,
            output_weight,
            group_size=group_size,
            epsilon=epsilon,
        )
        stored_pre = pre.to(storage_dtype).float()
        stored_post = post.to(storage_dtype).float()
        stored_core = core.to(storage_dtype).float()
        candidate = grouped_dense_reconstruction(
            left,
            right,
            stored_pre,
            stored_core,
            stored_post,
        ).to(storage_dtype).float()
        error = _weighted_error(target32, candidate, input_weight, output_weight)
        if math.isfinite(error) and error < best_error:
            best_error = error
            best_pre = stored_pre
            best_post = stored_post
            best_core = stored_core
            best_reconstruction = candidate
        post = _fit_post(target32, left, right, pre, core, input_weight, epsilon)
        pre = _fit_pre(target32, left, right, core, post, output_weight, epsilon)
        if protected is not None:
            pre[protected] = 0
    return GroupedMiddleCoreResult(
        selected.cpu(),
        best_pre,
        best_core,
        best_post,
        best_reconstruction,
        diagonal_bits,
        grouped_bits,
        before_error,
        best_error,
        best_error < before_error,
    )


__all__ = [
    "GroupedMiddleCoreResult",
    "fit_equal_rate_grouped_middle_core",
    "grouped_dense_reconstruction",
    "grouped_rank_at_or_below_diagonal_rate",
]
