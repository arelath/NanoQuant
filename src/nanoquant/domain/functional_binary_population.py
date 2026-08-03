"""Deterministic, gauge-aware proposals around functionally tuned binary factors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FunctionalBinaryCandidate:
    label: str
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    canonical_hash: str
    left_flips: int
    right_flips: int
    proposal_score: float
    components: tuple[int, ...]


def _sign(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))


def canonical_binary_signs(
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove row, component, and column sign gauges for comparison only."""

    if left_binary.ndim != 2 or right_binary.ndim != 2:
        raise ValueError("binary factors must be matrices")
    if left_binary.shape[1] != right_binary.shape[0]:
        raise ValueError("binary factor ranks do not match")
    if not left_binary.numel() or not right_binary.numel():
        raise ValueError("binary factors must be non-empty")
    left = _sign(left_binary.detach()).cpu()
    right = _sign(right_binary.detach()).cpu()
    row_signs = left[:, 0].clone()
    left *= row_signs[:, None]
    component_signs = left[0].clone()
    left *= component_signs[None, :]
    right *= component_signs[:, None]
    column_signs = right[0].clone()
    right *= column_signs[None, :]
    return left, right


def canonical_binary_hash(left_binary: torch.Tensor, right_binary: torch.Tensor) -> str:
    """Hash a sign pair after exact gauge removal."""

    left, right = canonical_binary_signs(left_binary, right_binary)
    bits = torch.cat((left.reshape(-1), right.reshape(-1))).gt(0).to(torch.uint8)
    packed = torch.zeros((bits.numel() + 7) // 8, dtype=torch.uint8)
    for shift in range(8):
        selected = bits[shift::8]
        packed[: selected.numel()] |= selected << shift
    shape = f"{left.shape[0]}:{left.shape[1]}:{right.shape[1]}:".encode()
    return "sha256:" + hashlib.sha256(shape + packed.numpy().tobytes()).hexdigest()


def _ranked_components(
    left_score: torch.Tensor,
    right_score: torch.Tensor,
    left_confidence: torch.Tensor | None,
    right_confidence: torch.Tensor | None,
    seed: int,
) -> tuple[tuple[str, torch.Tensor], ...]:
    beneficial = (
        left_score.clamp_min(0).mean(dim=0)
        + right_score.clamp_min(0).mean(dim=1)
    )
    pools: list[tuple[str, torch.Tensor]] = [
        ("gradient", beneficial.argsort(descending=True)),
    ]
    if left_confidence is not None and right_confidence is not None:
        uncertainty = (
            left_confidence.abs().mean(dim=0)
            + right_confidence.abs().mean(dim=1)
        )
        pools.append(("low-confidence", uncertainty.argsort()))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pools.append(("coverage", torch.randperm(left_score.shape[1], generator=generator)))
    return tuple(pools)


def build_functional_binary_population(
    left_latent: torch.Tensor,
    right_latent: torch.Tensor,
    left_gradient: torch.Tensor,
    right_gradient: torch.Tensor,
    *,
    population_size: int,
    flips_per_factor: int,
    components_per_candidate: int = 1,
    seed: int = 0,
) -> tuple[FunctionalBinaryCandidate, ...]:
    """Build coupled component proposals ranked by the block-loss STE gradient.

    The first member is always the incumbent.  Remaining members flip rows and
    columns from the same rank component.  The portfolio alternates functional
    gradient, latent uncertainty, and deterministic coverage pools, and removes
    candidates that differ only by an exact sign gauge.
    """

    if (
        left_latent.ndim != 2
        or right_latent.ndim != 2
        or left_latent.shape != left_gradient.shape
        or right_latent.shape != right_gradient.shape
        or left_latent.shape[1] != right_latent.shape[0]
        or population_size <= 0
        or flips_per_factor <= 0
        or components_per_candidate <= 0
    ):
        raise ValueError("functional binary population settings or shapes are invalid")
    left = _sign(left_latent.detach()).cpu()
    right = _sign(right_latent.detach()).cpu()
    # A flip has first-order delta -2 * sign * gradient, so larger values are
    # the changes predicted to reduce functional loss most strongly.
    left_score = (left_latent.detach().sign() * left_gradient.detach()).float().cpu()
    right_score = (right_latent.detach().sign() * right_gradient.detach()).float().cpu()
    pools = _ranked_components(
        left_score,
        right_score,
        left_latent.detach().float().cpu(),
        right_latent.detach().float().cpu(),
        seed,
    )
    incumbent_hash = canonical_binary_hash(left, right)
    candidates = [
        FunctionalBinaryCandidate(
            "incumbent",
            left,
            right,
            incumbent_hash,
            0,
            0,
            0.0,
            (),
        )
    ]
    seen = {incumbent_hash}
    rank = left.shape[1]
    attempts = 0
    maximum_attempts = max(population_size * 12, rank * len(pools))
    while len(candidates) < population_size and attempts < maximum_attempts:
        source, order = pools[attempts % len(pools)]
        start = (attempts // len(pools)) % rank
        component_count = min(components_per_candidate, rank)
        components = tuple(
            int(order[(start + offset) % rank]) for offset in range(component_count)
        )
        proposal_left = left.clone()
        proposal_right = right.clone()
        score = 0.0
        left_budget = max(1, flips_per_factor // component_count)
        right_budget = max(1, flips_per_factor // component_count)
        for component in components:
            left_indices = left_score[:, component].topk(
                min(left_budget, left.shape[0])
            ).indices
            right_indices = right_score[component].topk(
                min(right_budget, right.shape[1])
            ).indices
            proposal_left[left_indices, component] *= -1
            proposal_right[component, right_indices] *= -1
            score += float(left_score[left_indices, component].sum())
            score += float(right_score[component, right_indices].sum())
        candidate_hash = canonical_binary_hash(proposal_left, proposal_right)
        if candidate_hash not in seen:
            seen.add(candidate_hash)
            candidates.append(
                FunctionalBinaryCandidate(
                    f"{source}-{len(candidates)}",
                    proposal_left,
                    proposal_right,
                    candidate_hash,
                    int((proposal_left != left).sum()),
                    int((proposal_right != right).sum()),
                    score,
                    components,
                )
            )
        attempts += 1
    return tuple(candidates)
