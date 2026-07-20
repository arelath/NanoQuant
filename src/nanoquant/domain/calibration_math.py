"""Pure robust activation-statistics accumulation matching legacy semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

LEGACY_CALIBRATION_CHUNK_TOKENS = 512


def _row_chunks(rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(
        rows[start : start + LEGACY_CALIBRATION_CHUNK_TOKENS]
        for start in range(0, rows.shape[0], LEGACY_CALIBRATION_CHUNK_TOKENS)
    )


def robust_tau(tensor: torch.Tensor, percentile: float = 0.999, pre_scale: float = 1.0) -> torch.Tensor:
    rows = tensor.detach().flatten(0, -2)
    if rows.shape[0] == 0:
        return rows.new_zeros((), dtype=torch.float32)
    norms = torch.cat(
        tuple(torch.linalg.vector_norm(chunk.float() * pre_scale, dim=1) for chunk in _row_chunks(rows))
    )
    count = max(1, int(norms.numel() * (1 - percentile)))
    return torch.topk(norms, count).values[-1]


def activation_square_mean(
    tensor: torch.Tensor, *, pre_scale: float = 1.0, post_scale: float = 1.0, clip_tau: torch.Tensor | None = None
) -> torch.Tensor:
    rows = tensor.detach().flatten(0, -2)
    if rows.shape[0] == 0:
        raise ValueError("cannot accumulate an empty activation batch")
    total = torch.zeros(rows.shape[-1], dtype=torch.float32, device="cpu")
    threshold = None if clip_tau is None else clip_tau.to(device=rows.device, dtype=torch.float32)
    for chunk in _row_chunks(rows):
        values = chunk.float() * pre_scale
        if threshold is not None:
            norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
            values = values * torch.clamp(threshold / (norms + 1e-8), max=1.0)
        total.add_(values.square().sum(dim=0).cpu())
    total.div_(rows.shape[0])
    return total * post_scale


@dataclass(slots=True)
class OnlineClippedAccumulator:
    width: int
    pre_scale: float = 1.0
    post_scale: float = 1.0
    percentile: float = 0.999
    total: torch.Tensor = field(init=False)
    global_max: torch.Tensor | None = field(init=False, default=None)
    batch_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.total = torch.zeros(self.width, dtype=torch.float32)
        self.global_max: torch.Tensor | None = None
        self.batch_count = 0

    def update(self, tensor: torch.Tensor) -> None:
        tau = robust_tau(tensor, self.percentile, self.pre_scale).cpu()
        if self.global_max is None:
            self.global_max = tau
        else:
            correction = torch.where(
                tau > self.global_max, (tau / (self.global_max + 1e-8)).square(), torch.ones_like(tau)
            )
            self.total.mul_(correction)
            self.global_max = torch.maximum(self.global_max, tau)
        self.total.add_(
            activation_square_mean(
                tensor, pre_scale=self.pre_scale, post_scale=self.post_scale, clip_tau=self.global_max
            ).cpu()
        )
        self.batch_count += 1

    def finalize(self) -> torch.Tensor:
        if self.batch_count == 0:
            raise ValueError("accumulator has no batches")
        return self.total / self.batch_count


@dataclass(slots=True)
class FixedClippedAccumulator:
    width: int
    threshold: torch.Tensor
    pre_scale: float = 1.0
    post_scale: float = 1.0
    total: torch.Tensor = field(init=False)
    batch_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.total = torch.zeros(self.width, dtype=torch.float32)
        self.batch_count = 0

    def update(self, tensor: torch.Tensor) -> None:
        self.total.add_(
            activation_square_mean(
                tensor, pre_scale=self.pre_scale, post_scale=self.post_scale, clip_tau=self.threshold
            ).cpu()
        )
        self.batch_count += 1

    def finalize(self) -> torch.Tensor:
        if self.batch_count == 0:
            raise ValueError("accumulator has no batches")
        return self.total / self.batch_count


@dataclass(slots=True)
class MeanAccumulator:
    """Unbiased feature mean over all activation rows."""

    width: int
    total: torch.Tensor = field(init=False)
    row_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("mean accumulator width must be positive")
        self.total = torch.zeros(self.width, dtype=torch.float64)

    def update(self, tensor: torch.Tensor) -> None:
        rows = tensor.detach().reshape(-1, tensor.shape[-1])
        if rows.shape[-1] != self.width or rows.shape[0] == 0:
            raise ValueError("mean accumulator received an incompatible activation batch")
        self.total.add_(rows.double().sum(dim=0).cpu())
        self.row_count += rows.shape[0]

    def finalize(self) -> torch.Tensor:
        if self.row_count == 0:
            raise ValueError("mean accumulator has no rows")
        return (self.total / self.row_count).float()


def shrink_importance(value: torch.Tensor, shrinkage: float) -> torch.Tensor:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    result = value.detach().clone()
    if 0 < shrinkage < 1 and result.numel():
        mean = result.mean()
        result.mul_(1 - shrinkage).add_(mean * shrinkage)
    return result


def weighted_group_output_importance(
    member_values: tuple[torch.Tensor, ...],
    member_multipliers: tuple[float, ...],
) -> torch.Tensor:
    """Apply squared member weights while preserving the stacked objective mean."""

    if not member_values or len(member_values) != len(member_multipliers):
        raise ValueError("group output importance requires aligned member values and multipliers")
    if any(value.ndim != 1 or value.numel() == 0 for value in member_values):
        raise ValueError("group output importance members must be non-empty vectors")
    if any(not math.isfinite(multiplier) or multiplier <= 0 for multiplier in member_multipliers):
        raise ValueError("group output importance multipliers must be finite and positive")
    baseline = torch.cat(member_values).contiguous()
    weighted = torch.cat(
        tuple(value * (multiplier**2) for value, multiplier in zip(member_values, member_multipliers, strict=True))
    ).contiguous()
    weighted_mean = weighted.mean()
    if not torch.isfinite(weighted_mean) or float(weighted_mean) <= 0:
        raise ValueError("group output importance weighted mean must be finite and positive")
    return weighted * (baseline.mean() / weighted_mean)
