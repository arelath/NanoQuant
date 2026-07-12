"""Pure robust activation-statistics accumulation matching legacy semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch


def robust_tau(tensor: torch.Tensor, percentile: float = 0.999, pre_scale: float = 1.0) -> torch.Tensor:
    rows = tensor.detach().flatten(0, -2).float() * pre_scale
    norms = torch.linalg.vector_norm(rows, dim=1)
    if norms.numel() == 0:
        return cast(torch.Tensor, norms.new_zeros(()))
    count = max(1, int(norms.numel() * (1 - percentile)))
    return torch.topk(norms, count).values[-1]


def activation_square_mean(
    tensor: torch.Tensor, *, pre_scale: float = 1.0, post_scale: float = 1.0, clip_tau: torch.Tensor | None = None
) -> torch.Tensor:
    rows = tensor.detach().flatten(0, -2).float() * pre_scale
    if rows.shape[0] == 0:
        raise ValueError("cannot accumulate an empty activation batch")
    if clip_tau is not None:
        norms = torch.linalg.vector_norm(rows, dim=1, keepdim=True)
        rows = rows * torch.clamp(clip_tau.float() / (norms + 1e-8), max=1.0)
    return rows.square().mean(dim=0) * post_scale


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


def shrink_importance(value: torch.Tensor, shrinkage: float) -> torch.Tensor:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    result = value.detach().clone()
    if 0 < shrinkage < 1 and result.numel():
        mean = result.mean()
        result.mul_(1 - shrinkage).add_(mean * shrinkage)
    return result
