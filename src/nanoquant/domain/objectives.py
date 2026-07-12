"""Executable reconstruction objectives and Hessian utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from .metrics import dense_hessian_squared_error, weighted_squared_error


@dataclass(frozen=True, slots=True)
class DiagonalObjective:
    input_importance: torch.Tensor
    output_importance: torch.Tensor
    epsilon: float = 1e-12

    def weighted_error(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return weighted_squared_error(target, prediction, self.input_importance, self.output_importance)

    def normalized_error(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        norm = self.weighted_error(target, torch.zeros_like(target))
        return self.weighted_error(target, prediction) / norm.clamp_min(self.epsilon)

    def transform_for_factorizer(self, weight: torch.Tensor) -> torch.Tensor:
        return (
            weight
            * self.output_importance.detach().sqrt().reshape(-1, 1)
            * self.input_importance.detach().sqrt().reshape(1, -1)
        )


@dataclass(frozen=True, slots=True)
class DenseHessianObjective:
    covariance: torch.Tensor
    output_importance: torch.Tensor
    epsilon: float = 1e-12

    def weighted_error(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return dense_hessian_squared_error(target, prediction, self.covariance, self.output_importance)

    def normalized_error(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        norm = self.weighted_error(target, torch.zeros_like(target))
        return self.weighted_error(target, prediction) / norm.clamp_min(self.epsilon)

    def transform_for_factorizer(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        chol = regularized_cholesky(self.covariance)
        return (weight.float() @ chol) * self.output_importance.detach().float().sqrt().reshape(-1, 1), chol


@dataclass(frozen=True, slots=True)
class BlockDiagonalObjective:
    blocks: tuple[torch.Tensor, ...]
    block_size: int
    output_importance: torch.Tensor


@dataclass(frozen=True, slots=True)
class LowRankDiagonalObjective:
    diagonal: torch.Tensor
    factors: torch.Tensor
    output_importance: torch.Tensor


def regularize_covariance(
    covariance: torch.Tensor, damp_fraction: float = 0.01, identity_shrinkage: float = 0.0, diagonal_blend: float = 0.0
) -> torch.Tensor:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not 0 <= identity_shrinkage <= 1 or not 0 <= diagonal_blend <= 1 or damp_fraction < 0:
        raise ValueError("invalid covariance regularization parameter")
    value = covariance.detach().float().clone()
    value = 0.5 * (value + value.mT)
    diagonal = torch.diag(torch.diagonal(value))
    value = (1 - diagonal_blend) * value + diagonal_blend * diagonal
    mean_diagonal = torch.diagonal(value).mean().clamp_min(torch.finfo(value.dtype).eps)
    identity = torch.eye(value.shape[0], dtype=value.dtype, device=value.device) * mean_diagonal
    value = (1 - identity_shrinkage) * value + identity_shrinkage * identity
    value.diagonal().add_(damp_fraction * mean_diagonal)
    return value


def regularized_cholesky(covariance: torch.Tensor, *, jitter_attempts: int = 5) -> torch.Tensor:
    value = covariance.detach().float().clone()
    value = 0.5 * (value + value.mT)
    scale = torch.diagonal(value).mean().abs().clamp_min(torch.finfo(value.dtype).eps)
    for attempt in range(jitter_attempts + 1):
        jitter = 0 if attempt == 0 else scale * (10.0 ** (attempt - 7))
        trial = value.clone()
        trial.diagonal().add_(jitter)
        factor, info = torch.linalg.cholesky_ex(trial)
        if int(info.max()) == 0:
            return cast(torch.Tensor, factor)
    raise ValueError("covariance is not positive definite after regularization")


def whiten(weight: torch.Tensor, cholesky: torch.Tensor) -> torch.Tensor:
    return weight.float() @ cholesky.float()


def unwhiten(weight: torch.Tensor, cholesky: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, torch.linalg.solve_triangular(cholesky.float().mT, weight.float().mT, upper=True).mT)
