"""Rank decomposition for additive independently scaled binary factors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nanoquant.domain.factorization import AdmmParameters, ADMMResult, factorize_admm_with_parameters
from nanoquant.domain.seeds import logical_seed


def residual_stage_ranks(
    out_features: int, in_features: int, total_rank: int, *, multiple: int = 1
) -> tuple[int, ...]:
    """Chunk funded components into physical-sized stages without capping their sum."""
    if min(out_features, in_features, total_rank, multiple) <= 0:
        raise ValueError("residual factor dimensions, rank and alignment must be positive")
    stage_limit = min(out_features, in_features) // multiple * multiple
    if stage_limit == 0:
        raise ValueError("matrix cannot fund an aligned residual factor stage")
    full, remainder = divmod(total_rank, stage_limit)
    return (stage_limit,) * full + ((remainder,) if remainder else ())


@dataclass(frozen=True, slots=True)
class ResidualAdmmResult:
    stages: tuple[ADMMResult, ...]
    reconstruction: torch.Tensor


def factorize_residual_admm(
    weight: torch.Tensor, input_importance: torch.Tensor, output_importance: torch.Tensor,
    rank: int, seed: int, parameters: AdmmParameters, *, multiple: int,
) -> ResidualAdmmResult:
    """ADMM-only response probe for the same additive representation used in production."""
    ranks = residual_stage_ranks(weight.shape[0], weight.shape[1], rank, multiple=multiple)
    prediction = torch.zeros_like(weight)
    stages = []
    for index, stage_rank in enumerate(ranks):
        generator = torch.Generator(device=weight.device).manual_seed(
            seed if index == 0 else logical_seed(seed, "residual-rank-probe", attempt=index)
        )
        result = factorize_admm_with_parameters(
            (weight.float() - prediction.float()).to(weight.dtype), input_importance, output_importance,
            stage_rank, generator, parameters,
        )
        prediction = prediction + result.reconstruction
        stages.append(result)
    return ResidualAdmmResult(tuple(stages), prediction)
