"""Pure salient-outlier selection and reconstruction."""

from __future__ import annotations

from collections.abc import Callable

import torch

ResidualProbeFactorizer = Callable[[torch.Tensor, int, torch.Generator], torch.Tensor]


def fisher_scores(
    weight: torch.Tensor, input_importance: torch.Tensor, output_importance: torch.Tensor | None = None
) -> torch.Tensor:
    scores = weight.detach().float().square() * input_importance.detach().float().reshape(1, -1)
    if output_importance is not None:
        scores = scores * output_importance.detach().float().reshape(-1, 1)
    return scores.sum(dim=0)


def select_top_columns(scores: torch.Tensor, count: int) -> torch.Tensor:
    if count < 0 or count > scores.numel():
        raise ValueError("outlier count is outside score vector")
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.topk(scores, count, sorted=True).indices.sort().values


def remove_columns(weight: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual = weight.detach().clone()
    values = residual[:, indices].clone()
    residual[:, indices] = 0
    return residual, values


def reconstruct_with_outliers(
    base: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, scales: torch.Tensor | None = None
) -> torch.Tensor:
    result = base.detach().clone()
    restored = values.detach().to(device=result.device, dtype=result.dtype)
    if scales is not None:
        restored = restored * scales.detach().to(device=result.device, dtype=result.dtype).reshape(1, -1)
    result[:, indices] += restored
    return result


def quantize_int8_columns(values: torch.Tensor, epsilon: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    scale = values.detach().float().abs().amax(dim=0).clamp_min(epsilon) / 127.0
    quantized = torch.round(values.detach().float() / scale.reshape(1, -1)).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def residual_probe_scores(
    weight: torch.Tensor,
    rank: int,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    factorizer: ResidualProbeFactorizer,
    generator: torch.Generator,
    covariance: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction = factorizer(weight.detach().clone(), rank, generator).detach().float()
    if prediction.shape != weight.shape:
        raise ValueError("residual probe factorizer returned the wrong shape")
    residual = prediction - weight.detach().float()
    output_weight = output_importance.detach().float().reshape(-1, 1)
    if covariance is None:
        return (residual.square() * output_weight * input_importance.detach().float().reshape(1, -1)).sum(dim=0)
    hessian = covariance.detach().float()
    if hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("covariance dimension does not match weight")
    residual_hessian = residual @ hessian
    reduction = 2 * residual * residual_hessian - residual.square() * hessian.diagonal().reshape(1, -1)
    return (reduction * output_weight).sum(dim=0)


def store_outlier_values(values: torch.Tensor, dtype: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    normalized = dtype.lower()
    if normalized in {"bfloat16", "bf16"}:
        return values.detach().to(torch.bfloat16), None
    if normalized in {"float16", "fp16"}:
        return values.detach().to(torch.float16), None
    if normalized == "int8":
        return quantize_int8_columns(values)
    raise ValueError(f"unsupported outlier storage dtype: {dtype}")
