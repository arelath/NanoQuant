"""Fit independently scaled sign factors to successive reconstruction residuals."""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import replace

import torch

from nanoquant.application.quantization_stages import (
    FactorizationAttemptStage,
    MaterializedScaleFitStageRequest,
    ScaleFitStage,
)
from nanoquant.application.stages import StageContext, execute_stage
from nanoquant.domain.metrics import reconstruction_metrics
from nanoquant.domain.models import (
    ComponentRef,
    FactorizationRequest,
    FactorizationResult,
    ScaleFitRequest,
    ScaleFitResult,
    TrainableFactors,
)
from nanoquant.domain.residual_factorization import residual_stage_ranks
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.domain.seeds import logical_seed


def reconstruct_factors(factors: TrainableFactors, context: StageContext, device: str) -> torch.Tensor:
    """Materialize the sum without discarding independently fitted outer scales."""
    if factors.scales.mid is None:
        raise ValueError("factor term is missing its component scale")
    with ExitStack() as stack:
        values = [stack.enter_context(context.tensor_store.read(ref, device)) for ref in (
            factors.left_binary, factors.right_binary,
            factors.scales.pre, factors.scales.mid, factors.scales.post,
        )]
        prediction = reconstruct(*values)
    for stage in factors.residual_stages:
        prediction = prediction + reconstruct_factors(stage, context, device)
    return prediction


def fit_residual_stages(
    request: FactorizationRequest, context: StageContext, *,
    factor_stage: FactorizationAttemptStage, scale_stage: ScaleFitStage,
    fit_scales: bool, multiple: int, device: str,
) -> FactorizationResult:
    """Reuse the normal ADMM and scale stages, fitting each next term to the remaining error."""
    if request.product_codebook_right_free_rows is not None:
        raise ValueError("independent residual stages cannot use product coding")
    started = time.perf_counter()
    ranks = residual_stage_ranks(request.residual_weight.spec.shape[0], request.residual_weight.spec.shape[1],
                                 request.rank, multiple=multiple)
    results: list[FactorizationResult] = []
    fits: list[ScaleFitResult | None] = []
    target = request.residual_weight
    with context.tensor_store.read(target, device) as value:
        original = value.clone()
    prediction = torch.zeros_like(original)
    for index, rank in enumerate(ranks):
        context.cancellation.raise_if_cancelled()
        stage_request = replace(
            request, residual_weight=target, rank=rank,
            logical_seed=(request.logical_seed if index == 0 else logical_seed(
                request.logical_seed, "residual-factor-stage", request.layer.block.index,
                request.layer.path, index,
            )),
            generator_state=request.generator_state if index == 0 else None,
        )
        result = execute_stage(factor_stage, stage_request, context)
        fitted = None
        if fit_scales:
            fitted = execute_stage(scale_stage, MaterializedScaleFitStageRequest(
                ScaleFitRequest(request.layer, target, result.factors, request.objective, None),
                request.objective.input_importance, request.objective.output_importance,
            ), context)
            result = replace(result, factors=replace(result.factors, scales=fitted.scales))
        results.append(result)
        fits.append(fitted)
        prediction = prediction + reconstruct_factors(result.factors, context, device)
        if index + 1 < len(ranks):
            target = context.tensor_store.put("residual-stage-target", {
                "weight": (original.float() - prediction.float()).to(original.dtype),
            })["weight"]
    with (
        context.tensor_store.read(request.objective.input_importance, device) as input_importance,
        context.tensor_store.read(request.objective.output_importance, device) as output_importance,
    ):
        metrics = reconstruction_metrics(original, prediction, input_importance, output_importance,
                                         objective_mode=request.objective.kind)
    first = results[0]
    return replace(
        first, producer=ComponentRef("independent-residual-factorization", "1"),
        rank=request.rank, factors=replace(first.factors, residual_stages=tuple(
            result.factors for result in results[1:]
        )), metrics=metrics, wall_seconds=time.perf_counter() - started,
        peak_workspace_bytes=max(result.peak_workspace_bytes for result in results),
        stage_factorizations=tuple(results), stage_scale_fits=tuple(fits),
    )
