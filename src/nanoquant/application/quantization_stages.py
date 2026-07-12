"""Resident outlier, factorization-attempt, and scale-fit stages."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from nanoquant.application.stages import StageContext
from nanoquant.config.schema import ADMMConfig, ScaleFitConfig
from nanoquant.domain.factorization import factorize_admm
from nanoquant.domain.metrics import reconstruction_metrics
from nanoquant.domain.models import (
    ComponentRef,
    ConvergenceMetrics,
    FactorizationRequest,
    FactorizationResult,
    OutlierSelectionRequest,
    OutlierSelectionResult,
    ScaleFitRequest,
    ScaleFitResult,
    ScaleState,
    StatisticSummary,
    TensorRef,
    TrainableFactors,
)
from nanoquant.domain.outliers import (
    fisher_scores,
    remove_columns,
    residual_probe_scores,
    select_top_columns,
    store_outlier_values,
)
from nanoquant.domain.planning import outlier_bit_cost
from nanoquant.domain.scale_fit import fit_scales, reconstruct
from nanoquant.domain.stages import HostInventory, ResourceEstimate, ValidationFinding, ValidationReport


def _summary(value: torch.Tensor) -> StatisticSummary:
    return StatisticSummary(
        float(value.min()) if value.numel() else 0.0,
        float(value.max()) if value.numel() else 0.0,
        float(value.float().mean()) if value.numel() else 0.0,
        float((value == 0).float().mean()) if value.numel() else 1.0,
        int((~torch.isfinite(value)).sum()),
    )


class OutlierSelectionStage:
    name = "select-outliers"
    version = "2"

    def __init__(self, *, device: str = "cpu", residual_probe_iterations: int = 20) -> None:
        if residual_probe_iterations <= 0:
            raise ValueError("residual probe iterations must be positive")
        self.device = device
        self.residual_probe_iterations = residual_probe_iterations

    def estimate(self, request: OutlierSelectionRequest, host: HostInventory) -> ResourceEstimate:
        elements = 1
        for dimension in request.source_weight.spec.shape:
            elements *= dimension
        return ResourceEstimate(peak_cpu_bytes=elements * 12, bytes_read=elements * 4)

    def execute(self, request: OutlierSelectionRequest, context: StageContext) -> OutlierSelectionResult:
        context.cancellation.raise_if_cancelled()
        with context.executor.device_scope(self.device):
            with (
                context.tensor_store.read(request.source_weight, self.device) as weight_value,
                context.tensor_store.read(request.objective.input_importance, self.device) as input_value,
                context.tensor_store.read(request.objective.output_importance, self.device) as output_value,
            ):
                weight, input_importance, output_importance = (
                    weight_value,
                    input_value.float(),
                    output_value.float(),
                )
                if request.plan.selector == "none" or request.plan.count == 0:
                    indices = torch.empty(0, dtype=torch.int64, device=self.device)
                elif request.plan.selector == "fisher":
                    indices = select_top_columns(
                        fisher_scores(weight, input_importance, output_importance), request.plan.count
                    )
                elif request.plan.selector == "residual":

                    def probe(value: torch.Tensor, rank: int, generator: torch.Generator) -> torch.Tensor:
                        return factorize_admm(
                            value,
                            input_importance,
                            output_importance,
                            rank,
                            generator,
                            outer_iterations=self.residual_probe_iterations,
                            inner_iterations=3,
                        ).reconstruction

                    scores = residual_probe_scores(
                        weight,
                        request.probe_rank,
                        input_importance,
                        output_importance,
                        probe,
                        torch.Generator(device=self.device).manual_seed(request.logical_seed),
                    )
                    indices = select_top_columns(scores, request.plan.count)
                else:
                    raise ValueError(f"unsupported outlier selector: {request.plan.selector}")
                residual, raw_values = remove_columns(weight, indices)
                stored_values, scales = store_outlier_values(raw_values, request.plan.storage_dtype)
                tensors = {"indices": indices.to(torch.int64), "values": stored_values, "residual_weight": residual}
                if scales is not None:
                    tensors["scales"] = scales
                refs = context.tensor_store.put("outlier-selection", tensors)
        bits = {"bfloat16": 16, "float16": 16, "int8": 8}.get(request.plan.storage_dtype, 16)
        cost = outlier_bit_cost(weight.shape[0], indices.numel(), value_bits=bits)
        context.events.emit(
            self.name,
            "info",
            "outliers.selected",
            layer=str(request.layer),
            count=indices.numel(),
            selector=request.plan.selector,
            bits=cost.total,
        )
        return OutlierSelectionResult(
            1,
            ComponentRef(self.name, self.version),
            request.layer,
            refs["indices"],
            refs["values"],
            refs.get("scales"),
            refs["residual_weight"],
            _summary(indices.float()),
            cost,
        )

    def validate(self, result: OutlierSelectionResult, context: StageContext) -> ValidationReport:
        findings = () if result.bit_cost.total >= 0 else (ValidationFinding("OUT001", "negative bit cost"),)
        return ValidationReport(findings)


class FactorizationAttemptStage:
    name = "factorize-attempt"
    version = "3"

    def __init__(self, admm: ADMMConfig | None = None, *, device: str = "cpu") -> None:
        self.admm = admm or ADMMConfig(outer_iterations=400)
        self.device = device

    def estimate(self, request: FactorizationRequest, host: HostInventory) -> ResourceEstimate:
        output, inputs = request.source_weight.spec.shape
        return ResourceEstimate(
            peak_cpu_bytes=(output * inputs * 16) + request.rank * (output + inputs) * 16,
            bytes_read=output * inputs * 8,
        )

    def execute(self, request: FactorizationRequest, context: StageContext) -> FactorizationResult:
        started = time.perf_counter()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        with context.executor.device_scope(self.device):
            with (
                context.tensor_store.read(request.residual_weight, self.device) as residual_value,
                context.tensor_store.read(request.objective.input_importance, self.device) as input_value,
                context.tensor_store.read(request.objective.output_importance, self.device) as output_value,
            ):
                residual = residual_value
                input_importance = input_value.float()
                output_importance = output_value.float()
                result = factorize_admm(
                    residual,
                    input_importance,
                    output_importance,
                    request.rank,
                    torch.Generator(device=self.device).manual_seed(request.logical_seed),
                    outer_iterations=self.admm.outer_iterations,
                    inner_iterations=self.admm.inner_iterations,
                    regularization=self.admm.regularization,
                    penalty_schedule=self.admm.penalty_schedule,
                    convergence_check_interval=self.admm.convergence_check_interval,
                    early_stop_tolerance=self.admm.early_stop_tolerance,
                )
                metrics = reconstruction_metrics(
                    residual,
                    result.reconstruction,
                    input_importance,
                    output_importance,
                    objective_mode=request.objective.kind,
                    latent_prediction=result.left_latent @ result.right_latent,
                )
                refs = context.tensor_store.put(
                    "factorization-attempt",
                    {
                        "left_latent": result.left_latent,
                        "right_latent": result.right_latent,
                        "left_binary": result.left_binary,
                        "right_binary": result.right_binary,
                        "scale_pre": result.scale_pre,
                        "scale_mid": result.scale_mid,
                        "scale_post": result.scale_post,
                    },
                )
        factors = TrainableFactors(
            refs["left_latent"],
            refs["right_latent"],
            refs["left_binary"],
            refs["right_binary"],
            ScaleState(refs["scale_pre"], refs["scale_mid"], refs["scale_post"]),
        )
        convergence = ConvergenceMetrics(
            result.iterations_completed,
            result.stopped_early,
            result.trace[-1].primal_residual if result.trace else None,
            result.trace[-1].dual_residual if result.trace else None,
            None,
        )
        context.events.emit(
            self.name,
            "info",
            "factorization.attempt_completed",
            layer=str(request.layer),
            rank=request.rank,
            weighted_error=metrics.export_weighted_normalized_error,
        )
        return FactorizationResult(
            1,
            ComponentRef(self.name, self.version),
            request.layer,
            request.rank,
            factors,
            metrics,
            convergence,
            time.perf_counter() - started,
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.startswith("cuda")
            else self.estimate(request, HostInventory(0, 0, 0)).peak_cpu_bytes,
        )

    def validate(self, result: FactorizationResult, context: StageContext) -> ValidationReport:
        findings = []
        if result.rank <= 0:
            findings.append(ValidationFinding("FAC001", "rank is not positive"))
        if not torch.isfinite(torch.tensor(result.metrics.export_weighted_error)):
            findings.append(ValidationFinding("FAC002", "non-finite reconstruction metric"))
        return ValidationReport(tuple(findings))


@dataclass(frozen=True, slots=True)
class MaterializedScaleFitStageRequest:
    request: ScaleFitRequest
    input_importance: TensorRef
    output_importance: TensorRef


class ScaleFitStage:
    name = "fit-scales"
    version = "1"

    def __init__(self, config: ScaleFitConfig | None = None, *, device: str = "cpu") -> None:
        self.config = config or ScaleFitConfig()
        self.device = device

    def estimate(self, request: MaterializedScaleFitStageRequest, host: HostInventory) -> ResourceEstimate:
        output, inputs = request.request.target_weight.spec.shape
        return ResourceEstimate(peak_cpu_bytes=output * inputs * 12)

    def execute(self, request: MaterializedScaleFitStageRequest, context: StageContext) -> ScaleFitResult:
        item = request.request
        if item.factors.scales.mid is None:
            raise ValueError("scale-fit stage requires an explicit mid scale")
        with context.executor.device_scope(self.device):
            with (
                context.tensor_store.read(item.target_weight, self.device) as target,
                context.tensor_store.read(item.factors.left_binary, self.device) as left,
                context.tensor_store.read(item.factors.right_binary, self.device) as right,
                context.tensor_store.read(item.factors.scales.pre, self.device) as pre,
                context.tensor_store.read(item.factors.scales.mid, self.device) as mid,
                context.tensor_store.read(item.factors.scales.post, self.device) as post,
                context.tensor_store.read(request.input_importance, self.device) as input_importance,
                context.tensor_store.read(request.output_importance, self.device) as output_importance,
            ):
                protected = None
                if item.protected_columns is not None:
                    with context.tensor_store.read(item.protected_columns, self.device) as value:
                        protected = value.clone()
                original_prediction = reconstruct(left, right, pre, mid, post)
                fitted = fit_scales(
                    target,
                    left,
                    right,
                    pre,
                    mid,
                    post,
                    input_importance,
                    output_importance,
                    alternating_passes=self.config.alternating_passes,
                    epsilon=self.config.epsilon,
                    protected_columns=protected,
                    rollback_on_regression=self.config.rollback_on_regression,
                )
                refs = context.tensor_store.put(
                    "scale-fit",
                    {
                        "scale_pre": fitted.scale_pre,
                        "scale_mid": fitted.scale_mid,
                        "scale_post": fitted.scale_post,
                    },
                )
                before = reconstruction_metrics(target, original_prediction, input_importance, output_importance)
                after = reconstruction_metrics(target, fitted.reconstruction, input_importance, output_importance)
        return ScaleFitResult(
            ScaleState(refs["scale_pre"], refs["scale_mid"], refs["scale_post"]),
            before,
            after,
            fitted.accepted,
            fitted.rollback_reason,
        )

    def validate(self, result: ScaleFitResult, context: StageContext) -> ValidationReport:
        if result.accepted and result.after.export_weighted_error > result.before.export_weighted_error:
            return ValidationReport((ValidationFinding("SCL001", "accepted scale fit regressed"),))
        return ValidationReport()
