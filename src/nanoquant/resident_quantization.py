"""Auditable resident quantization composition for pinned Transformers snapshots."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.assembly import assemble_frozen_model
from nanoquant.application.calibration import MaterializedLayerCalibration, calibrate_block, calibrate_causal_model
from nanoquant.application.calibration_artifacts import (
    PersistedCalibration,
    PersistedObjectives,
    build_objectives,
    persist_calibration,
)
from nanoquant.application.layers import (
    BlockEditor,
    LayerFreezer,
    TrainableFactorizedLinear,
    freeze_block_auxiliary_parameters,
    restore_block_auxiliary_parameters,
)
from nanoquant.application.loss_snapshots import BlockLossRecorder
from nanoquant.application.planning import PersistedPlan, PlanningRequest, build_quantization_plan, persist_plan
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.application.quantization_stages import (
    FactorizationAttemptStage,
    MaterializedScaleFitStageRequest,
    OutlierSelectionStage,
    ScaleFitStage,
)
from nanoquant.application.reconstruction_report import render_reconstruction_tables
from nanoquant.application.retry_loop import AcceptedFactorization, run_factorization_attempts
from nanoquant.application.stages import StageContext, execute_stage
from nanoquant.application.tuning import TuningRequest, post_block_refit, tune_factorized, tune_non_factorized
from nanoquant.config.codec import canonical_json, from_dict, to_dict
from nanoquant.config.schema import (
    ADMMConfig,
    AllocationStrategy,
    ObjectiveConfig,
    OutlierConfig,
    RankAllocationConfig,
    RankBoundsConfig,
    RankRetryConfig,
    RetryThresholdConfig,
    ScaleFitConfig,
)
from nanoquant.domain.metrics import reconstruction_metrics
from nanoquant.domain.models import (
    ArtifactRef,
    BlockResult,
    CalibrationStats,
    CheckpointInventory,
    DatasetIdentity,
    FactorizationRequest,
    FactorizationResult,
    FrozenBlockState,
    FrozenModelResult,
    FrozenNanoQuantState,
    FrozenOutlierState,
    LayerId,
    LayerPlan,
    LayerResult,
    ModelInventory,
    ObjectiveSpec,
    OutlierSelectionRequest,
    OutlierSelectionResult,
    QuantizationPlan,
    ScaleFitRequest,
    ScaleFitResult,
    TensorRef,
)
from nanoquant.domain.outliers import reconstruct_with_outliers
from nanoquant.domain.runs import BudgetState
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import (
    CommitIdentity,
    commit_block,
    commit_layer,
    load_block_activations,
    load_committed_block,
    load_committed_layer,
    retire_block_activations,
)
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.progress import ProgressJournal
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore

RESIDENT_ALGORITHM_VERSION = 17


@contextmanager
def _legacy_cuda_numerics() -> Iterator[None]:
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@dataclass(frozen=True, slots=True)
class ResidentQuantizationRequest:
    snapshot: Path
    output: Path
    source: str
    revision: str
    token_ids: torch.Tensor | tuple[tuple[int, ...], ...]
    device: str = "cuda"
    target_bpw: float = 1.0
    rank_multiple: int = 32
    allocation_strategy: AllocationStrategy = AllocationStrategy.SENSITIVITY
    rank_floor_fraction: float = 0.5
    rank_ceiling_fraction: float = 4.5
    rank_sensitivity_alpha: float = 0.5
    rank_edge_boost: float = 0.0
    layer_order: tuple[str, ...] = ()
    admm: ADMMConfig = ADMMConfig(outer_iterations=1, inner_iterations=1)
    outliers: OutlierConfig = OutlierConfig()
    scale_fit: ScaleFitConfig = ScaleFitConfig(enabled=False)
    factorized_tuning_epochs: int = 0
    factorized_tuning_batch_size: int = 8
    factorized_tuning_learning_rate: float = 1e-5
    nonfactorized_tuning_epochs: int = 0
    nonfactorized_tuning_epochs_by_layer: tuple[int, ...] = ()
    nonfactorized_tuning_batch_size: int = 8
    nonfactorized_tuning_learning_rate: float = 1e-4
    nonfactorized_tuning_early_stop_relative_tolerance: float | None = None
    post_block_refit_epochs: int = 0
    post_block_refit_batch_size: int = 8
    post_block_refit_learning_rate: float = 1e-5
    tuning_microbatch_size: int | None = None
    legacy_tuning_seed_reset: bool = False
    activation_retention: str = "rolling"
    seed: int = 0
    verify_hashes: bool = True
    interrupt_after_layer_commits: int | None = None
    interrupt_after_block_commits: int | None = None
    block_forward_batch_size: int = 8
    quality_token_ids: torch.Tensor | tuple[tuple[int, ...], ...] | None = None
    calibration_method: str = "forward_only"
    calibration_shrinkage: float = 0.0
    calibration_batch_size: int = 1
    precomputed_calibration: ArtifactRef | None = None
    precomputed_objectives: ArtifactRef | None = None
    precomputed_plan: ArtifactRef | None = None
    restore_completed_blocks: bool = True
    evaluate_inline_quality: bool = True
    defer_layer_loss_snapshots: bool = False


@dataclass(frozen=True, slots=True)
class ResidentQuantizationResult:
    inventory: ModelInventory
    plan: QuantizationPlan
    identity: CommitIdentity
    frozen_model: FrozenModelResult
    blocks: tuple[BlockResult, ...]
    report: ArtifactRef
    reference_nll: float
    compressed_nll: float
    logit_mse: float
    argmax_agreement: float
    peak_device_bytes: int
    peak_host_bytes: int
    artifact_bytes: int
    elapsed_seconds: float
    reused_commit_count: int


@dataclass(frozen=True, slots=True)
class ResidentFactorizationSliceResult:
    layer: LayerResult | None
    identity: CommitIdentity
    elapsed_seconds: float
    peak_device_bytes: int
    remaining_layers: int


_FACTOR_SLICE_SOURCE_CACHE: dict[
    tuple[str, str, str, bool, tuple[str, ...]],
    tuple[SafetensorsModelSource, CheckpointInventory, ModelInventory],
] = {}


def _checkpoint_dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    if not isinstance(value, str):
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(value, torch.float32)


def _decoder_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    base = getattr(model, "model", None)
    values = getattr(base, "layers", None)
    if not isinstance(values, nn.ModuleList):
        raise TypeError("model does not expose a supported decoder layer stack")
    return tuple(values)


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape != target.shape:
        raise ValueError("MSE prediction and target shapes differ")
    if prediction.ndim == 0:
        return float((prediction.detach().float() - target.detach().float()).square())
    total = torch.zeros((), device=prediction.device)
    for index in range(prediction.shape[0]):
        total += (prediction[index].detach().float() - target[index].detach().float()).square().sum()
    return float(total / prediction.numel())


def _weighted_mse(prediction: torch.Tensor, target: torch.Tensor, importance: torch.Tensor) -> float:
    if importance.ndim != 1 or importance.shape[0] != prediction.shape[-1]:
        raise ValueError("block output importance does not match hidden width")
    weights = importance.to(device=prediction.device, dtype=torch.float32)
    total = torch.zeros((), device=prediction.device)
    for index in range(prediction.shape[0]):
        error = prediction[index].detach().float() - target[index].detach().float()
        total += (error.square() * weights).sum()
    return float(total / prediction.numel())


def _self_reference_weighted_mse(value: torch.Tensor, importance: torch.Tensor) -> float:
    # x - x == 0.0 for every finite IEEE-754 value, so the weighted MSE of a tensor
    # against itself is always exactly 0.0 unless it contains non-finite entries.
    if bool(torch.isfinite(value).all()):
        return 0.0
    return _weighted_mse(value, value, importance)


def _block_loss(
    adapter: Any,
    block: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    output_importance: torch.Tensor,
    metadata: dict[str, object],
    batch_size: int,
) -> float:
    block_device = next(iter(block.parameters()), None)
    device = inputs.device if block_device is None else block_device.device
    weights = output_importance.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        squared_error = torch.zeros((), device=device)
        elements = 0
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            input_batch = inputs[start:end].to(device, non_blocking=True)
            prediction = adapter.run_block(block, input_batch, **metadata)
            target = targets[start:end].to(device, non_blocking=True)
            squared_error += ((prediction.float() - target.float()).square() * weights).sum()
            elements += target.numel()
        return float(squared_error / elements)


def _run_block_batched(
    adapter: Any,
    block: nn.Module,
    inputs: torch.Tensor,
    metadata: dict[str, object],
    batch_size: int,
    storage_device: str | torch.device | None = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("block forward batch size must be positive")
    block_parameter = next(iter(block.parameters()), None)
    compute_device = inputs.device if block_parameter is None else block_parameter.device
    destination = inputs.device if storage_device is None else torch.device(storage_device)
    result: torch.Tensor | None = None
    for start in range(0, inputs.shape[0], batch_size):
        end = min(start + batch_size, inputs.shape[0])
        input_batch = inputs[start:end].to(compute_device, non_blocking=True)
        output = adapter.run_block(block, input_batch, **metadata)
        if result is None:
            result = torch.empty(
                (inputs.shape[0], *output.shape[1:]),
                device=destination,
                dtype=output.dtype,
            )
        result[start:end].copy_(output)
    if result is None:
        raise ValueError("cannot run a block over empty inputs")
    return result


def _run_prefix_batched(
    adapter: Any,
    model: nn.Module,
    tokens: torch.Tensor,
    batch_size: int,
    storage_device: str | torch.device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("prefix batch size must be positive")
    destination = torch.device(storage_device)
    result: torch.Tensor | None = None
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            end = min(start + batch_size, tokens.shape[0])
            output = adapter.run_prefix(model, tokens[start:end])
            if result is None:
                result = torch.empty(
                    (tokens.shape[0], *output.shape[1:]),
                    device=destination,
                    dtype=output.dtype,
                )
            result[start:end].copy_(output)
    if result is None:
        raise ValueError("cannot run a prefix over empty tokens")
    return result


def _nll(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    prediction = logits[:, :-1].float().reshape(-1, logits.shape[-1])
    target = tokens[:, 1:].reshape(-1)
    return float(torch.nn.functional.cross_entropy(prediction, target))


def _artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _token_tensor(value: torch.Tensor | tuple[tuple[int, ...], ...], device: str) -> torch.Tensor:
    result = value.detach().clone() if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("resident quantization tokens must be a non-empty rectangular rank-2 tensor")
    return result.to(device=device, dtype=torch.long)


def _layer_type_multiplier(path: str) -> float:
    if path.endswith(("gate_proj", "up_proj", "fc1")):
        return 1.4
    if path.endswith(("down_proj", "fc2", "v_proj")):
        return 1.2
    if path.endswith(("q_proj", "k_proj")):
        return 0.8
    return 1.0


def _module_at_path(module: nn.Module, path: str) -> nn.Module:
    current = module
    for part in path.split("."):
        child = current[part] if isinstance(current, nn.ModuleDict) else getattr(current, part, None)
        if not isinstance(child, nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current


def _nonfactorized_epochs(request: ResidentQuantizationRequest, layer_position: int) -> int:
    schedule = request.nonfactorized_tuning_epochs_by_layer
    if not schedule:
        return request.nonfactorized_tuning_epochs
    return schedule[min(layer_position, len(schedule) - 1)]


def _tuning_seed(
    request: ResidentQuantizationRequest, stage: str, block: int, layer: str | None
) -> int:
    if request.legacy_tuning_seed_reset:
        return request.seed
    return logical_seed(request.seed, stage, block, layer, 0)


def _rehydrate_trainable_layer(
    state: FrozenNanoQuantState,
    tensors: LocalTensorStore,
    *,
    device: str,
    dtype: torch.dtype,
) -> TrainableFactorizedLinear:
    frozen = LayerFreezer().load(state, tensors, device=device, dtype=dtype).module
    return TrainableFactorizedLinear(
        frozen.left_binary,
        frozen.right_binary,
        frozen.scale_pre,
        frozen.scale_mid,
        frozen.scale_post,
        bias=frozen.bias,
        outlier_indices=frozen.outlier_indices,
        outlier_values=frozen.outlier_values,
        outlier_scales=frozen.outlier_scales,
    ).to(device=device, dtype=dtype)


def _run_resident_factorization_attempts(
    layer_plan: LayerPlan,
    source_weight: TensorRef,
    request: ResidentQuantizationRequest,
    budget: BudgetState,
    context: StageContext,
    config_hash: str,
    factor_stage: FactorizationAttemptStage,
    outlier_stage: OutlierSelectionStage,
    scale_stage: ScaleFitStage,
) -> tuple[AcceptedFactorization, OutlierSelectionResult, ScaleFitResult | None]:
    """Execute the complete legacy rank attempt: outliers, ADMM, scale fit, and full metric."""
    companions: list[tuple[OutlierSelectionResult, ScaleFitResult | None]] = []

    def execute_attempt(rank: int, attempt: int) -> FactorizationResult:
        started = time.perf_counter()
        if request.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(request.device)
        outlier_seed = (
            request.seed
            if request.legacy_tuning_seed_reset
            else logical_seed(
                request.seed,
                "outliers",
                layer_plan.layer.block.index,
                layer_plan.layer.path,
                attempt,
            )
        )
        factor_seed = (
            request.seed
            if request.legacy_tuning_seed_reset
            else logical_seed(
                request.seed,
                "factorize-attempt",
                layer_plan.layer.block.index,
                layer_plan.layer.path,
                attempt,
            )
        )
        outliers = execute_stage(
            outlier_stage,
            OutlierSelectionRequest(
                layer_plan.layer,
                source_weight,
                layer_plan.objective,
                layer_plan.outliers,
                rank,
                outlier_seed,
            ),
            context,
        )
        probe_peak = (
            int(torch.cuda.max_memory_allocated(request.device))
            if request.device.startswith("cuda")
            else 0
        )
        objective = replace(
            layer_plan.objective,
            input_importance=outliers.factor_input_importance,
        )
        factorized = execute_stage(
            factor_stage,
            FactorizationRequest(
                1,
                layer_plan.layer,
                source_weight,
                outliers.residual_weight,
                objective,
                rank,
                factor_seed,
                config_hash,
                outliers.factor_generator_state,
            ),
            context,
        )
        fitted = None
        scales = factorized.factors.scales
        if request.scale_fit.enabled:
            fitted = execute_stage(
                scale_stage,
                MaterializedScaleFitStageRequest(
                    ScaleFitRequest(
                        layer_plan.layer,
                        outliers.residual_weight,
                        factorized.factors,
                        objective,
                        outliers.indices,
                    ),
                    objective.input_importance,
                    layer_plan.objective.output_importance,
                ),
                context,
            )
            scales = fitted.scales
        if scales.mid is None:
            raise AssertionError("factorizer omitted required mid scale")
        with (
            context.tensor_store.read(source_weight, request.device) as source,
            context.tensor_store.read(factorized.factors.left_binary, request.device) as left,
            context.tensor_store.read(factorized.factors.right_binary, request.device) as right,
            context.tensor_store.read(scales.pre, request.device) as scale_pre,
            context.tensor_store.read(scales.mid, request.device) as scale_mid,
            context.tensor_store.read(scales.post, request.device) as scale_post,
            context.tensor_store.read(outliers.indices, request.device) as indices,
            context.tensor_store.read(outliers.values, request.device) as values,
            context.tensor_store.read(layer_plan.objective.input_importance, request.device) as input_importance,
            context.tensor_store.read(layer_plan.objective.output_importance, request.device) as output_importance,
        ):
            prediction = reconstruct(left, right, scale_pre, scale_mid, scale_post)
            outlier_scales = None
            if outliers.scales is not None:
                with context.tensor_store.read(outliers.scales, request.device) as stored_scales:
                    outlier_scales = stored_scales.clone()
            prediction = reconstruct_with_outliers(
                prediction,
                indices.long(),
                values,
                outlier_scales,
            )
            metrics = reconstruction_metrics(
                source,
                prediction,
                input_importance,
                output_importance,
            )
        companions.append((outliers, fitted))
        peak = (
            max(probe_peak, int(torch.cuda.max_memory_allocated(request.device)))
            if request.device.startswith("cuda")
            else factorized.peak_workspace_bytes
        )
        return replace(
            factorized,
            factors=replace(factorized.factors, scales=scales),
            metrics=metrics,
            wall_seconds=time.perf_counter() - started,
            peak_workspace_bytes=peak,
        )

    accepted = run_factorization_attempts(
        layer_plan,
        source_weight,
        source_weight,
        request.seed,
        config_hash,
        budget,
        context,
        lambda _result, _attempts: None,
        factor_stage,
        legacy_seed_reset=request.legacy_tuning_seed_reset,
        attempt_executor=execute_attempt,
    )
    accepted_index = next(index for index, item in enumerate(accepted.attempts) if item.accepted)
    outliers, fitted = companions[accepted_index]
    return accepted, outliers, fitted


def _resident_config_hash(request: ResidentQuantizationRequest) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_json(
                {
                    "resident_algorithm_version": RESIDENT_ALGORITHM_VERSION,
                    "runtime": {
                        "torch": str(torch.__version__),
                        "transformers": transformers.__version__,
                        "cuda": torch.version.cuda,
                    },
                    "target_bpw": request.target_bpw,
                    "rank_multiple": request.rank_multiple,
                    "allocation_strategy": request.allocation_strategy,
                    "rank_floor_fraction": request.rank_floor_fraction,
                    "rank_ceiling_fraction": request.rank_ceiling_fraction,
                    "rank_sensitivity_alpha": request.rank_sensitivity_alpha,
                    "rank_edge_boost": request.rank_edge_boost,
                    "layer_order": request.layer_order,
                    "admm": request.admm,
                    "outliers": request.outliers,
                    "scale_fit": request.scale_fit,
                    "factorized_tuning_epochs": request.factorized_tuning_epochs,
                    "factorized_tuning_batch_size": request.factorized_tuning_batch_size,
                    "factorized_tuning_learning_rate": request.factorized_tuning_learning_rate,
                    "nonfactorized_tuning_epochs": request.nonfactorized_tuning_epochs,
                    "nonfactorized_tuning_epochs_by_layer": request.nonfactorized_tuning_epochs_by_layer,
                    "nonfactorized_tuning_batch_size": request.nonfactorized_tuning_batch_size,
                    "nonfactorized_tuning_learning_rate": request.nonfactorized_tuning_learning_rate,
                    "nonfactorized_tuning_early_stop_relative_tolerance": (
                        request.nonfactorized_tuning_early_stop_relative_tolerance
                    ),
                    "post_block_refit_epochs": request.post_block_refit_epochs,
                    "post_block_refit_batch_size": request.post_block_refit_batch_size,
                    "post_block_refit_learning_rate": request.post_block_refit_learning_rate,
                    "tuning_microbatch_size": request.tuning_microbatch_size,
                    "block_forward_batch_size": request.block_forward_batch_size,
                    "legacy_tuning_seed_reset": request.legacy_tuning_seed_reset,
                    "activation_retention": request.activation_retention,
                    "calibration_method": request.calibration_method,
                    "calibration_shrinkage": request.calibration_shrinkage,
                    "calibration_batch_size": request.calibration_batch_size,
                    "seed": request.seed,
                }
            ).encode()
        ).hexdigest()
    )


def _factor_slice_source_inventory(
    request: ResidentQuantizationRequest,
) -> tuple[SafetensorsModelSource, CheckpointInventory, ModelInventory]:
    key = (
        str(request.snapshot.resolve()),
        request.source,
        request.revision,
        request.verify_hashes,
        request.layer_order,
    )
    cached = _FACTOR_SLICE_SOURCE_CACHE.get(key)
    if cached is not None:
        return cached
    source = SafetensorsModelSource(
        request.snapshot,
        source=request.source,
        revision=request.revision,
        verify_hashes=request.verify_hashes,
    )
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    inventory = adapter.model_inventory(source)
    if request.layer_order:
        inventory = replace(
            inventory,
            blocks=tuple(
                replace(
                    block,
                    quantizable_layers=tuple(
                        {layer.layer.path: layer for layer in block.quantizable_layers}[path]
                        for path in request.layer_order
                    ),
                )
                for block in inventory.blocks
            ),
        )
    result = (source, checkpoint, inventory)
    _FACTOR_SLICE_SOURCE_CACHE[key] = result
    return result


def _legacy_sensitivity_profile(
    inventory: ModelInventory,
    calibration: Any,
    source: SafetensorsModelSource,
    tensors: LocalTensorStore,
    *,
    alpha: float,
    device: str = "cpu",
) -> tuple[tuple[str, float], ...]:
    stats = {item.layer: item for item in calibration.stats.layers}
    entries: list[dict[str, Any]] = []
    for block in inventory.blocks:
        for layer in block.quantizable_layers:
            layer_stats = stats[layer.layer]
            with (
                source.read_tensor(layer.weight, device=device) as weight,
                tensors.read(layer_stats.input_importance, device) as input_importance,
                tensors.read(layer_stats.output_importance, device) as output_importance,
            ):
                energy = float(
                    (
                        weight.float().square()
                        * output_importance.float()[:, None].clamp_min(1e-12)
                        * input_importance.float()[None, :].clamp_min(1e-12)
                    )
                    .mean()
                    .sqrt()
                )
            entries.append({"block": block.block.index, "path": layer.layer.path, "energy": energy})
    block_medians = {
        block.block.index: max(
            statistics.median(item["energy"] for item in entries if item["block"] == block.block.index), 1e-12
        )
        for block in inventory.blocks
    }
    for item in entries:
        item["relative"] = item["energy"] / block_medians[item["block"]]
    type_medians = {
        path: max(statistics.median(item["relative"] for item in entries if item["path"] == path), 1e-12)
        for path in {item["path"] for item in entries}
    }
    result = []
    for item in entries:
        residual = max(item["relative"] / type_medians[item["path"]], 1e-12)
        score = residual**alpha * _layer_type_multiplier(item["path"])
        result.append((f"{item['block']}:{item['path']}", score))
    return tuple(result)


def _load_precomputed_preprocessing(
    request: ResidentQuantizationRequest,
    artifacts: LocalArtifactStore,
    inventory: ModelInventory,
    dataset: DatasetIdentity,
    total_tokens: int,
) -> tuple[PersistedCalibration, PersistedObjectives, PersistedPlan] | None:
    references = (
        request.precomputed_calibration,
        request.precomputed_objectives,
        request.precomputed_plan,
    )
    if all(reference is None for reference in references):
        return None
    if any(reference is None for reference in references):
        raise ValueError("precomputed calibration, objectives, and plan must be supplied together")
    calibration_ref, objectives_ref, plan_ref = cast(tuple[ArtifactRef, ArtifactRef, ArtifactRef], references)
    for reference in references:
        artifacts.validate(cast(ArtifactRef, reference).artifact_id)
    calibration_payload = json.loads(
        (artifacts.path_for(calibration_ref.artifact_id) / "stats.json").read_text(encoding="utf-8")
    )
    calibration = PersistedCalibration(
        calibration_ref,
        from_dict(CalibrationStats, calibration_payload, path="calibration"),
    )
    objective_payload = json.loads(
        (artifacts.path_for(objectives_ref.artifact_id) / "objectives.json").read_text(encoding="utf-8")
    )
    objectives = PersistedObjectives(
        objectives_ref,
        tuple(
            from_dict(ObjectiveSpec, item, path=f"objectives[{index}]")
            for index, item in enumerate(objective_payload)
        ),
    )
    plan_payload = json.loads((artifacts.path_for(plan_ref.artifact_id) / "plan.json").read_text(encoding="utf-8"))
    persisted_plan = PersistedPlan(plan_ref, from_dict(QuantizationPlan, plan_payload, path="plan"))
    if calibration.stats.model != inventory.model or calibration.stats.dataset != dataset:
        raise ValueError("precomputed calibration identity does not match the requested model/dataset")
    if calibration.stats.method != request.calibration_method or calibration.stats.total_tokens != total_tokens:
        raise ValueError("precomputed calibration protocol does not match the request")
    if any(objective.source_calibration != calibration_ref for objective in objectives.objectives):
        raise ValueError("precomputed objectives do not reference the selected calibration")
    if persisted_plan.plan.model != inventory.model or persisted_plan.plan.calibration != calibration_ref:
        raise ValueError("precomputed plan identity does not match the selected model/calibration")
    if persisted_plan.plan.target_bpw != request.target_bpw:
        raise ValueError("precomputed plan target BPW does not match the request")
    return calibration, objectives, persisted_plan


def _run_resident_quantization(request: ResidentQuantizationRequest) -> ResidentQuantizationResult:
    """Quantize all decoder linears while the source model remains resident on one device."""
    started = time.perf_counter()
    if request.block_forward_batch_size <= 0:
        raise ValueError("resident quantization block forward batch size must be positive")
    if request.calibration_batch_size <= 0:
        raise ValueError("resident quantization calibration batch size must be positive")
    if request.factorized_tuning_epochs < 0 or request.nonfactorized_tuning_epochs < 0:
        raise ValueError("resident quantization tuning epochs cannot be negative")
    if any(epoch < 0 for epoch in request.nonfactorized_tuning_epochs_by_layer):
        raise ValueError("resident quantization non-factorized tuning schedule cannot contain negative epochs")
    if request.post_block_refit_epochs < 0:
        raise ValueError("resident quantization post-block refit epochs cannot be negative")
    if request.factorized_tuning_epochs > 0 and request.factorized_tuning_batch_size <= 0:
        raise ValueError("resident quantization factorized tuning batch size must be positive")
    if (
        request.nonfactorized_tuning_epochs > 0 or any(request.nonfactorized_tuning_epochs_by_layer)
    ) and request.nonfactorized_tuning_batch_size <= 0:
        raise ValueError("resident quantization non-factorized tuning batch size must be positive")
    if request.post_block_refit_epochs > 0 and request.post_block_refit_batch_size <= 0:
        raise ValueError("resident quantization post-block refit batch size must be positive")
    if request.tuning_microbatch_size is not None and request.tuning_microbatch_size <= 0:
        raise ValueError("resident quantization tuning microbatch size must be positive")
    if request.activation_retention not in {"rolling", "all"}:
        raise ValueError("resident quantization activation retention must be 'rolling' or 'all'")
    if request.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA resident quantization requested without CUDA")
    if request.defer_layer_loss_snapshots and (
        request.factorized_tuning_epochs > 0
        or request.nonfactorized_tuning_epochs > 0
        or any(request.nonfactorized_tuning_epochs_by_layer)
        or request.post_block_refit_epochs > 0
    ):
        raise ValueError("deferred layer losses are incompatible with activation-based tuning")
    artifacts = LocalArtifactStore(request.output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    executor = ResidentExecutor()
    events = JsonlEventSink(request.output / "events.jsonl", "resident-quantization")
    context = StageContext("resident-quantization", executor, artifacts, tensors, events, Cancellation())
    source = SafetensorsModelSource(
        request.snapshot,
        source=request.source,
        revision=request.revision,
        verify_hashes=request.verify_hashes,
    )
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    inventory = adapter.model_inventory(source)
    if request.layer_order:
        reordered_blocks = []
        for inventory_block in inventory.blocks:
            by_path = {layer.layer.path: layer for layer in inventory_block.quantizable_layers}
            if set(by_path) != set(request.layer_order):
                raise ValueError("requested layer order does not exactly match adapter quantizable layers")
            reordered_blocks.append(
                replace(
                    inventory_block,
                    quantizable_layers=tuple(by_path[path] for path in request.layer_order),
                )
            )
        inventory = replace(inventory, blocks=tuple(reordered_blocks))
    tokens = _token_tensor(request.token_ids, request.device)
    quality_tokens = _token_tensor(
        request.token_ids if request.quality_token_ids is None else request.quality_token_ids,
        request.device,
    )
    if request.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(request.device)
    model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(
            request.snapshot,
            local_files_only=True,
            torch_dtype=_checkpoint_dtype(checkpoint.config),
            attn_implementation=adapter.attention_implementation,
        ),
    ).to(request.device)
    model.eval()
    decoder_layers = _decoder_layers(model)
    reference_logits = None
    if request.evaluate_inline_quality:
        with torch.no_grad():
            reference_logits = cast(
                torch.Tensor,
                cast(Any, model)(input_ids=quality_tokens, use_cache=False).logits,
            ).detach()
    text_model = getattr(model, "model", model)
    capture = capture_prefix_invocations(
        decoder_layers[0],
        (lambda: cast(Any, text_model)(input_ids=tokens[:1], use_cache=False),),
    )[0]
    captured_input = capture.positional[0]
    if not isinstance(captured_input, torch.Tensor):
        raise TypeError("captured first-block hidden state is not a tensor")
    initial_inputs = _run_prefix_batched(
        adapter,
        model,
        tokens,
        request.block_forward_batch_size,
        "cpu",
    ).detach()
    if not torch.equal(initial_inputs[:1], captured_input.detach().cpu()):
        raise ValueError("adapter prefix does not match the model's first-block input")
    metadata = capture.keyword
    if request.device.startswith("cuda"):
        # Prefix capture traverses the full embedding/prefix path in large
        # no-grad batches. Its workspaces are dead once activations are on CPU.
        torch.cuda.empty_cache()
    token_bytes = tokens.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    dataset = DatasetIdentity(
        "sha256:" + hashlib.sha256(token_bytes).hexdigest(),
        ("deterministic-token-fixture",),
        ("1",),
        checkpoint.tokenizer_hash,
        "raw-token-ids-v1",
    )
    preprocessed = _load_precomputed_preprocessing(request, artifacts, inventory, dataset, tokens.numel())

    calibration_values: list[tuple[LayerId, MaterializedLayerCalibration]] = []
    if preprocessed is not None:
        pass
    elif request.calibration_method in {"online_fisher", "two_phase_fisher"}:
        causal_layers: list[tuple[str, nn.Linear]] = []
        causal_ids: dict[str, LayerId] = {}
        for block_inventory, block in zip(inventory.blocks, decoder_layers, strict=True):
            block_modules = dict(block.named_modules())
            for layer in adapter.quantizable_layers(block, block_inventory.block):
                module = block_modules.get(layer.path)
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"causal calibration target is not a linear layer: {layer}")
                key = f"block.{block_inventory.block.index}.{layer.path}"
                causal_layers.append((key, module))
                causal_ids[key] = layer
        stats = calibrate_causal_model(
            model,
            tuple(
                tokens[start : start + request.calibration_batch_size]
                for start in range(0, tokens.shape[0], request.calibration_batch_size)
            ),
            tuple(causal_layers),
            method=request.calibration_method,
            shrinkage=request.calibration_shrinkage,
        )
        calibration_values.extend((causal_ids[item.path], item) for item in stats)
    elif request.calibration_method == "forward_only":
        calibration_inputs = initial_inputs
        for block_inventory, block in zip(inventory.blocks, decoder_layers, strict=True):
            paths = tuple(layer.path for layer in adapter.quantizable_layers(block, block_inventory.block))

            def calibration_runner(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
                parameter = next(iter(module.parameters()), None)
                device = value.device if parameter is None else parameter.device
                return adapter.run_block(module, value.to(device, non_blocking=True), **metadata)

            stats = calibrate_block(
                block,
                tuple(
                    calibration_inputs[start : start + request.block_forward_batch_size]
                    for start in range(0, calibration_inputs.shape[0], request.block_forward_batch_size)
                ),
                paths,
                calibration_runner,
                method="forward_only",
                shrinkage=request.calibration_shrinkage,
            )
            calibration_values.extend((LayerId(block_inventory.block, item.path), item) for item in stats)
            with torch.no_grad():
                calibration_inputs = _run_block_batched(
                    adapter,
                    block,
                    calibration_inputs,
                    metadata,
                    request.block_forward_batch_size,
                ).detach()
    else:
        raise ValueError(f"unsupported resident calibration method: {request.calibration_method}")
    if preprocessed is None:
        calibration = persist_calibration(
            tuple(calibration_values),
            inventory.model,
            dataset,
            request.calibration_method,
            "float32",
            artifacts,
            tensors,
            total_tokens=tokens.numel(),
        )
        objectives = build_objectives(calibration, ObjectiveConfig(), artifacts)
        sensitivity_profile = (
            _legacy_sensitivity_profile(
                inventory,
                calibration,
                source,
                tensors,
                alpha=request.rank_sensitivity_alpha,
                device=request.device,
            )
            if request.allocation_strategy is AllocationStrategy.SENSITIVITY
            else ()
        )
        allocation = RankAllocationConfig(
            target_bpw=request.target_bpw,
            strategy=request.allocation_strategy,
            sensitivity_alpha=request.rank_sensitivity_alpha,
            bounds=RankBoundsConfig(
                multiple=request.rank_multiple,
                floor_fraction_of_uniform=request.rank_floor_fraction,
                ceiling_fraction_of_uniform=request.rank_ceiling_fraction,
                edge_block_boost=request.rank_edge_boost,
            ),
            retry=RankRetryConfig(
                enabled=True,
                thresholds=RetryThresholdConfig(
                    weighted_normalized_error=0.5,
                    raw_normalized_error=0.5,
                ),
                rank_increase_fraction=0.25,
                maximum_attempts=3,
                extra_bit_budget_fraction=0.02,
                allow_above_allocator_cap=True,
            ),
        )
        plan = build_quantization_plan(
            PlanningRequest(
                inventory,
                calibration.stats,
                calibration.reference,
                objectives.objectives,
                allocation,
                request.outliers,
                sensitivity_profile,
            )
        )
        persisted_plan = persist_plan(plan, artifacts)
    else:
        calibration, objectives, persisted_plan = preprocessed
        plan = persisted_plan.plan
    config_hash = _resident_config_hash(request)
    identity = CommitIdentity(config_hash, inventory.model.config_hash, persisted_plan.reference.artifact_id)
    journal = ProgressJournal(request.output / "state", "resident-quantization", artifacts)
    discovery = journal.discover(plan, identity)
    discovered_records = (*discovery.valid_records, *discovery.orphan_records)
    block_records = sorted(
        (record for record in discovered_records if record.kind == "block"), key=lambda record: record.block
    )
    committed_blocks = [
        (
            ArtifactRef("block-result", record.artifact_id, 1),
            load_committed_block(ArtifactRef("block-result", record.artifact_id, 1), artifacts, identity).result,
        )
        for record in block_records
    ]
    if request.activation_retention == "rolling":
        for _reference, old_block in committed_blocks[:-1]:
            retire_block_activations(old_block, artifacts)
    accepted_bits = sum(layer.actual_bit_cost.total for _, block in committed_blocks for layer in block.layers)
    retry_bits_spent = sum(
        layer.extra_retry_bits for _, block in committed_blocks for layer in block.layers
    )
    budget = BudgetState(plan.planned_cost.total, accepted_bits, retry_bits_spent)
    if committed_blocks:
        teacher_inputs, compressed_inputs = load_block_activations(committed_blocks[-1][0], artifacts, "cpu")
    else:
        teacher_inputs = initial_inputs
        compressed_inputs = initial_inputs
    del initial_inputs
    completed_block_indexes = {block.block.index for _, block in committed_blocks}
    layer_container = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layer_container, nn.ModuleList):
        raise TypeError("model does not expose a mutable decoder layer stack")
    if request.restore_completed_blocks:
        for _, completed_block in committed_blocks:
            restored_block = layer_container[completed_block.block.index]
            for state in completed_block.frozen_state.quantized_layers:
                frozen = LayerFreezer().load(
                    state,
                    tensors,
                    device=request.device,
                    dtype=compressed_inputs.dtype,
                    backend="factorized",
                )
                BlockEditor().install_frozen_layer(
                    restored_block,
                    state.layer.path,
                    frozen.module,
                )
            restore_block_auxiliary_parameters(
                restored_block,
                completed_block.frozen_state.auxiliary_parameters,
                tensors,
                device=request.device,
            )
    del decoder_layers
    if request.device.startswith("cuda") and completed_block_indexes:
        # Restoring a deep resume prefix replaces many full-precision CUDA weights.
        # Release those now-unreachable allocator reservations once at the resume
        # boundary so activation-based tuning has the same workspace as a cold run.
        torch.cuda.empty_cache()
    partial_layer_records = {
        (record.block, record.layer): record
        for record in discovered_records
        if record.kind == "layer" and record.block not in completed_block_indexes
    }
    peak_device_bytes = 0
    factorization_wall_seconds = 0.0
    new_layer_commits = 0
    new_block_commits = 0
    factor_stage = FactorizationAttemptStage(request.admm, device=request.device)
    outlier_stage = OutlierSelectionStage(
        device=request.device,
        residual_probe_iterations=request.outliers.residual_probe.iterations,
    )
    scale_stage = ScaleFitStage(request.scale_fit, device=request.device)

    for block_plan in plan.blocks:
        if block_plan.block.index in completed_block_indexes:
            continue
        block_started = time.perf_counter()
        deferred_slice = request.defer_layer_loss_snapshots
        block_index = block_plan.block.index
        source_block = layer_container[block_index]
        working_block = adapter.load_block(source, block_plan.block, request.device)
        working_block.eval()
        with torch.no_grad():
            teacher_outputs = _run_block_batched(
                adapter,
                source_block,
                teacher_inputs,
                metadata,
                request.block_forward_batch_size,
                "cpu",
            ).detach()
        block_output_stats = next(
            (
                item
                for item in calibration.stats.layers
                if item.layer.block.index == block_index and item.layer.path == "mlp.down_proj"
            ),
            next(item for item in calibration.stats.layers if item.layer.block.index == block_index),
        )
        with tensors.read(block_output_stats.output_importance, request.device) as value:
            block_output_importance = value.clone()
        recorder = BlockLossRecorder()
        recorder.record_source_reference(_self_reference_weighted_mse(teacher_outputs, block_output_importance))
        recorder.record_block_entry(
            0.0
            if deferred_slice
            else _block_loss(
                adapter,
                working_block,
                compressed_inputs,
                teacher_outputs,
                block_output_importance,
                metadata,
                request.block_forward_batch_size,
            )
        )
        if request.device.startswith("cuda"):
            # The no-grad source/output probes use the larger forward batch and
            # otherwise leave attention workspaces reserved during backprop.
            torch.cuda.empty_cache()
        layer_results: list[LayerResult] = []
        frozen_states = []
        quantization_targets: dict[str, TensorRef] = {}

        for layer_position, layer_plan in enumerate(block_plan.layers):
            nonfactorized_epochs = _nonfactorized_epochs(request, layer_position)
            if nonfactorized_epochs > 0:
                tune_non_factorized(
                    working_block,
                    TuningRequest(
                        compressed_inputs,
                        teacher_outputs,
                        nonfactorized_epochs,
                        request.nonfactorized_tuning_batch_size,
                        request.nonfactorized_tuning_learning_rate,
                        early_stop_relative_tolerance=(
                            request.nonfactorized_tuning_early_stop_relative_tolerance
                        ),
                        output_importance=block_output_importance,
                        seed=_tuning_seed(
                            request, "nonfactorized-tuning", block_index, layer_plan.layer.path
                        ),
                        microbatch_size=request.tuning_microbatch_size,
                    ),
                    lambda module, value: adapter.run_block(module, value, **metadata),
                )
            source_linear = _module_at_path(working_block, layer_plan.layer.path)
            source_weight = getattr(source_linear, "weight", None)
            if not isinstance(source_weight, torch.Tensor):
                raise TypeError(f"quantization target has no materialized weight: {layer_plan.layer.path}")
            source_ref = tensors.put("source-layer", {"weight": source_weight.detach().cpu()})["weight"]
            quantization_targets[layer_plan.layer.path] = source_ref
            prior_record = partial_layer_records.get((block_index, layer_plan.layer.path))
            if prior_record is not None:
                prior = load_committed_layer(
                    ArtifactRef("layer-result", prior_record.artifact_id, 1), artifacts, identity
                ).result
                frozen = LayerFreezer().load(
                    prior.frozen_state,
                    tensors,
                    device=request.device,
                    dtype=compressed_inputs.dtype,
                    backend="factorized",
                )
                BlockEditor().install_frozen_layer(working_block, layer_plan.layer.path, frozen.module)
                frozen_states.append(prior.frozen_state)
                layer_results.append(prior)
                budget = replace(
                    budget,
                    accepted_bits=budget.accepted_bits + prior.actual_bit_cost.total,
                    retry_bits_spent=budget.retry_bits_spent + prior.extra_retry_bits,
                )
                if not deferred_slice:
                    recorder.record_after_layer(
                        layer_plan.layer,
                        _block_loss(
                            adapter,
                            working_block,
                            compressed_inputs,
                            teacher_outputs,
                            block_output_importance,
                            metadata,
                            request.block_forward_batch_size,
                        ),
                    )
                continue
            accepted, outliers, fitted = _run_resident_factorization_attempts(
                layer_plan,
                source_ref,
                request,
                budget,
                context,
                config_hash,
                factor_stage,
                outlier_stage,
                scale_stage,
            )
            factorized = accepted.result
            peak_device_bytes = max(peak_device_bytes, accepted.peak_workspace_bytes)
            factorization_wall_seconds += accepted.wall_seconds
            scales = factorized.factors.scales
            mid_ref = scales.mid
            if mid_ref is None:
                raise AssertionError("factorizer omitted required mid scale")
            outlier_indices = None
            outlier_values = None
            outlier_scales = None
            if layer_plan.outliers.count:
                with (
                    tensors.read(outliers.indices, request.device) as indices,
                    tensors.read(outliers.values, request.device) as values,
                ):
                    outlier_indices = indices.clone()
                    outlier_values = values.clone()
                if outliers.scales is not None:
                    with tensors.read(outliers.scales, request.device) as values:
                        outlier_scales = values.clone()
            left_initial = (
                factorized.factors.left_latent
                if request.factorized_tuning_epochs > 0
                else factorized.factors.left_binary
            )
            right_initial = (
                factorized.factors.right_latent
                if request.factorized_tuning_epochs > 0
                else factorized.factors.right_binary
            )
            with (
                tensors.read(left_initial, request.device) as left,
                tensors.read(right_initial, request.device) as right,
                tensors.read(scales.pre, request.device) as scale_pre,
                tensors.read(mid_ref, request.device) as scale_mid,
                tensors.read(scales.post, request.device) as scale_post,
            ):
                trainable = TrainableFactorizedLinear(
                    left,
                    right,
                    scale_pre,
                    scale_mid,
                    scale_post,
                    outlier_indices=outlier_indices,
                    outlier_values=outlier_values,
                    outlier_scales=outlier_scales,
                ).to(device=request.device, dtype=compressed_inputs.dtype)
            tuning = None
            if request.factorized_tuning_epochs > 0:
                BlockEditor().install_trainable_layer(working_block, layer_plan.layer.path, trainable)
                tuning = tune_factorized(
                    working_block,
                    layer_plan.layer.path,
                    TuningRequest(
                        compressed_inputs,
                        teacher_outputs,
                        request.factorized_tuning_epochs,
                        request.factorized_tuning_batch_size,
                        request.factorized_tuning_learning_rate,
                        output_importance=block_output_importance,
                        seed=_tuning_seed(request, "factorized-tuning", block_index, layer_plan.layer.path),
                        microbatch_size=request.tuning_microbatch_size,
                    ),
                    lambda module, value: adapter.run_block(module, value, **metadata),
                )
            frozen_outliers = (
                None
                if layer_plan.outliers.count == 0
                else FrozenOutlierState(outliers.indices, outliers.values, outliers.scales)
            )
            frozen = LayerFreezer().freeze(
                layer_plan.layer,
                trainable,
                tensors,
                outliers=frozen_outliers,
                backend="factorized",
            )
            with (
                tensors.read(source_ref, request.device) as source_value,
                tensors.read(layer_plan.objective.input_importance, request.device) as input_importance,
                tensors.read(layer_plan.objective.output_importance, request.device) as output_importance,
            ):
                final_metrics = reconstruction_metrics(
                    source_value,
                    frozen.module.dense_weight(),
                    input_importance,
                    output_importance,
                )
            frozen_module = frozen.module.to(device=request.device, dtype=compressed_inputs.dtype)
            BlockEditor().install_frozen_layer(working_block, layer_plan.layer.path, frozen_module)
            frozen_states.append(frozen.state)
            accepted_attempt = next(
                index for index, attempt in enumerate(accepted.attempts) if attempt.accepted
            )
            layer_result = LayerResult(
                1,
                layer_plan.layer,
                layer_plan,
                accepted.attempts,
                accepted_attempt,
                factorized.factors.left_binary.artifact,
                fitted,
                tuning,
                frozen.state,
                final_metrics,
                accepted.actual_bit_cost,
                accepted.extra_retry_bits,
                ()
                if request.factorized_tuning_epochs > 0 and request.scale_fit.enabled
                else (("tuning_disabled",) if request.scale_fit.enabled else ("scale_fit_disabled", "tuning_disabled")),
            )
            committed_layer = commit_layer(layer_result, artifacts, identity)
            journal.append(
                "layer",
                block_index,
                layer_plan.layer.path,
                committed_layer.reference.artifact_id,
                identity,
            )
            new_layer_commits += 1
            if (
                request.interrupt_after_layer_commits is not None
                and new_layer_commits >= request.interrupt_after_layer_commits
            ):
                raise InterruptedError(f"injected interruption after {new_layer_commits} new layer commits")
            layer_results.append(layer_result)
            budget = accepted.budget
            recorder.record_after_layer(
                layer_plan.layer,
                _block_loss(
                    adapter,
                    working_block,
                    compressed_inputs,
                    teacher_outputs,
                    block_output_importance,
                    metadata,
                    request.block_forward_batch_size,
                ),
            )
        if request.post_block_refit_epochs > 0:
            trainable_by_path: dict[str, TrainableFactorizedLinear] = {}
            for state in frozen_states:
                trainable = _rehydrate_trainable_layer(
                    state,
                    tensors,
                    device=request.device,
                    dtype=compressed_inputs.dtype,
                )
                BlockEditor().install_trainable_layer(working_block, state.layer.path, trainable)
                trainable_by_path[state.layer.path] = trainable
            post_block_refit(
                working_block,
                TuningRequest(
                    compressed_inputs,
                    teacher_outputs,
                    request.post_block_refit_epochs,
                    request.post_block_refit_batch_size,
                    request.post_block_refit_learning_rate,
                    output_importance=block_output_importance,
                    seed=_tuning_seed(request, "post-block-refit", block_index, None),
                    microbatch_size=request.tuning_microbatch_size,
                ),
                lambda module, value: adapter.run_block(module, value, **metadata),
            )
            refitted_states = []
            refitted_results = []
            for layer_result in layer_results:
                refitted = LayerFreezer().freeze(
                    layer_result.layer,
                    trainable_by_path[layer_result.layer.path],
                    tensors,
                    outliers=layer_result.frozen_state.outliers,
                    backend="factorized",
                )
                with (
                    tensors.read(quantization_targets[layer_result.layer.path], request.device) as source_value,
                    tensors.read(layer_result.plan.objective.input_importance, request.device) as input_importance,
                    tensors.read(layer_result.plan.objective.output_importance, request.device) as output_importance,
                ):
                    metrics = reconstruction_metrics(
                        source_value,
                        refitted.module.dense_weight(),
                        input_importance,
                        output_importance,
                    )
                frozen_module = refitted.module.to(device=request.device, dtype=compressed_inputs.dtype)
                BlockEditor().install_frozen_layer(working_block, layer_result.layer.path, frozen_module)
                refitted_states.append(refitted.state)
                refitted_results.append(
                    replace(layer_result, frozen_state=refitted.state, final_reconstruction=metrics)
                )
            frozen_states = refitted_states
            layer_results = refitted_results
            recorder.record_post_block_refit(
                _block_loss(
                    adapter,
                    working_block,
                    compressed_inputs,
                    teacher_outputs,
                    block_output_importance,
                    metadata,
                    request.block_forward_batch_size,
                )
            )
        with torch.no_grad():
            compressed_outputs = _run_block_batched(
                adapter,
                working_block,
                compressed_inputs,
                metadata,
                request.block_forward_batch_size,
                "cpu",
            ).detach()
        recorder.record_final_frozen_pre_kd(
            _weighted_mse(compressed_outputs, teacher_outputs, block_output_importance)
        )
        auxiliary_parameters = freeze_block_auxiliary_parameters(working_block, tensors)
        frozen_block = FrozenBlockState(
            block_plan.block,
            tuple(frozen_states),
            (),
            auxiliary_parameters,
        )
        block_peak = int(torch.cuda.max_memory_allocated(request.device)) if request.device.startswith("cuda") else 0
        peak_device_bytes = max(peak_device_bytes, block_peak)
        committed = commit_block(
            block_plan.block,
            tuple(layer_results),
            frozen_block,
            recorder.finalize(),
            teacher_outputs,
            compressed_outputs,
            budget.retry_bits_spent,
            artifacts,
            identity,
            wall_seconds=time.perf_counter() - block_started,
            peak_gpu_bytes=block_peak,
            warnings=()
            if request.factorized_tuning_epochs > 0 and request.scale_fit.enabled
            else (("tuning_disabled",) if request.scale_fit.enabled else ("scale_fit_disabled", "tuning_disabled")),
        )
        journal.append("block", block_index, None, committed.reference.artifact_id, identity)
        if request.activation_retention == "rolling" and committed_blocks:
            retire_block_activations(committed_blocks[-1][1], artifacts)
        committed_blocks.append((committed.reference, committed.result))
        new_block_commits += 1
        if (
            request.interrupt_after_block_commits is not None
            and new_block_commits >= request.interrupt_after_block_commits
        ):
            raise InterruptedError(f"injected interruption after {new_block_commits} new block commits")
        teacher_inputs = teacher_outputs
        compressed_inputs = compressed_outputs
        layer_container[block_index] = working_block
        del working_block

    compressed_logits = None
    if request.evaluate_inline_quality:
        if not request.restore_completed_blocks and completed_block_indexes:
            raise ValueError("inline quality evaluation requires completed-block restoration")
        with torch.no_grad():
            compressed_logits = cast(
                torch.Tensor,
                cast(Any, model)(input_ids=quality_tokens, use_cache=False).logits,
            ).detach()
    original_elements = sum(
        layer.in_features * layer.out_features for block in inventory.blocks for layer in block.quantizable_layers
    )
    frozen_model = assemble_frozen_model(
        inventory.model,
        persisted_plan.reference,
        tuple(committed_blocks),
        (),
        original_elements,
    )
    if reference_logits is None or compressed_logits is None:
        reference_nll = compressed_nll = logit_mse = argmax_agreement = float("nan")
    else:
        reference_nll = _nll(reference_logits, quality_tokens)
        compressed_nll = _nll(compressed_logits, quality_tokens)
        logit_mse = _mse(compressed_logits, reference_logits)
        argmax_agreement = float((compressed_logits.argmax(-1) == reference_logits.argmax(-1)).float().mean())
    elapsed = time.perf_counter() - started
    peak_host_bytes = peak_process_memory_bytes()
    ranks = [layer.rank for block in plan.blocks for layer in block.layers]
    artifact_bytes_before_report = _artifact_bytes(artifacts.root)
    report_payload = {
        "schema_version": 1,
        "source": request.source,
        "revision": request.revision,
        "model": to_dict(inventory.model),
        "plan": persisted_plan.reference.artifact_id,
        "block_count": len(committed_blocks),
        "layer_count": sum(len(block.layers) for _, block in committed_blocks),
        "target_bpw": request.target_bpw,
        "effective_bpw": frozen_model.effective_bpw,
        "actual_total_bits": frozen_model.actual_total_bits,
        "rank_minimum": min(ranks),
        "rank_maximum": max(ranks),
        "rank_mean": sum(ranks) / len(ranks),
        "mean_final_block_loss": sum(block.losses.final_frozen_pre_kd for _, block in committed_blocks)
        / len(committed_blocks),
        "factorization_wall_seconds": factorization_wall_seconds,
        "reference_nll": reference_nll,
        "compressed_nll": compressed_nll,
        "logit_mse": logit_mse,
        "argmax_agreement": argmax_agreement,
        "peak_device_bytes": peak_device_bytes,
        "peak_host_bytes": peak_host_bytes,
        "artifact_bytes_before_report": artifact_bytes_before_report,
        "elapsed_seconds": elapsed,
        "admm": to_dict(request.admm),
        "warnings": [
            *([] if request.scale_fit.enabled else ["scale_fit_disabled"]),
            *([] if request.factorized_tuning_epochs > 0 else ["tuning_disabled"]),
            "single_fixture_quality_only",
        ],
        "reused_commit_count": len(discovered_records),
    }
    with artifacts.begin_write("resident-quantization-report") as writer:
        (writer.path / "report.json").write_text(json.dumps(report_payload, sort_keys=True, indent=2), encoding="utf-8")
        (writer.path / "reconstruction.md").write_text(
            render_reconstruction_tables(tuple(block for _, block in committed_blocks)), encoding="utf-8"
        )
        descriptor = writer.commit()
    report = ArtifactRef("resident-quantization-report", descriptor.artifact_id, descriptor.schema_version)
    artifact_bytes = _artifact_bytes(artifacts.root)
    executor.release()
    events.close()
    return ResidentQuantizationResult(
        inventory,
        plan,
        identity,
        frozen_model,
        tuple(block for _, block in committed_blocks),
        report,
        reference_nll,
        compressed_nll,
        logit_mse,
        argmax_agreement,
        peak_device_bytes,
        peak_host_bytes,
        artifact_bytes,
        elapsed,
        len(discovered_records),
    )


def _run_resident_factorization_slice(request: ResidentQuantizationRequest) -> ResidentFactorizationSliceResult:
    """Commit the next planned weight-only layer without loading the Transformers model."""
    started = time.perf_counter()
    if (
        request.factorized_tuning_epochs > 0
        or request.nonfactorized_tuning_epochs > 0
        or any(request.nonfactorized_tuning_epochs_by_layer)
        or request.post_block_refit_epochs > 0
    ):
        raise ValueError("factor-only slices do not support activation-based tuning")
    artifacts = LocalArtifactStore(request.output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    source, checkpoint, inventory = _factor_slice_source_inventory(request)
    tokens = _token_tensor(request.token_ids, "cpu")
    token_bytes = tokens.contiguous().view(torch.uint8).numpy().tobytes()
    dataset = DatasetIdentity(
        "sha256:" + hashlib.sha256(token_bytes).hexdigest(),
        ("deterministic-token-fixture",),
        ("1",),
        checkpoint.tokenizer_hash,
        "raw-token-ids-v1",
    )
    preprocessed = _load_precomputed_preprocessing(request, artifacts, inventory, dataset, tokens.numel())
    if preprocessed is None:
        raise ValueError("factor-only slices require precomputed calibration, objectives, and plan")
    _calibration, _objectives, persisted_plan = preprocessed
    plan = persisted_plan.plan
    config_hash = _resident_config_hash(request)
    identity = CommitIdentity(config_hash, inventory.model.config_hash, persisted_plan.reference.artifact_id)
    journal = ProgressJournal(request.output / "state", "resident-quantization", artifacts)
    discovery = journal.discover(plan, identity)
    records = (*discovery.valid_records, *discovery.orphan_records)
    complete_blocks = {record.block for record in records if record.kind == "block"}
    completed_results = [
        load_committed_block(ArtifactRef("block-result", record.artifact_id, 1), artifacts, identity).result
        for record in records
        if record.kind == "block"
    ]
    partial_results = [
        load_committed_layer(ArtifactRef("layer-result", record.artifact_id, 1), artifacts, identity).result
        for record in records
        if record.kind == "layer" and record.block not in complete_blocks
    ]
    prior_layers = [
        *(layer for block in completed_results for layer in block.layers),
        *partial_results,
    ]
    budget = BudgetState(
        plan.planned_cost.total,
        sum(layer.actual_bit_cost.total for layer in prior_layers),
        sum(layer.extra_retry_bits for layer in prior_layers),
    )
    complete_layers = {
        (record.block, record.layer)
        for record in records
        if record.kind == "layer" and record.block not in complete_blocks
    }
    pending = [
        layer
        for block in plan.blocks
        if block.block.index not in complete_blocks
        for layer in block.layers
        if (block.block.index, layer.layer.path) not in complete_layers
    ]
    if not pending:
        return ResidentFactorizationSliceResult(None, identity, time.perf_counter() - started, 0, 0)
    layer_plan = pending[0]
    block_index = layer_plan.layer.block.index
    executor = ResidentExecutor()
    events = JsonlEventSink(request.output / "events.jsonl", "resident-factorization-slice")
    context = StageContext("resident-factorization-slice", executor, artifacts, tensors, events, Cancellation())
    if request.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(request.device)
    try:
        factor_stage = FactorizationAttemptStage(request.admm, device=request.device)
        outlier_stage = OutlierSelectionStage(
            device=request.device,
            residual_probe_iterations=request.outliers.residual_probe.iterations,
        )
        scale_stage = ScaleFitStage(request.scale_fit, device=request.device)
        with source.read_tensor(layer_plan.source_weight, device="cpu") as source_weight:
            source_ref = tensors.put("source-layer", {"weight": source_weight})["weight"]
        accepted, outliers, fitted = _run_resident_factorization_attempts(
            layer_plan,
            source_ref,
            request,
            budget,
            context,
            config_hash,
            factor_stage,
            outlier_stage,
            scale_stage,
        )
        factorized = accepted.result
        scales = factorized.factors.scales
        if scales.mid is None:
            raise AssertionError("factorizer omitted required mid scale")
        outlier_indices = outlier_values = outlier_scales = None
        if layer_plan.outliers.count:
            with (
                tensors.read(outliers.indices, request.device) as indices,
                tensors.read(outliers.values, request.device) as values,
            ):
                outlier_indices = indices.clone()
                outlier_values = values.clone()
            if outliers.scales is not None:
                with tensors.read(outliers.scales, request.device) as values:
                    outlier_scales = values.clone()
        with (
            tensors.read(factorized.factors.left_binary, request.device) as left,
            tensors.read(factorized.factors.right_binary, request.device) as right,
            tensors.read(scales.pre, request.device) as scale_pre,
            tensors.read(scales.mid, request.device) as scale_mid,
            tensors.read(scales.post, request.device) as scale_post,
        ):
            trainable = TrainableFactorizedLinear(
                left,
                right,
                scale_pre,
                scale_mid,
                scale_post,
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                outlier_scales=outlier_scales,
            )
        frozen_outliers = (
            None
            if layer_plan.outliers.count == 0
            else FrozenOutlierState(outliers.indices, outliers.values, outliers.scales)
        )
        frozen = LayerFreezer().freeze(layer_plan.layer, trainable, tensors, outliers=frozen_outliers)
        with (
            tensors.read(source_ref, request.device) as source_value,
            tensors.read(layer_plan.objective.input_importance, request.device) as input_importance,
            tensors.read(layer_plan.objective.output_importance, request.device) as output_importance,
        ):
            final_metrics = reconstruction_metrics(
                source_value,
                frozen.module.dense_weight(),
                input_importance,
                output_importance,
            )
        accepted_attempt = next(
            index for index, attempt in enumerate(accepted.attempts) if attempt.accepted
        )
        layer_result = LayerResult(
            1,
            layer_plan.layer,
            layer_plan,
            accepted.attempts,
            accepted_attempt,
            factorized.factors.left_binary.artifact,
            fitted,
            None,
            frozen.state,
            final_metrics,
            accepted.actual_bit_cost,
            accepted.extra_retry_bits,
            ("tuning_disabled",) if request.scale_fit.enabled else ("scale_fit_disabled", "tuning_disabled"),
        )
        committed = commit_layer(layer_result, artifacts, identity)
        journal.append("layer", block_index, layer_plan.layer.path, committed.reference.artifact_id, identity)
        peak = int(torch.cuda.max_memory_allocated(request.device)) if request.device.startswith("cuda") else 0
        return ResidentFactorizationSliceResult(
            layer_result,
            identity,
            time.perf_counter() - started,
            peak,
            len(pending) - 1,
        )
    finally:
        executor.release()
        events.close()


def run_resident_factorization_slice(request: ResidentQuantizationRequest) -> ResidentFactorizationSliceResult:
    if request.device.startswith("cuda"):
        with acquire_device_lease(request.device), _legacy_cuda_numerics():
            return _run_resident_factorization_slice(request)
    return _run_resident_factorization_slice(request)


def run_resident_quantization(request: ResidentQuantizationRequest) -> ResidentQuantizationResult:
    """Run with an exclusive cross-process lease for CUDA resident state."""
    if request.device.startswith("cuda"):
        with acquire_device_lease(request.device), _legacy_cuda_numerics():
            return _run_resident_quantization(request)
    return _run_resident_quantization(request)
