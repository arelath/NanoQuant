"""Auditable resident quantization composition for pinned Transformers snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from torch import nn

from nanoquant.application.assembly import assemble_frozen_model
from nanoquant.application.calibration import MaterializedLayerCalibration, calibrate_block, calibrate_causal_model
from nanoquant.application.calibration_artifacts import (
    PersistedCalibration,
    PersistedObjectives,
    build_objectives,
    persist_calibration,
)
from nanoquant.application.covariance import SplitDenseCovarianceAccumulator
from nanoquant.application.device_batches import iter_device_batches
from nanoquant.application.kl_budget import (
    interaction_corrected_unit_kl_anchors,
    kl_calibrated_sensitivities,
    load_kl_budget_profile,
    measured_unit_kl_anchors,
    validate_kl_budget_profile,
)
from nanoquant.application.layers import (
    BlockEditor,
    LayerFreezer,
    SharedInputGroupFreezer,
    TrainableFactorizedLinear,
    TrainableSharedInputFactorGroup,
    freeze_block_auxiliary_parameters,
    restore_block_auxiliary_parameters,
)
from nanoquant.application.loss_snapshots import BlockLossRecorder, normalized_activation_error
from nanoquant.application.planning import PersistedPlan, PlanningRequest, build_quantization_plan, persist_plan
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.application.quantization_stages import (
    BiasCorrectionStage,
    BiasCorrectionStageRequest,
    FactorizationAttemptStage,
    LowRankPatchStage,
    LowRankPatchStageRequest,
    MaterializedScaleFitStageRequest,
    OutlierSelectionStage,
    ScaleFitStage,
)
from nanoquant.application.reconstruction_report import render_reconstruction_tables
from nanoquant.application.retry_loop import AcceptedFactorization, run_factorization_attempts
from nanoquant.application.stages import StageContext, execute_stage
from nanoquant.application.tuning import (
    EpochLossMode,
    TuningRequest,
    TuningResumeState,
    post_block_refit,
    tune_factorized,
    tune_non_factorized,
)
from nanoquant.config.codec import canonical_json, from_dict, to_dict
from nanoquant.config.schema import (
    MEASURED_UNIT_KL_OBJECTIVES,
    ActivationGpuCacheMode,
    ADMMConfig,
    AllocationStrategy,
    BiasCorrectionConfig,
    ExecutorKind,
    KlAllocationObjective,
    KlSensitivityGranularity,
    LayerRankBudgetConfig,
    LowRankPatchConfig,
    ObjectiveConfig,
    ObservabilityConfig,
    OutlierConfig,
    ProfilingConfig,
    ProfilingLevel,
    RankAllocationConfig,
    RankBoundsConfig,
    RankResponseCurveConfig,
    RankResponseSegmentConfig,
    RankResponseSource,
    RankRetryConfig,
    ReconstructionImportanceConfig,
    ReconstructionRankPlanningConfig,
    RetryThresholdConfig,
    RunConfig,
    ScaleFitConfig,
    SharedInputGroupConfig,
)
from nanoquant.domain.calibration_math import weighted_group_output_importance
from nanoquant.domain.factorization import factorize_admm
from nanoquant.domain.metrics import reconstruction_metrics
from nanoquant.domain.models import (
    ArtifactRef,
    ArtifactTypes,
    BiasCorrectionResult,
    BlockPlan,
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
    FrozenSharedInputGroupState,
    LayerId,
    LayerInventory,
    LayerPlan,
    LayerResult,
    LowRankPatchResult,
    ModelInventory,
    ObjectiveSpec,
    OutlierSelectionRequest,
    OutlierSelectionResult,
    QuantizationPlan,
    ReconstructionRankDecision,
    ScaleFitRequest,
    ScaleFitResult,
    SharedInputGroupCandidate,
    SharedInputGroupPlan,
    SharedInputGroupResult,
    SourceTensor,
    TensorId,
    TensorRef,
    TensorSpec,
)
from nanoquant.domain.outliers import reconstruct_with_outliers
from nanoquant.domain.planning import (
    RankResponseSegment,
    ReconstructionAllocationUnit,
    allocate_reconstruction_rank_budget,
    apply_reconstruction_rank_trust_region,
)
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
from nanoquant.domain.resources import (
    ResolvedMemoryPlan,
    ResourceAdmissionError,
    select_fastest_observed_batch,
    throughput_batch_candidates,
)
from nanoquant.domain.runs import BudgetState, RunManifest, RunStatus
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import (
    CommitIdentity,
    commit_block,
    commit_layer,
    commit_shared_input_group,
    latest_complete_identity,
    load_block_activations,
    load_committed_block,
    load_committed_layer,
    load_committed_shared_input_group,
    retire_block_activations,
)
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.device_memory import (
    PeakWindow,
    is_cuda_oom,
    release_cached_host_memory,
    sample_device_memory,
)
from nanoquant.infrastructure.environment import capture_environment
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.live_reconstruction import update_live_weight_error_report
from nanoquant.infrastructure.model_adapters import TransformersModelAdapter, adapter_for_config
from nanoquant.infrastructure.profiling import profiled_run
from nanoquant.infrastructure.progress_journal import JournalRecord, ProgressJournal
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.resource_planning import (
    load_memory_plan_revision,
    revise_resident_memory_plan_for_throughput,
)
from nanoquant.infrastructure.resource_usage import peak_device_memory_bytes, peak_process_memory_bytes
from nanoquant.infrastructure.run_session import open_run_session
from nanoquant.infrastructure.runs import (
    RunDirectory,
    initial_manifest_from_resolved,
    launcher_provenance,
    transition,
)
from nanoquant.infrastructure.runtime_export import load_frozen_run_planning_reference
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.infrastructure.tuning_checkpoint import (
    TuningCheckpointIdentity,
    active_tuning_checkpoint,
    clear_tuning_checkpoint,
    save_tuning_checkpoint,
)
from nanoquant.ports.event_sink import EventSink
from nanoquant.ports.model_adapter import ModelAdapter

RESIDENT_ALGORITHM_VERSION = 47
_THROUGHPUT_PROBE_REPETITIONS = 5


def _release_throughput_probe_caches(device: str) -> None:
    """Prevent candidate probes from accumulating allocator state across trials."""

    if not device.startswith("cuda"):
        return
    torch.cuda.empty_cache()
    release_cached_host_memory()


@dataclass(frozen=True, slots=True)
class _ActivePreprocessingState:
    schema_version: int
    resident_config_hash: str
    calibration: ArtifactRef
    objectives: ArtifactRef
    plan: ArtifactRef


@contextmanager
def _logged_operation(
    events: EventSink,
    operation: str,
    **fields: object,
) -> Iterator[None]:
    """Emit a bounded lifecycle pair and preserve failure location/context."""

    started = time.perf_counter()
    cast(Any, events).emit(
        "resident-quantization",
        "info",
        f"{operation}.started",
        **fields,
    )
    try:
        yield
    except BaseException as exc:
        if not hasattr(exc, "nanoquant_operation"):
            try:
                exc.__dict__["nanoquant_operation"] = operation
            except Exception:
                pass
        cast(Any, events).emit(
            "resident-quantization",
            "error",
            f"{operation}.failed",
            wall_seconds=time.perf_counter() - started,
            error_type=type(exc).__name__,
            error=str(exc),
            **fields,
        )
        raise
    else:
        cast(Any, events).emit(
            "resident-quantization",
            "info",
            f"{operation}.completed",
            wall_seconds=time.perf_counter() - started,
            **fields,
        )


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


@contextmanager
def _profile_block_phase(
    recorder: PhaseRecorder,
    block: int,
    name: str,
) -> Iterator[None]:
    with recorder.phase("blocks"):
        with recorder.phase("block", block=block):
            with recorder.phase(name):
                yield


@contextmanager
def _profile_layer_phase(
    recorder: PhaseRecorder,
    block: int,
    layer: str,
    name: str,
) -> Iterator[None]:
    with recorder.phase("blocks"):
        with recorder.phase("block", block=block):
            with recorder.phase("layer", layer=layer):
                with recorder.phase(name):
                    yield


_DEFAULT_RESIDENT_RANK_RETRY = RankRetryConfig(
    enabled=True,
    thresholds=RetryThresholdConfig(weighted_normalized_error=0.5, raw_normalized_error=0.5),
    rank_increase_fraction=0.25,
    maximum_attempts=3,
    extra_bit_budget_fraction=0.02,
    allow_above_allocator_cap=True,
)


@dataclass(frozen=True, slots=True)
class ResidentQuantizationRequest:
    snapshot: Path
    output: Path
    source: str
    revision: str
    token_ids: torch.Tensor | tuple[tuple[int, ...], ...]
    device: str = "cuda"
    executor: ExecutorKind = ExecutorKind.RESIDENT
    activation_gpu_cache: ActivationGpuCacheMode = ActivationGpuCacheMode.OFF
    activation_gpu_reserve_bytes: int = 2**30
    target_bpw: float = 1.0
    rank_multiple: int = 32
    allocation_strategy: AllocationStrategy = AllocationStrategy.SENSITIVITY
    rank_floor_fraction: float = 0.5
    rank_ceiling_fraction: float = 4.5
    rank_sensitivity_alpha: float = 0.5
    rank_edge_boost: float = 0.0
    maximum_rank_layer_patterns: tuple[str, ...] = ()
    layer_budget_multipliers: tuple[LayerRankBudgetConfig, ...] = ()
    rank_retry: RankRetryConfig = _DEFAULT_RESIDENT_RANK_RETRY
    reconstruction_rank_planning: ReconstructionRankPlanningConfig = ReconstructionRankPlanningConfig()
    kl_profile_artifact: str | None = None
    kl_profile_key: str | None = None
    kl_sensitivity_granularity: KlSensitivityGranularity = KlSensitivityGranularity.EXACT_OR_TYPE_BLOCK
    layer_order: tuple[str, ...] = ()
    shared_input_groups: tuple[SharedInputGroupConfig, ...] = ()
    admm: ADMMConfig = ADMMConfig(outer_iterations=1, inner_iterations=1)
    outliers: OutlierConfig = OutlierConfig()
    scale_fit: ScaleFitConfig = ScaleFitConfig(enabled=False)
    bias_correction: BiasCorrectionConfig = BiasCorrectionConfig()
    low_rank_patch: LowRankPatchConfig = LowRankPatchConfig()
    factorized_tuning_epochs: int = 0
    factorized_tuning_batch_size: int = 8
    factorized_tuning_learning_rate: float = 1e-5
    factorized_tuning_epoch_cooldown_seconds: float = 0.0
    initial_cooldown_seconds: float = 0.0
    nonfactorized_tuning_epochs: int = 0
    nonfactorized_tuning_epochs_by_layer: tuple[int, ...] = ()
    nonfactorized_tuning_batch_size: int = 8
    nonfactorized_tuning_learning_rate: float = 1e-4
    nonfactorized_tuning_epoch_cooldown_seconds: float = 0.0
    nonfactorized_tuning_early_stop_relative_tolerance: float | None = None
    post_block_refit_epochs: int = 0
    post_block_refit_batch_size: int = 8
    post_block_refit_learning_rate: float = 1e-5
    post_block_refit_epoch_cooldown_seconds: float = 0.0
    tuning_microbatch_size: int | None = None
    post_block_refit_microbatch_size: int | None = None
    legacy_tuning_seed_reset: bool = False
    restore_best_tuning_state: bool = True
    tuning_epoch_loss_mode: EpochLossMode = "full_evaluation"
    activation_retention: str = "rolling"
    seed: int = 0
    verify_hashes: bool = True
    interrupt_after_layer_commits: int | None = None
    interrupt_after_rank_probe_commits: int | None = None
    interrupt_after_block_commits: int | None = None
    interrupt_after_factorized_tuning_epoch_commits: int | None = None
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
    profiling: ProfilingConfig = ProfilingConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    maximum_wddm_shared_bytes: int | None = None
    registry_root: Path | None = None
    run_config: RunConfig | None = None
    launcher_path: Path | None = None
    defer_run_completion: bool = False
    memory_plan: ResolvedMemoryPlan | None = None
    memory_plan_reference: ArtifactRef | None = None


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


def _release_uncompleted_decoder_blocks(
    layers: nn.ModuleList,
    completed_block_indexes: set[int],
) -> int:
    """Release dense source blocks that will be streamed from the checkpoint."""
    released = 0
    for index in range(len(layers)):
        if index not in completed_block_indexes:
            layers[index] = nn.Identity()
            released += 1
    return released


def _place_completed_decoder_block(
    layers: nn.ModuleList,
    index: int,
    block: nn.Module,
    *,
    retain: bool,
) -> bool:
    """Keep a completed block only when a later in-process full-model forward needs it."""

    if retain:
        layers[index] = block
        return True
    layers[index] = nn.Identity()
    return False


def _weighted_mse(prediction: torch.Tensor, target: torch.Tensor, importance: torch.Tensor) -> float:
    if importance.ndim != 1 or importance.shape[0] != prediction.shape[-1]:
        raise ValueError("block output importance does not match hidden width")
    weights = importance.to(device=prediction.device, dtype=torch.float32)
    total = torch.zeros((), device=prediction.device)
    for index in range(prediction.shape[0]):
        error = prediction[index].detach().float() - target[index].detach().float()
        total += (error.square() * weights).sum()
    return float(total / prediction.numel())


def _weighted_target_mean_square(target: torch.Tensor, importance: torch.Tensor) -> float:
    """Measure teacher signal power with bounded temporary memory."""

    if importance.ndim != 1 or importance.shape[0] != target.shape[-1]:
        raise ValueError("block output importance does not match hidden width")
    weights = importance.to(device=target.device, dtype=torch.float32)
    hidden = target.shape[-1]
    rows = target.detach().reshape(-1, hidden)
    total = torch.zeros((), device=target.device)
    for start in range(0, rows.shape[0], 256):
        stop = min(start + 256, rows.shape[0])
        value = rows[start:stop].float()
        value.square_()
        value.mul_(weights)
        total += value.sum()
    return float(total / target.numel())


def _self_reference_weighted_mse(value: torch.Tensor, importance: torch.Tensor) -> float:
    # x - x == 0.0 for every finite IEEE-754 value, so the weighted MSE of a tensor
    # against itself is always exactly 0.0 unless it contains non-finite entries.
    if bool(torch.isfinite(value).all()):
        return 0.0
    return _weighted_mse(value, value, importance)


def _clone_forward_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Clone captured forward metadata so one decoder block cannot mutate the next block's inputs."""

    def clone(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return value

    return {key: clone(value) for key, value in metadata.items()}


def _forward_metadata_to_device(metadata: dict[str, object], device: str) -> dict[str, object]:
    """Move captured tensor metadata alongside one active offloaded block."""

    def move(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    return {key: move(value) for key, value in metadata.items()}


def _model_placement_device(request: ResidentQuantizationRequest) -> str:
    return "cpu" if request.executor is ExecutorKind.CPU_OFFLOAD else request.device


def _activation_cache_fits(required_bytes: int, free_bytes: int, reserve_bytes: int) -> bool:
    return required_bytes >= 0 and reserve_bytes >= 0 and required_bytes + reserve_bytes <= free_bytes


def _activation_cache_reserve_bytes(
    required_bytes: int,
    configured_reserve_bytes: int,
    *,
    automatic: bool,
) -> int:
    """Keep one activation-stream-sized workspace when automatic caching is enabled.

    The configured reserve remains authoritative for explicit cache policies.  In
    automatic mode, admitting two multi-GiB streams with only the schema's 1 GiB
    reserve can leave no room for gradients, optimizer state, or factorization
    workspaces.  One stream is a conservative, model-scaled lower bound that lets
    AUTO degrade from BOTH to INPUTS before tuning starts.
    """

    if automatic:
        return max(required_bytes, configured_reserve_bytes)
    return configured_reserve_bytes


def _cache_activation_tensor(
    value: torch.Tensor,
    request: ResidentQuantizationRequest,
    events: EventSink,
    *,
    role: str,
    required: bool,
) -> torch.Tensor:
    if not request.device.startswith("cuda") or value.device.type == "cuda":
        return value
    required_bytes = value.numel() * value.element_size()
    configured_reserve_bytes = request.activation_gpu_reserve_bytes
    reserve_bytes = _activation_cache_reserve_bytes(
        required_bytes,
        configured_reserve_bytes,
        automatic=not required,
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(request.device)
    if not _activation_cache_fits(required_bytes, int(free_bytes), reserve_bytes):
        cast(Any, events).emit(
            "resource",
            "warning" if required else "info",
            "activation_gpu_cache.skipped",
            role=role,
            required=required,
            required_bytes=required_bytes,
            configured_reserve_bytes=configured_reserve_bytes,
            reserve_bytes=reserve_bytes,
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
        )
        if required:
            raise RuntimeError(
                f"activation GPU cache for {role} requires {required_bytes} bytes plus "
                f"{reserve_bytes} reserved bytes; only {int(free_bytes)} bytes are free"
            )
        return value
    cached = value.to(request.device)
    cast(Any, events).emit(
        "resource",
        "info",
        "activation_gpu_cache.loaded",
        role=role,
        bytes=required_bytes,
        configured_reserve_bytes=configured_reserve_bytes,
        reserve_bytes=reserve_bytes,
        free_bytes_before=int(free_bytes),
    )
    return cached


def _block_loss(
    adapter: Any,
    block: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    output_importance: torch.Tensor,
    metadata: dict[str, object],
    batch_size: int,
    recorder: PhaseRecorder = NULL_RECORDER,
) -> float:
    block_device = next(iter(block.parameters()), None)
    device = inputs.device if block_device is None else block_device.device
    weights = output_importance.to(device=device, dtype=torch.float32)

    def accumulate(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Converting the complete activation batch to fp32 used several hidden-width
        # temporaries at once.  On Gemma 3 4B that could request another 512 MiB after
        # a layer had already committed.  Stream token rows and reuse the error buffer
        # so diagnostic loss memory is independent of sequence and batch length.
        hidden = prediction.shape[-1]
        prediction_rows = prediction.detach().reshape(-1, hidden)
        target_rows = target.detach().reshape(-1, hidden)
        total = torch.zeros((), device=device)
        for start in range(0, prediction_rows.shape[0], 256):
            stop = min(start + 256, prediction_rows.shape[0])
            error = prediction_rows[start:stop].float()
            error.sub_(target_rows[start:stop])
            error.square_()
            error.mul_(weights)
            total += error.sum()
        return total

    with torch.no_grad():
        squared_error = torch.zeros((), device=device)
        elements = 0
        if recorder is not NULL_RECORDER:
            batches = iter(iter_device_batches((inputs, targets), batch_size, device))
            batch_count = (inputs.shape[0] + batch_size - 1) // batch_size
            for _batch_index in range(batch_count):
                with recorder.phase("batch_stage"):
                    input_batch, target = next(batches)
                    if inputs.device.type == "cpu" and device.type == "cuda":
                        recorder.add(
                            "transfer.h2d_bytes",
                            input_batch.numel() * input_batch.element_size() + target.numel() * target.element_size(),
                        )
                with recorder.phase("forward"):
                    prediction = adapter.run_block(block, input_batch, **metadata)
                with recorder.phase("loss"):
                    squared_error += accumulate(prediction, target)
                    elements += target.numel()
                recorder.add("forward.batches", 1)
                recorder.add("forward.elements", target.numel())
            with recorder.phase("synchronize"):
                return float(squared_error / elements)
        for input_batch, target in iter_device_batches((inputs, targets), batch_size, device):
            prediction = adapter.run_block(block, input_batch, **metadata)
            squared_error += accumulate(prediction, target)
            elements += target.numel()
        return float(squared_error / elements)


@torch.no_grad()
def _run_block_batched(
    adapter: Any,
    block: nn.Module,
    inputs: torch.Tensor,
    metadata: dict[str, object],
    batch_size: int,
    storage_device: str | torch.device | None = None,
    recorder: PhaseRecorder = NULL_RECORDER,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("block forward batch size must be positive")
    block_parameter = next(iter(block.parameters()), None)
    compute_device = inputs.device if block_parameter is None else block_parameter.device
    destination = inputs.device if storage_device is None else torch.device(storage_device)
    result: torch.Tensor | None = None
    if recorder is not NULL_RECORDER:
        batches = iter(iter_device_batches((inputs,), batch_size, compute_device))
        batch_count = (inputs.shape[0] + batch_size - 1) // batch_size
        for batch_index in range(batch_count):
            with recorder.phase("batch_stage"):
                (input_batch,) = next(batches)
                if inputs.device.type == "cpu" and compute_device.type == "cuda":
                    recorder.add(
                        "transfer.h2d_bytes",
                        input_batch.numel() * input_batch.element_size(),
                    )
            start = batch_index * batch_size
            end = min(start + batch_size, inputs.shape[0])
            with recorder.phase("forward"):
                output = adapter.run_block(block, input_batch, **metadata)
            if result is None:
                shape = (inputs.shape[0], *output.shape[1:])
                result = torch.empty(shape, device=destination, dtype=output.dtype)
            with recorder.phase("d2h" if destination.type == "cpu" and output.device.type == "cuda" else "store"):
                result[start:end].copy_(output)
                if destination.type == "cpu" and output.device.type == "cuda":
                    recorder.add(
                        "transfer.d2h_bytes",
                        output.numel() * output.element_size(),
                    )
            recorder.add("forward.batches", 1)
            recorder.add("forward.elements", output.numel())
        if result is None:
            raise ValueError("cannot run a block over empty inputs")
        return result
    for batch_index, (input_batch,) in enumerate(iter_device_batches((inputs,), batch_size, compute_device)):
        start = batch_index * batch_size
        end = min(start + batch_size, inputs.shape[0])
        output = adapter.run_block(block, input_batch, **metadata)
        if result is None:
            shape = (inputs.shape[0], *output.shape[1:])
            result = torch.empty(shape, device=destination, dtype=output.dtype)
        result[start:end].copy_(output)
    if result is None:
        raise ValueError("cannot run a block over empty inputs")
    return result


def _low_rank_patch_selected(request: ResidentQuantizationRequest, layer_path: str) -> bool:
    return request.low_rank_patch.enabled and any(
        fnmatchcase(layer_path, pattern) for pattern in request.low_rank_patch.layer_patterns
    )


@torch.no_grad()
def _capture_low_rank_patch_statistics(
    request: ResidentQuantizationRequest,
    adapter: Any,
    block: nn.Module,
    layer_path: str,
    inputs: torch.Tensor,
    metadata: dict[str, object],
    tensors: LocalTensorStore,
    in_features: int,
) -> tuple[TensorRef, TensorRef, TensorRef, TensorRef]:
    module = block.get_submodule(layer_path)
    accumulator = SplitDenseCovarianceAccumulator(
        in_features,
        request.low_rank_patch.fit_tokens,
        request.low_rank_patch.held_out_tokens,
        device=request.device,
    )

    def capture(_module: nn.Module, positional: tuple[object, ...]) -> None:
        if not positional or not isinstance(positional[0], torch.Tensor):
            raise TypeError(f"low-rank patch input is not a tensor: {layer_path}")
        accumulator.update(positional[0])

    rows_per_sample = math.prod(inputs.shape[1:-1]) if inputs.ndim > 2 else 1
    required_rows = request.low_rank_patch.fit_tokens + request.low_rank_patch.held_out_tokens
    required_samples = math.ceil(required_rows / rows_per_sample)
    if required_samples > inputs.shape[0]:
        raise ValueError(
            f"low-rank patch requires {required_rows} activation rows but only "
            f"{inputs.shape[0] * rows_per_sample} are available for {layer_path}"
        )
    handle = module.register_forward_pre_hook(capture)
    try:
        output = _run_block_batched(
            adapter,
            block,
            inputs[:required_samples],
            metadata,
            min(request.block_forward_batch_size, required_samples),
            "cpu",
        )
        del output
    finally:
        handle.remove()
    if not accumulator.complete:
        raise ValueError(f"low-rank patch activation capture was incomplete for {layer_path}")
    fit_covariance, fit_mean = accumulator.fit.materialize()
    held_out_covariance, held_out_mean = accumulator.held_out.materialize()
    refs = tensors.put(
        "low-rank-patch-calibration",
        {
            "fit_covariance": fit_covariance,
            "held_out_covariance": held_out_covariance,
            "fit_input_mean": fit_mean,
            "held_out_input_mean": held_out_mean,
        },
    )
    return (
        refs["fit_covariance"],
        refs["held_out_covariance"],
        refs["fit_input_mean"],
        refs["held_out_input_mean"],
    )


def _inventory_block_elements(block: Any) -> int:
    return sum(math.prod(tensor.spec.shape) for tensor in block.source_tensors)


@torch.no_grad()
def _autotune_block_forward_batch(
    request: ResidentQuantizationRequest,
    adapter: TransformersModelAdapter,
    source: SafetensorsModelSource,
    inventory: ModelInventory,
    decoder_layers: nn.ModuleList,
    inputs: torch.Tensor,
    captured_metadata: dict[str, object],
    events: EventSink,
) -> int:
    """Benchmark safe forward candidates without changing numerical semantics."""

    if (
        request.memory_plan is None
        or request.memory_plan.mode != "adaptive"
        or inputs.shape[0] < 2
        or any(
            warning.startswith("block_forward selected measured-throughput")
            for warning in request.memory_plan.warnings
        )
    ):
        return request.block_forward_batch_size
    maximum_safe = request.block_forward_batch_size
    configured = (
        maximum_safe
        if request.run_config is None
        else request.run_config.runtime.block_forward_batch_size
    )
    baseline = min(maximum_safe, configured)
    candidates = throughput_batch_candidates(maximum_safe, baseline)
    benchmark_samples = min(int(inputs.shape[0]), max(64, maximum_safe))
    benchmark_inputs = inputs[:benchmark_samples]
    streamed_block = request.executor is ExecutorKind.CPU_OFFLOAD
    representative = max(inventory.blocks, key=_inventory_block_elements)
    block = (
        adapter.load_block(source, representative.block, request.device)
        if streamed_block
        else decoder_layers[representative.block.index]
    )
    metadata = _forward_metadata_to_device(_clone_forward_metadata(captured_metadata), request.device)
    observations: list[tuple[int, float]] = []
    failures: list[dict[str, object]] = []
    output: torch.Tensor | None = None
    try:
        output = _run_block_batched(adapter, block, benchmark_inputs, metadata, baseline, "cpu")
        if request.device.startswith("cuda"):
            torch.cuda.synchronize(request.device)
        output = None
        _release_throughput_probe_caches(request.device)
        for batch_size in candidates:
            samples: list[float] = []
            try:
                for _ in range(_THROUGHPUT_PROBE_REPETITIONS):
                    started = time.perf_counter()
                    output = _run_block_batched(
                        adapter,
                        block,
                        benchmark_inputs,
                        metadata,
                        batch_size,
                        "cpu",
                    )
                    if request.device.startswith("cuda"):
                        torch.cuda.synchronize(request.device)
                    samples.append(time.perf_counter() - started)
                    del output
                    _release_throughput_probe_caches(request.device)
                observations.append((batch_size, statistics.median(samples)))
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                failures.append({"batch_size": batch_size, "error": str(exc)})
                _release_throughput_probe_caches(request.device)
        successful_batches = {batch for batch, _ in observations}
        if baseline not in successful_batches:
            raise ResourceAdmissionError(
                f"RES001 adaptive throughput baseline batch {baseline} failed its real block-forward probe"
            )
        selected = select_fastest_observed_batch(
            tuple(observations),
            baseline_batch=baseline,
            minimum_improvement_fraction=0.05,
        )
        baseline_seconds = dict(observations)[baseline]
        selected_seconds = dict(observations)[selected]
        cast(Any, events).emit(
            "memory",
            "info",
            "memory.throughput_autotuned",
            stage_name="block_forward",
            maximum_safe_batch_size=maximum_safe,
            baseline_batch_size=baseline,
            selected_batch_size=selected,
            benchmark_samples=benchmark_samples,
            selected_speedup=baseline_seconds / selected_seconds,
            observations=[{"batch_size": batch, "seconds": seconds} for batch, seconds in observations],
            failed_candidates=failures,
        )
        return selected
    finally:
        metadata = {}
        if streamed_block:
            del block
        if request.device.startswith("cuda"):
            torch.cuda.empty_cache()


def _autotune_tuning_microbatch(
    request: ResidentQuantizationRequest,
    adapter: TransformersModelAdapter,
    source: SafetensorsModelSource,
    inventory: ModelInventory,
    decoder_layers: nn.ModuleList,
    inputs: torch.Tensor,
    captured_metadata: dict[str, object],
    events: EventSink,
) -> tuple[int | None, int | None]:
    """Measure forward/backward throughput over admitted tuning microbatches."""

    maximum_safe = request.tuning_microbatch_size
    if (
        request.memory_plan is None
        or request.memory_plan.mode != "adaptive"
        or maximum_safe is None
        or inputs.shape[0] < 2
        or not (
            request.factorized_tuning_epochs > 0
            or request.nonfactorized_tuning_epochs > 0
            or any(request.nonfactorized_tuning_epochs_by_layer)
            or request.post_block_refit_epochs > 0
        )
        or any(
            warning.startswith("tuning selected measured-throughput")
            for warning in request.memory_plan.warnings
        )
    ):
        return request.tuning_microbatch_size, request.post_block_refit_microbatch_size
    configured = (
        maximum_safe
        if request.run_config is None or request.run_config.block_tuning.microbatch_size is None
        else request.run_config.block_tuning.microbatch_size
    )
    baseline = min(maximum_safe, configured)
    logical_batch = max(request.factorized_tuning_batch_size, request.nonfactorized_tuning_batch_size)
    benchmark_samples = min(int(inputs.shape[0]), max(maximum_safe, logical_batch))
    benchmark_inputs = inputs[:benchmark_samples]
    streamed_block = request.executor is ExecutorKind.CPU_OFFLOAD
    representative = max(inventory.blocks, key=_inventory_block_elements)
    block = (
        adapter.load_block(source, representative.block, request.device)
        if streamed_block
        else decoder_layers[representative.block.index]
    )
    metadata = _forward_metadata_to_device(_clone_forward_metadata(captured_metadata), request.device)
    observations: list[tuple[int, float]] = []
    failures: list[dict[str, object]] = []
    original_requires_grad = {id(parameter): parameter.requires_grad for parameter in block.parameters()}
    for parameter in block.parameters():
        parameter.requires_grad_(True)
    targets: torch.Tensor | None = None
    try:
        targets = _run_block_batched(
            adapter,
            block,
            benchmark_inputs,
            metadata,
            baseline,
            "cpu",
        ).detach()
        benchmark_targets = targets

        def benchmark_candidate(batch_size: int) -> float:
            block.zero_grad(set_to_none=True)
            started = time.perf_counter()
            for start in range(0, benchmark_samples, batch_size):
                stop = min(start + batch_size, benchmark_samples)
                input_batch = benchmark_inputs[start:stop].to(request.device)
                target_batch = benchmark_targets[start:stop].to(request.device)
                prediction = adapter.run_block(block, input_batch, **metadata)
                loss = (prediction.float() - target_batch.float()).square().mean()
                torch.autograd.backward(loss)
                del input_batch, target_batch, prediction, loss
            if request.device.startswith("cuda"):
                torch.cuda.synchronize(request.device)
            return time.perf_counter() - started

        benchmark_candidate(baseline)
        _release_throughput_probe_caches(request.device)
        for batch_size in throughput_batch_candidates(maximum_safe, baseline):
            samples: list[float] = []
            try:
                for _ in range(_THROUGHPUT_PROBE_REPETITIONS):
                    samples.append(benchmark_candidate(batch_size))
                    _release_throughput_probe_caches(request.device)
                observations.append((batch_size, statistics.median(samples)))
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                block.zero_grad(set_to_none=True)
                failures.append({"batch_size": batch_size, "error": str(exc)})
                _release_throughput_probe_caches(request.device)
        if baseline not in {batch for batch, _ in observations}:
            raise ResourceAdmissionError(
                f"RES001 adaptive tuning baseline microbatch {baseline} failed its real probe"
            )
        selected = select_fastest_observed_batch(
            tuple(observations),
            baseline_batch=baseline,
            minimum_improvement_fraction=0.05,
        )
        timings = dict(observations)
        cast(Any, events).emit(
            "memory",
            "info",
            "memory.throughput_autotuned",
            stage_name="tuning",
            maximum_safe_batch_size=maximum_safe,
            baseline_batch_size=baseline,
            selected_batch_size=selected,
            benchmark_samples=benchmark_samples,
            selected_speedup=timings[baseline] / timings[selected],
            observations=[{"batch_size": batch, "seconds": seconds} for batch, seconds in observations],
            failed_candidates=failures,
        )
        refit_maximum = request.post_block_refit_microbatch_size
        selected_refit = None if refit_maximum is None else min(refit_maximum, selected)
        return selected, selected_refit
    finally:
        block.zero_grad(set_to_none=True)
        for parameter in block.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
        targets = None
        if streamed_block:
            del block
        if request.device.startswith("cuda"):
            torch.cuda.empty_cache()


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
                shape = (tokens.shape[0], *output.shape[1:])
                result = torch.empty(shape, device=destination, dtype=output.dtype)
            result[start:end].copy_(output)
    if result is None:
        raise ValueError("cannot run a prefix over empty tokens")
    return result


_QUALITY_CHUNK_POSITIONS = 256


@torch.no_grad()
def _run_quality_logits_batched(
    adapter: ModelAdapter,
    model: nn.Module,
    tokens: torch.Tensor,
    storage_device: str | torch.device,
) -> torch.Tensor:
    """Run the full model one sequence at a time and park the logits off-device.

    Quality logits are (samples, sequence, vocabulary); for small models with
    large vocabularies they dwarf every other allocation in the run, so they
    are never materialized in a single forward or kept on the compute device.
    The result stays pageable: pinning multi-gigabyte logits can destabilize
    the host, and this tensor is read back exactly once per run.
    """
    destination = torch.device(storage_device)
    result: torch.Tensor | None = None
    for index in range(tokens.shape[0]):
        output = adapter.run_full_forward(model, tokens[index : index + 1]).detach()
        if result is None:
            result = torch.empty((tokens.shape[0], *output.shape[1:]), device=destination, dtype=output.dtype)
        result[index].copy_(output[0])
        # Drop the device allocation before Python evaluates the next forward;
        # rebinding ``output`` alone would otherwise keep the previous logits
        # alive until the following forward has already completed.
        del output
    if result is None:
        raise ValueError("cannot capture quality logits over empty tokens")
    return result


@torch.no_grad()
def _streamed_quality_metrics(
    adapter: ModelAdapter,
    model: nn.Module,
    tokens: torch.Tensor,
    reference_logits: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Return (reference NLL, compressed NLL, logit MSE, argmax agreement).

    The compressed forward and every float32 comparison run one sequence at a
    time in bounded position chunks, so peak device memory stays independent
    of the quality sample count and vocabulary size.
    """
    parameter = next(iter(model.parameters()), None)
    device = tokens.device if parameter is None else parameter.device
    reference_nll_sum = torch.zeros((), device=device)
    compressed_nll_sum = torch.zeros((), device=device)
    squared_error_sum = torch.zeros((), device=device)
    argmax_matches = torch.zeros((), device=device)
    for index in range(tokens.shape[0]):
        compressed = adapter.run_full_forward(model, tokens[index : index + 1]).detach()[0]
        targets = tokens[index, 1:]
        positions = compressed.shape[0]
        for start in range(0, positions, _QUALITY_CHUNK_POSITIONS):
            end = min(start + _QUALITY_CHUNK_POSITIONS, positions)
            compressed_chunk = compressed[start:end].float()
            reference_chunk = reference_logits[index, start:end].to(device=device, dtype=torch.float32)
            argmax_matches += (compressed_chunk.argmax(-1) == reference_chunk.argmax(-1)).sum()
            squared_error_sum += (compressed_chunk - reference_chunk).square().sum()
            loss_end = min(end, positions - 1)
            if start < loss_end:
                reference_nll_sum += torch.nn.functional.cross_entropy(
                    reference_chunk[: loss_end - start], targets[start:loss_end], reduction="sum"
                )
                compressed_nll_sum += torch.nn.functional.cross_entropy(
                    compressed_chunk[: loss_end - start], targets[start:loss_end], reduction="sum"
                )
            del compressed_chunk, reference_chunk
        # Ensure the next forward cannot overlap this sequence's full logits.
        del compressed
    nll_positions = tokens.shape[0] * max(0, tokens.shape[1] - 1)
    reference_nll = float(reference_nll_sum / nll_positions) if nll_positions else float("nan")
    compressed_nll = float(compressed_nll_sum / nll_positions) if nll_positions else float("nan")
    logit_mse = float(squared_error_sum / reference_logits.numel())
    argmax_agreement = float(argmax_matches / (tokens.shape[0] * tokens.shape[1]))
    return reference_nll, compressed_nll, logit_mse, argmax_agreement


def _artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _ensure_block_commit_disk_capacity(
    request: ResidentQuantizationRequest,
    teacher_outputs: torch.Tensor,
    compressed_outputs: torch.Tensor,
) -> tuple[int, int, int]:
    """Recheck live disk pressure before writing a resumable activation generation."""

    if request.memory_plan is None:
        return 0, 0, 0
    # The content-addressed writer must temporarily coexist with the current
    # generation. Only these two tensors are additional at this boundary; add
    # bounded metadata/filesystem slack as a largest-write guard.
    required = (
        teacher_outputs.numel() * teacher_outputs.element_size()
        + compressed_outputs.numel() * compressed_outputs.element_size()
        + 64 * 2**20
    )
    free = int(shutil.disk_usage(request.output.resolve()).free)
    reserve = request.memory_plan.envelope.temporary_disk_reserve_bytes
    safe_capacity = max(0, free - reserve)
    if required > safe_capacity:
        raise ResourceAdmissionError(
            "RES001 block commit requires "
            f"{required} additional temporary disk bytes but live safe capacity is {safe_capacity} "
            f"({free} free minus {reserve} reserved); relocate or reclaim the artifact store before resume"
        )
    return required, free, reserve


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


def _reconstruction_importance_policy(
    member_layers: tuple[LayerId, ...],
    importance: ReconstructionImportanceConfig,
) -> tuple[dict[LayerId, float], frozenset[LayerId], frozenset[int]]:
    multiplier_matches = {
        layer: tuple(rule for rule in importance.layer_multipliers if fnmatchcase(layer.path, rule.pattern))
        for layer in member_layers
    }
    ambiguous = {layer: matches for layer, matches in multiplier_matches.items() if len(matches) > 1}
    if ambiguous:
        raise ValueError(
            "multiple reconstruction importance patterns matched logical members: "
            + ", ".join(f"{layer}={tuple(rule.pattern for rule in matches)!r}" for layer, matches in ambiguous.items())
        )
    matched_multiplier_patterns = {matches[0].pattern for matches in multiplier_matches.values() if matches}
    unmatched_multiplier_patterns = {
        rule.pattern for rule in importance.layer_multipliers
    } - matched_multiplier_patterns
    if unmatched_multiplier_patterns:
        raise ValueError(
            "reconstruction importance patterns matched no logical member: "
            f"{sorted(unmatched_multiplier_patterns)}"
        )
    matched_protected_patterns = {
        pattern
        for pattern in importance.protected_layer_patterns
        if any(fnmatchcase(layer.path, pattern) for layer in member_layers)
    }
    unmatched_protected_patterns = set(importance.protected_layer_patterns) - matched_protected_patterns
    if unmatched_protected_patterns:
        raise ValueError(
            "reconstruction protected patterns matched no logical member: "
            f"{sorted(unmatched_protected_patterns)}"
        )
    ordered_blocks = sorted({layer.block.index for layer in member_layers})
    edge_count = min(importance.protected_edge_block_count, len(ordered_blocks))
    edge_blocks = (
        frozenset((*ordered_blocks[:edge_count], *ordered_blocks[-edge_count:]))
        if edge_count
        else frozenset()
    )
    multipliers = {
        layer: (
            (multiplier_matches[layer][0].multiplier if multiplier_matches[layer] else 1.0)
            * (importance.edge_block_multiplier if layer.block.index in edge_blocks else 1.0)
        )
        for layer in member_layers
    }
    protected = frozenset(
        layer
        for layer in member_layers
        if layer.block.index in edge_blocks
        or any(fnmatchcase(layer.path, pattern) for pattern in importance.protected_layer_patterns)
    )
    return multipliers, protected, edge_blocks


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


def _epoch_cooldown_observer(
    seconds: float,
    events: EventSink | None = None,
    *,
    tuning_kind: str | None = None,
    block: int | None = None,
    layer: str | None = None,
    target_weighted_mean_square: float | None = None,
) -> Any:
    """Sleep after completed tuning epochs, never after the initial loss probe."""

    if not seconds and events is None:
        return None

    def observe(epoch: int, loss: float) -> None:
        if epoch > 0:
            if events is not None:
                events.emit(
                    "resident-quantization",
                    "info",
                    "tuning.epoch_completed",
                    tuning_kind=tuning_kind,
                    block=block,
                    layer=layer,
                    epoch=epoch,
                    loss=loss,
                    normalized_loss=(
                        None
                        if target_weighted_mean_square is None
                        else normalized_activation_error(loss, target_weighted_mean_square)
                    ),
                    target_weighted_mean_square=target_weighted_mean_square,
                )
            if seconds:
                time.sleep(seconds)

    return observe


def _tuning_seed(request: ResidentQuantizationRequest, stage: str, block: int, layer: str | None) -> int:
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
        patch_left=frozen.patch_left,
        patch_right=frozen.patch_right,
    ).to(device=device, dtype=dtype)


def _rehydrate_trainable_group(
    state: FrozenSharedInputGroupState,
    tensors: LocalTensorStore,
    *,
    device: str,
    dtype: torch.dtype,
) -> TrainableSharedInputFactorGroup:
    frozen = (
        SharedInputGroupFreezer()
        .load(
            state,
            tensors,
            device=device,
            dtype=dtype,
            backend="factorized",
        )
        .owner
    )
    return TrainableSharedInputFactorGroup(
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


def _materialize_shared_input_plan(
    group: SharedInputGroupPlan,
    working_block: nn.Module,
    tensors: LocalTensorStore,
    *,
    device: str,
) -> tuple[LayerPlan, TensorRef, tuple[TensorRef, ...]]:
    """Create one transient stacked source/objective without persisting factors."""

    source_values: list[torch.Tensor] = []
    source_refs: list[TensorRef] = []
    input_values: list[torch.Tensor] = []
    output_values: list[torch.Tensor] = []
    for member, objective in zip(group.members, group.objectives, strict=True):
        module = _module_at_path(working_block, member.layer.path)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"shared-input member has no materialized weight: {member.layer.path}")
        source_values.append(weight.detach().cpu())
        source_refs.append(tensors.put("source-layer", {"weight": weight.detach().cpu()})["weight"])
        with (
            tensors.read(objective.input_importance, device) as input_importance,
            tensors.read(objective.output_importance, device) as output_importance,
        ):
            input_values.append(input_importance.detach().float().cpu())
            output_values.append(output_importance.detach().float().cpu())
    canonical_input = input_values[0]
    for member, value in zip(group.members[1:], input_values[1:], strict=True):
        left = canonical_input / canonical_input.mean().clamp_min(1e-12)
        right = value / value.mean().clamp_min(1e-12)
        if not torch.allclose(left, right, rtol=1e-4, atol=1e-6):
            raise ValueError(f"shared-input calibration differs for group {group.name}: {member.layer.path}")
    stacked = torch.cat(source_values, dim=0).contiguous()
    multipliers = group.objective_multipliers or (1.0,) * len(group.members)
    concatenated_output = weighted_group_output_importance(tuple(output_values), multipliers)
    objective_refs = tensors.put(
        "shared-input-objective",
        {
            "input_importance": canonical_input.contiguous(),
            "output_importance": concatenated_output,
        },
    )
    source_ref = tensors.put("source-shared-input-group", {"weight": stacked})["weight"]
    pseudo_layer = LayerId(group.block, group.name)
    objective = replace(
        group.objectives[0],
        layer=pseudo_layer,
        input_importance=objective_refs["input_importance"],
        output_importance=objective_refs["output_importance"],
        covariance=None,
        target_weighted_norm_squared=None,
    )
    source = SourceTensor(
        TensorId(pseudo_layer, "weight"),
        "+".join(member.weight.source_key for member in group.members),
        "+".join(sorted({member.weight.shard for member in group.members})),
        TensorSpec(tuple(stacked.shape), group.members[0].weight.spec.dtype),
        "sha256:"
        + hashlib.sha256("|".join(member.weight.content_hash for member in group.members).encode()).hexdigest(),
    )
    return (
        LayerPlan(
            group.schema_version,
            pseudo_layer,
            source,
            group.rank,
            group.rank_multiple,
            group.allocator_cap,
            objective,
            group.outliers,
            group.retry,
            group.estimated_cost,
        ),
        source_ref,
        tuple(source_refs),
    )


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
    recorder: PhaseRecorder = NULL_RECORDER,
) -> tuple[AcceptedFactorization, OutlierSelectionResult, ScaleFitResult | None]:
    """Execute the complete legacy rank attempt: outliers, ADMM, scale fit, and full metric."""
    companions: list[tuple[OutlierSelectionResult, ScaleFitResult | None]] = []

    def execute_attempt(rank: int, attempt: int) -> FactorizationResult:
        shape = "x".join(str(dimension) for dimension in layer_plan.source_weight.spec.shape)
        with recorder.phase("attempt", rank=rank, attempt=attempt, shape=shape):
            return execute_attempt_body(rank, attempt)

    def execute_attempt_body(rank: int, attempt: int) -> FactorizationResult:
        started = time.perf_counter()
        attempt_window = PeakWindow(request.device).start()
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
        with recorder.phase("outliers"):
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
        probe_peak = peak_device_memory_bytes(request.device)
        context.events.emit(
            "resident-quantization",
            "debug",
            "probe.completed",
            probe_kind="salient_residual",
            block=layer_plan.layer.block.index,
            layer=layer_plan.layer.path,
            rank=rank,
            attempt=attempt,
            peak_device_bytes=probe_peak,
        )
        objective = replace(
            layer_plan.objective,
            input_importance=outliers.factor_input_importance,
        )
        with recorder.phase("admm"):
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
            with recorder.phase("scale_fit"):
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
        with recorder.phase("metrics"):
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
        attempt_window.finish()
        peak = (
            max(probe_peak, attempt_window.peak_allocated_bytes)
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


def _run_bias_correction(
    layer_plan: LayerPlan,
    source_weight: TensorRef,
    factorized: FactorizationResult,
    outliers: OutlierSelectionResult,
    request: ResidentQuantizationRequest,
    context: StageContext,
    stage: BiasCorrectionStage,
) -> BiasCorrectionResult | None:
    if not request.bias_correction.enabled:
        return None
    input_mean = layer_plan.objective.input_mean
    if input_mean is None:
        raise ValueError(f"bias correction requires an input mean for {layer_plan.layer}")
    return execute_stage(
        stage,
        BiasCorrectionStageRequest(
            layer_plan.layer,
            source_weight,
            factorized.factors.left_binary,
            factorized.factors.right_binary,
            factorized.factors.scales,
            input_mean,
            outliers.indices,
            outliers.values,
            outliers.scales,
        ),
        context,
    )


def _run_low_rank_patch(
    layer_plan: LayerPlan,
    source_weight: TensorRef,
    factorized: FactorizationResult,
    outliers: OutlierSelectionResult,
    bias_correction: BiasCorrectionResult | None,
    request: ResidentQuantizationRequest,
    adapter: Any,
    working_block: nn.Module,
    compressed_inputs: torch.Tensor,
    metadata: dict[str, object],
    context: StageContext,
    tensors: LocalTensorStore,
    stage: LowRankPatchStage,
) -> LowRankPatchResult | None:
    if not _low_rank_patch_selected(request, layer_plan.layer.path):
        return None
    fit_covariance, held_out_covariance, fit_mean, held_out_mean = _capture_low_rank_patch_statistics(
        request,
        adapter,
        working_block,
        layer_plan.layer.path,
        compressed_inputs,
        metadata,
        tensors,
        layer_plan.source_weight.spec.shape[1],
    )
    return execute_stage(
        stage,
        LowRankPatchStageRequest(
            layer_plan.layer,
            source_weight,
            factorized.factors.left_binary,
            factorized.factors.right_binary,
            factorized.factors.scales,
            fit_covariance,
            held_out_covariance,
            fit_mean,
            held_out_mean,
            None if bias_correction is None else bias_correction.bias,
            outliers.indices,
            outliers.values,
            outliers.scales,
        ),
        context,
    )


def _account_for_patch_acceptance(
    accepted: AcceptedFactorization,
    patch: LowRankPatchResult | None,
) -> AcceptedFactorization:
    actual_patch_bits = 0 if patch is None or not patch.accepted else patch.bit_cost.patch_bits
    if actual_patch_bits == accepted.actual_bit_cost.patch_bits:
        return accepted
    delta = actual_patch_bits - accepted.actual_bit_cost.patch_bits
    return replace(
        accepted,
        actual_bit_cost=replace(accepted.actual_bit_cost, patch_bits=actual_patch_bits),
        budget=replace(accepted.budget, accepted_bits=accepted.budget.accepted_bits + delta),
    )


def _resident_config_hash(request: ResidentQuantizationRequest) -> str:
    adaptive_memory = request.memory_plan is not None and request.memory_plan.mode == "adaptive"
    semantic_config = {
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
        "maximum_rank_layer_patterns": request.maximum_rank_layer_patterns,
        "layer_budget_multipliers": request.layer_budget_multipliers,
        "reconstruction_rank_planning": request.reconstruction_rank_planning,
        "kl_profile_artifact": request.kl_profile_artifact,
        "kl_profile_key": request.kl_profile_key,
        "kl_sensitivity_granularity": request.kl_sensitivity_granularity,
        "layer_order": request.layer_order,
        "shared_input_groups": request.shared_input_groups,
        "admm": request.admm,
        "outliers": request.outliers,
        "scale_fit": request.scale_fit,
        "bias_correction": request.bias_correction,
        "low_rank_patch": request.low_rank_patch,
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
        "tuning_microbatch_size": "adaptive" if adaptive_memory else request.tuning_microbatch_size,
        "post_block_refit_microbatch_size": (
            "adaptive" if adaptive_memory else request.post_block_refit_microbatch_size
        ),
        "block_forward_batch_size": "adaptive" if adaptive_memory else request.block_forward_batch_size,
        "legacy_tuning_seed_reset": request.legacy_tuning_seed_reset,
        "restore_best_tuning_state": request.restore_best_tuning_state,
        "tuning_epoch_loss_mode": request.tuning_epoch_loss_mode,
        "activation_retention": request.activation_retention,
        "calibration_method": request.calibration_method,
        "calibration_shrinkage": request.calibration_shrinkage,
        "calibration_batch_size": (
            "adaptive"
            if adaptive_memory and request.calibration_method == "forward_only"
            else request.calibration_batch_size
        ),
        "seed": request.seed,
    }
    # Preserve commit identity for the previously hard-coded parity policy.
    # A non-default retry policy is new semantic input and must invalidate it.
    if request.rank_retry != _DEFAULT_RESIDENT_RANK_RETRY:
        semantic_config["rank_retry"] = request.rank_retry
    if adaptive_memory and request.run_config is not None:
        semantic_config["adaptive_memory"] = {
            "policy": request.run_config.runtime.memory_policy,
            "block_forward_batch_maximum": request.run_config.runtime.block_forward_batch_size,
            "tuning_microbatch_maximum": request.run_config.block_tuning.microbatch_size,
        }
    return "sha256:" + hashlib.sha256(canonical_json(semantic_config).encode()).hexdigest()


def _manifest_tensor_identity(value: torch.Tensor | tuple[tuple[int, ...], ...] | None) -> object:
    if value is None:
        return None
    tensor = value.detach().to(device="cpu").contiguous() if isinstance(value, torch.Tensor) else torch.tensor(value)
    payload = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "content_hash": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def _resident_manifest_config(request: ResidentQuantizationRequest, component: str) -> dict[str, object]:
    payload = cast(dict[str, object], to_dict(request))
    payload.pop("run_config")
    payload.pop("launcher_path")
    payload.pop("memory_plan")
    payload.pop("memory_plan_reference")
    payload["token_ids"] = _manifest_tensor_identity(request.token_ids)
    payload["quality_token_ids"] = _manifest_tensor_identity(request.quality_token_ids)
    payload["component"] = component
    if request.run_config is not None:
        payload["canonical_run_config"] = to_dict(request.run_config)
    if request.memory_plan is not None and request.memory_plan.mode == "adaptive":
        # These values are physical choices selected by the durable memory plan,
        # not changes to the user's canonical request. Keeping their concrete
        # values here made a throughput or OOM revision look like a reconfigured
        # run on restart even though commit identity correctly remained stable.
        payload.update(
            {
                "executor": "adaptive",
                "activation_gpu_cache": "adaptive",
                "block_forward_batch_size": "adaptive",
                "tuning_microbatch_size": "adaptive",
                "post_block_refit_microbatch_size": "adaptive",
                "restore_completed_blocks": "adaptive",
                "evaluate_inline_quality": "adaptive",
            }
        )
        if request.calibration_method == "forward_only":
            payload["calibration_batch_size"] = "adaptive"
    return payload


def _resident_manifest(request: ResidentQuantizationRequest, component: str) -> RunManifest:
    resolved = _resident_manifest_config(request, component)
    resolved_hash = "sha256:" + hashlib.sha256(canonical_json(resolved).encode()).hexdigest()
    launcher_path = request.launcher_path or Path(__file__)
    experiment_number = None if request.run_config is None else request.run_config.intent.experiment_number
    return initial_manifest_from_resolved(
        resolved_hash,
        resolved,
        launcher_provenance(launcher_path, experiment_number),
        capture_environment(),
    )


def _start_resident_manifest(manifest: RunManifest, current: RunManifest) -> RunManifest:
    if manifest.status in {RunStatus.CREATED, RunStatus.INTERRUPTED, RunStatus.FAILED}:
        started = transition(manifest, RunStatus.RUNNING)
    elif manifest.status is RunStatus.RUNNING:
        started = manifest
    else:
        raise ValueError(f"cannot resume resident run in terminal state {manifest.status.value}")
    return replace(
        started,
        config_hash=current.config_hash,
        resolved_config=current.resolved_config,
        launcher=current.launcher,
        environment=current.environment,
        failure=None,
    )


def _write_resident_manifest(output: Path, manifest: RunManifest) -> None:
    RunDirectory(output.parent, output.name).write_manifest(manifest)


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


def _resolve_shared_input_groups(
    adapter: ModelAdapter,
    inventory: ModelInventory,
    configured: tuple[SharedInputGroupConfig, ...],
) -> tuple[SharedInputGroupCandidate, ...]:
    if not configured:
        return ()
    resolved: list[SharedInputGroupCandidate] = []
    for block in inventory.blocks:
        candidates = {candidate.name: candidate for candidate in adapter.shared_input_group_candidates(block.block)}
        for group in configured:
            candidate = candidates.get(group.name)
            if candidate is None:
                raise ValueError(
                    f"adapter does not declare shared-input group {group.name!r} for block {block.block.index}"
                )
            actual_members = tuple(member.path for member in candidate.members)
            if actual_members != group.members:
                raise ValueError(
                    f"configured members differ from adapter topology for {group.name!r}: "
                    f"{group.members!r} != {actual_members!r}"
                )
            configured_multipliers = {entry.member: entry.multiplier for entry in group.member_multipliers}
            resolved.append(
                replace(
                    candidate,
                    objective_multipliers=tuple(configured_multipliers.get(member, 1.0) for member in actual_members),
                )
            )
    return tuple(resolved)


def _planning_unit_ids(
    inventory: ModelInventory,
    groups: tuple[SharedInputGroupCandidate, ...],
) -> tuple[str, ...]:
    by_member = {member: group for group in groups for member in group.members}
    result: list[str] = []
    for block in inventory.blocks:
        emitted: set[str] = set()
        for layer in block.quantizable_layers:
            group = by_member.get(layer.layer)
            name = layer.layer.path if group is None else group.name
            if name not in emitted:
                result.append(f"{block.block.index}:{name}")
                emitted.add(name)
    return tuple(result)


def _load_requested_kl_sensitivities(
    request: ResidentQuantizationRequest,
    inventory: ModelInventory,
    groups: tuple[SharedInputGroupCandidate, ...],
) -> tuple[tuple[str, float], ...]:
    configured = request.kl_profile_artifact
    if configured is None:
        raise ValueError("KL-calibrated planning is missing its profile artifact")
    path = Path(configured)
    if not path.is_absolute():
        repository = request.launcher_path.parent.parent if request.launcher_path is not None else Path.cwd()
        path = repository / path
    if path.is_dir():
        path = path / "kl-budget-profile.json"
    profile = load_kl_budget_profile(path)
    validate_kl_budget_profile(
        profile,
        model_source=request.source,
        model_revision=request.revision,
        expected_profile_key=request.kl_profile_key,
    )
    unit_ids = _planning_unit_ids(inventory, groups)
    objective = request.reconstruction_rank_planning.kl_objective
    if objective is KlAllocationObjective.INTERACTION_NORMALIZED_UNIT_KL:
        return interaction_corrected_unit_kl_anchors(profile, unit_ids)
    if objective is KlAllocationObjective.MEASURED_UNIT_KL:
        return measured_unit_kl_anchors(profile, unit_ids)
    return kl_calibrated_sensitivities(
        profile,
        unit_ids,
        use_exact_unit_arms=(
            request.kl_sensitivity_granularity
            is not KlSensitivityGranularity.TYPE_BLOCK
        ),
    )


@dataclass(frozen=True, slots=True)
class _RankProbePoint:
    rank: int
    raw_squared_error: float
    normalized_squared_error: float
    weighted_normalized_squared_error: float


@dataclass(frozen=True, slots=True)
class _RankProbeEvidence:
    schema_version: int
    probe_plan: ArtifactRef
    unit_id: str
    block: int
    name: str
    members: tuple[LayerId, ...]
    source_weight_hashes: tuple[str, ...]
    baseline_rank: int
    response_curve: RankResponseCurveConfig
    response_points: tuple[_RankProbePoint, ...]
    raw_squared_error: float
    normalized_squared_error: float
    weighted_normalized_squared_error: float
    relative_frobenius_error: float
    member_squared_errors: tuple[float, ...]
    member_normalized_squared_errors: tuple[float, ...]
    member_sensitivity_energies: tuple[float, ...]
    member_weight_norms_squared: tuple[float, ...]
    logical_seed: int
    wall_seconds: float
    peak_workspace_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("unsupported reconstruction rank-probe evidence schema")
        if not self.response_points or tuple(point.rank for point in self.response_points) != tuple(
            sorted({point.rank for point in self.response_points})
        ):
            raise ValueError("rank-probe response points must be non-empty, unique, and ordered")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.raw_squared_error,
                self.normalized_squared_error,
                self.weighted_normalized_squared_error,
                self.relative_frobenius_error,
            )
        ):
            raise ValueError("reconstruction rank-probe errors must be finite and positive")


def _rank_probe_units(
    plan: QuantizationPlan,
) -> tuple[tuple[str, LayerPlan | SharedInputGroupPlan], ...]:
    units: list[tuple[str, LayerPlan | SharedInputGroupPlan]] = []
    for block in plan.blocks:
        layers = {layer.layer.path: layer for layer in block.layers}
        groups = {group.name: group for group in block.shared_input_groups}
        schedule = block.unit_order or (*groups, *layers)
        for name in schedule:
            unit = groups.get(name) or layers.get(name)
            if unit is None:
                raise ValueError(f"rank probe schedule refers to an absent unit: {block.block.index}:{name}")
            units.append((f"{block.block.index}:{name}", unit))
    return tuple(units)


def _matched_rank_response_curve(
    name: str,
    curves: tuple[RankResponseCurveConfig, ...],
) -> RankResponseCurveConfig:
    matches = tuple(curve for curve in curves if fnmatchcase(name, curve.unit_pattern))
    if len(matches) != 1:
        raise ValueError(f"quantization unit {name!r} must match exactly one reconstruction response curve")
    return matches[0]


def _aligned_probe_bounds(
    baseline_rank: int,
    physical_cap: int,
    *,
    multiple: int,
    floor_fraction: float,
    ceiling_fraction: float,
) -> tuple[int, int]:
    floor_rank = math.ceil(baseline_rank * floor_fraction / multiple) * multiple
    ceiling_rank = math.floor(baseline_rank * ceiling_fraction / multiple) * multiple
    ceiling_rank = min(ceiling_rank, math.floor(physical_cap / multiple) * multiple)
    if not multiple <= floor_rank <= baseline_rank <= ceiling_rank:
        raise ValueError("measured rank-response bounds do not contain the aligned baseline rank")
    return floor_rank, ceiling_rank


def _measured_response_curve(
    name: str,
    baseline_rank: int,
    points: tuple[_RankProbePoint, ...],
) -> RankResponseCurveConfig:
    by_rank = {point.rank: point for point in points}
    baseline = by_rank.get(baseline_rank)
    if baseline is None:
        raise ValueError("measured rank response has no baseline point")
    lower_rank = min(by_rank)
    upper_rank = max(by_rank)

    def slope(left_rank: int, right_rank: int) -> float:
        if left_rank == right_rank:
            return 0.0
        left = by_rank[left_rank].weighted_normalized_squared_error
        right = by_rank[right_rank].weighted_normalized_squared_error
        # ADMM noise can make a higher-rank probe slightly worse.  A monotone
        # lower envelope treats that interval as having no demonstrated gain;
        # it never fabricates a positive response from a regression.
        return max(0.0, math.log(left / right) / (2.0 * (right_rank - left_rank)))

    lower_beta = slope(lower_rank, baseline_rank)
    upper_beta = slope(baseline_rank, upper_rank)
    if lower_rank == baseline_rank:
        lower_beta = upper_beta
    if upper_rank == baseline_rank:
        upper_beta = lower_beta
    segments: list[RankResponseSegmentConfig] = []
    if lower_rank < baseline_rank:
        segments.append(RankResponseSegmentConfig(1.0, lower_beta))
    if upper_rank > baseline_rank:
        segments.append(RankResponseSegmentConfig(upper_rank / baseline_rank, upper_beta))
    if not segments:
        segments.append(RankResponseSegmentConfig(1.0, 0.0))
    return RankResponseCurveConfig(
        name,
        lower_rank / baseline_rank,
        upper_rank / baseline_rank,
        tuple(segments),
    )


def _persist_rank_probe_plan(
    request: ResidentQuantizationRequest,
    baseline_plan: QuantizationPlan,
    artifacts: LocalArtifactStore,
) -> ArtifactRef:
    reconstruction = request.reconstruction_rank_planning
    payload = {
        "schema_version": 2,
        "resident_config_hash": _resident_config_hash(request),
        "objective_mode": reconstruction.objective_mode,
        "probe_admm": to_dict(reconstruction.probe_admm),
        "response_source": reconstruction.response_source.value,
        "response_curves": to_dict(reconstruction.response_curves),
        "response_profile_provenance": reconstruction.response_profile_provenance,
        "baseline_plan": to_dict(baseline_plan),
    }
    with artifacts.begin_write(ArtifactTypes.RANK_PROBE_PLAN) as writer:
        (writer.path / "rank-probe-plan.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    return ArtifactRef(ArtifactTypes.RANK_PROBE_PLAN, descriptor.artifact_id, 1)


def _rank_probe_journal_path(output: Path) -> Path:
    return output / "state" / "rank-probe-journal.jsonl"


def _load_rank_probe_results(
    request: ResidentQuantizationRequest,
    probe_plan: ArtifactRef,
    artifacts: LocalArtifactStore,
) -> dict[str, tuple[ArtifactRef, _RankProbeEvidence]]:
    path = _rank_probe_journal_path(request.output)
    if not path.exists():
        return {}
    results: dict[str, tuple[ArtifactRef, _RankProbeEvidence]] = {}
    for sequence, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
            if record["sequence"] != sequence:
                raise ValueError("sequence is not contiguous")
            if record["probe_plan_artifact"] != probe_plan.artifact_id:
                continue
            reference = ArtifactRef(
                ArtifactTypes.RANK_PROBE_RESULT,
                str(record["artifact_id"]),
                1,
            )
            descriptor = artifacts.validate(reference.artifact_id)
            if descriptor.artifact_type != ArtifactTypes.RANK_PROBE_RESULT:
                raise ValueError("artifact type differs")
            payload = json.loads(
                (artifacts.path_for(reference.artifact_id) / "rank-probe-result.json").read_text(encoding="utf-8")
            )
            evidence = from_dict(_RankProbeEvidence, payload, path="rank_probe_result")
            if evidence.probe_plan != probe_plan or evidence.unit_id != record["unit_id"]:
                raise ValueError("result identity differs")
            prior = results.get(evidence.unit_id)
            if prior is not None and prior[0] != reference:
                raise ValueError("duplicate rank probe results differ")
            results[evidence.unit_id] = (reference, evidence)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"rank probe journal is invalid at sequence {sequence}") from exc
    return results


def _commit_rank_probe_result(
    request: ResidentQuantizationRequest,
    evidence: _RankProbeEvidence,
    artifacts: LocalArtifactStore,
) -> ArtifactRef:
    with artifacts.begin_write(ArtifactTypes.RANK_PROBE_RESULT) as writer:
        (writer.path / "rank-probe-result.json").write_text(
            json.dumps(to_dict(evidence), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    reference = ArtifactRef(ArtifactTypes.RANK_PROBE_RESULT, descriptor.artifact_id, 1)
    journal_path = _rank_probe_journal_path(request.output)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    sequence = 1
    if journal_path.exists():
        sequence += len(journal_path.read_text(encoding="utf-8").splitlines())
    with journal_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "sequence": sequence,
                    "probe_plan_artifact": evidence.probe_plan.artifact_id,
                    "unit_id": evidence.unit_id,
                    "artifact_id": reference.artifact_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
    return reference


def _run_reconstruction_rank_probes(
    request: ResidentQuantizationRequest,
    baseline_plan: QuantizationPlan,
    probe_plan: ArtifactRef,
    source: SafetensorsModelSource,
    calibration: PersistedCalibration,
    tensors: LocalTensorStore,
    artifacts: LocalArtifactStore,
    events: EventSink,
    recorder: PhaseRecorder,
    kl_sensitivities: tuple[tuple[str, float], ...] = (),
) -> tuple[ArtifactRef, tuple[ReconstructionRankDecision, ...]]:
    reconstruction = request.reconstruction_rank_planning
    probe_admm = reconstruction.probe_admm
    if probe_admm is None:
        raise ValueError("reconstruction-aware planning requires resolved probe ADMM settings")
    calibration_by_layer = {item.layer: item for item in calibration.stats.layers}
    units = _rank_probe_units(baseline_plan)
    persisted = _load_rank_probe_results(request, probe_plan, artifacts)
    committed_this_execution = 0
    for unit_id, unit in units:
        member_inventories = (
            unit.members
            if isinstance(unit, SharedInputGroupPlan)
            else tuple(layer for block in baseline_plan.blocks for layer in block.layers if layer.layer == unit.layer)
        )
        if unit_id in persisted:
            evidence = persisted[unit_id][1]
            expected_hashes = tuple(
                member.weight.content_hash if isinstance(member, LayerInventory) else member.source_weight.content_hash
                for member in member_inventories
            )
            expected_members = tuple(member.layer for member in member_inventories)
            if (
                evidence.members != expected_members
                or evidence.source_weight_hashes != expected_hashes
                or evidence.baseline_rank != unit.rank
            ):
                raise ValueError(f"persisted rank probe differs from current unit: {unit_id}")
            cast(Any, events).emit(
                "resident-quantization",
                "info",
                "rank_probe.unit_reused",
                unit_id=unit_id,
                artifact_id=persisted[unit_id][0].artifact_id,
            )
            continue
        unit_name = unit.name if isinstance(unit, SharedInputGroupPlan) else unit.layer.path
        configured_curve = (
            _matched_rank_response_curve(unit_name, reconstruction.response_curves)
            if reconstruction.response_source is RankResponseSource.CONFIGURED
            else None
        )
        member_layers = tuple(member.layer for member in member_inventories)
        seed = logical_seed(
            request.seed,
            "rank-reconstruction-probe",
            unit.block.index if isinstance(unit, SharedInputGroupPlan) else unit.layer.block.index,
            unit.name if isinstance(unit, SharedInputGroupPlan) else unit.layer.path,
            0,
        )
        cast(Any, events).emit(
            "resident-quantization",
            "info",
            "rank_probe.unit_started",
            unit_id=unit_id,
            members=tuple(f"{layer.block.index}:{layer.path}" for layer in member_layers),
            rank=unit.rank,
        )
        started = time.perf_counter()
        if request.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(request.device)
        with ExitStack() as stack:
            weights = tuple(
                stack.enter_context(
                    source.read_tensor(
                        member.weight if isinstance(member, LayerInventory) else member.source_weight,
                        device=request.device,
                    )
                )
                for member in member_inventories
            )
            input_importances = tuple(
                stack.enter_context(tensors.read(calibration_by_layer[layer].input_importance, request.device))
                for layer in member_layers
            )
            output_importances = tuple(
                stack.enter_context(tensors.read(calibration_by_layer[layer].output_importance, request.device))
                for layer in member_layers
            )
            stacked = weights[0] if len(weights) == 1 else torch.cat(weights, dim=0)
            if reconstruction.objective_mode == "calibration_weighted":
                probe_input_importance = input_importances[0].float()
                if any(
                    not torch.allclose(probe_input_importance, value.float(), rtol=1e-5, atol=1e-7)
                    for value in input_importances[1:]
                ):
                    raise ValueError(f"shared-input unit has inconsistent input importance: {unit_id}")
                probe_output_importance = torch.cat(tuple(value.float() for value in output_importances))
            else:
                probe_input_importance = torch.ones(stacked.shape[1], device=stacked.device, dtype=torch.float32)
                probe_output_importance = torch.ones(stacked.shape[0], device=stacked.device, dtype=torch.float32)

            def factorize_probe(
                rank: int,
                *,
                probe_seed: int = seed,
                probe_weight: torch.Tensor = stacked,
                probe_input: torch.Tensor = probe_input_importance,
                probe_output: torch.Tensor = probe_output_importance,
            ) -> Any:
                generator = torch.Generator(device=request.device).manual_seed(probe_seed)
                return factorize_admm(
                    probe_weight,
                    probe_input,
                    probe_output,
                    rank,
                    generator,
                    outer_iterations=probe_admm.outer_iterations,
                    inner_iterations=probe_admm.inner_iterations,
                    regularization=probe_admm.regularization,
                    penalty_schedule=probe_admm.penalty_schedule,
                    convergence_check_interval=probe_admm.convergence_check_interval,
                    early_stop_tolerance=probe_admm.early_stop_tolerance,
                    transpose_wide=probe_admm.transpose_wide,
                )

            with recorder.phase("rank_probe", unit=unit_id):
                factorized = factorize_probe(unit.rank)
            difference = stacked.float() - factorized.reconstruction.float()
            raw_error = float(difference.square().sum())
            target_norm = float(stacked.float().square().sum())
            member_errors: list[float] = []
            member_normalized: list[float] = []
            member_energies: list[float] = []
            member_norms: list[float] = []
            weighted_error = 0.0
            weighted_target_norm = 0.0
            row = 0
            for weight, input_importance, output_importance in zip(
                weights,
                input_importances,
                output_importances,
                strict=True,
            ):
                rows = weight.shape[0]
                member_difference = difference[row : row + rows]
                member_error = float(member_difference.square().sum())
                member_norm = float(weight.float().square().sum())
                member_errors.append(member_error)
                member_normalized.append(member_error / max(member_norm, 1e-30))
                member_norms.append(member_norm)
                member_weights = (
                    output_importance.float()[:, None].clamp_min(1e-12)
                    * input_importance.float()[None, :].clamp_min(1e-12)
                )
                weighted_error += float((member_difference.square() * member_weights).sum())
                weighted_target_norm += float((weight.float().square() * member_weights).sum())
                member_energies.append(
                    float(
                        (
                            weight.float().square()
                            * member_weights
                        )
                        .mean()
                        .sqrt()
                    )
                )
                row += rows
            response_points = [
                _RankProbePoint(
                    unit.rank,
                    raw_error,
                    raw_error / max(target_norm, 1e-30),
                    weighted_error / max(weighted_target_norm, 1e-30),
                )
            ]
            if reconstruction.response_source is RankResponseSource.MEASURED:
                floor_rank, ceiling_rank = _aligned_probe_bounds(
                    unit.rank,
                    min(stacked.shape),
                    multiple=request.rank_multiple,
                    floor_fraction=request.rank_floor_fraction,
                    ceiling_fraction=request.rank_ceiling_fraction,
                )
                for response_rank in (floor_rank, ceiling_rank):
                    if response_rank == unit.rank:
                        continue
                    with recorder.phase("rank_probe_response", unit=unit_id, rank=response_rank):
                        response_factorized = factorize_probe(response_rank)
                    response_difference = stacked.float() - response_factorized.reconstruction.float()
                    response_raw = float(response_difference.square().sum())
                    response_weighted = 0.0
                    response_row = 0
                    for weight, input_importance, output_importance in zip(
                        weights,
                        input_importances,
                        output_importances,
                        strict=True,
                    ):
                        rows = weight.shape[0]
                        member_difference = response_difference[response_row : response_row + rows]
                        member_weights = (
                            output_importance.float()[:, None].clamp_min(1e-12)
                            * input_importance.float()[None, :].clamp_min(1e-12)
                        )
                        response_weighted += float((member_difference.square() * member_weights).sum())
                        response_row += rows
                    response_points.append(
                        _RankProbePoint(
                            response_rank,
                            response_raw,
                            response_raw / max(target_norm, 1e-30),
                            response_weighted / max(weighted_target_norm, 1e-30),
                        )
                    )
                    del response_factorized
            ordered_response_points = tuple(sorted(response_points, key=lambda point: point.rank))
            curve = configured_curve or _measured_response_curve(
                unit_name,
                unit.rank,
                ordered_response_points,
            )
            peak = int(torch.cuda.max_memory_allocated(request.device)) if request.device.startswith("cuda") else 0
            evidence = _RankProbeEvidence(
                3,
                probe_plan,
                unit_id,
                member_layers[0].block.index,
                unit.name if isinstance(unit, SharedInputGroupPlan) else unit.layer.path,
                member_layers,
                tuple(
                    member.weight.content_hash
                    if isinstance(member, LayerInventory)
                    else member.source_weight.content_hash
                    for member in member_inventories
                ),
                unit.rank,
                curve,
                ordered_response_points,
                raw_error,
                raw_error / max(target_norm, 1e-30),
                weighted_error / max(weighted_target_norm, 1e-30),
                math.sqrt(raw_error / max(target_norm, 1e-30)),
                tuple(member_errors),
                tuple(member_normalized),
                tuple(member_energies),
                tuple(member_norms),
                seed,
                time.perf_counter() - started,
                peak,
            )
        reference = _commit_rank_probe_result(request, evidence, artifacts)
        persisted[unit_id] = (reference, evidence)
        committed_this_execution += 1
        cast(Any, events).emit(
            "resident-quantization",
            "info",
            "rank_probe.unit_completed",
            unit_id=unit_id,
            artifact_id=reference.artifact_id,
            rank=unit.rank,
            relative_frobenius_error=evidence.relative_frobenius_error,
            wall_seconds=evidence.wall_seconds,
            peak_workspace_bytes=evidence.peak_workspace_bytes,
        )
        del factorized
        if request.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if (
            request.interrupt_after_rank_probe_commits is not None
            and committed_this_execution >= request.interrupt_after_rank_probe_commits
        ):
            raise InterruptedError(
                f"injected interruption after {committed_this_execution} reconstruction rank probe commits"
            )

    if set(persisted) != {unit_id for unit_id, _unit in units}:
        raise ValueError("rank probe profile does not cover every quantization unit")
    member_entries: list[tuple[LayerId, float]] = []
    for unit_id, _unit in units:
        evidence = persisted[unit_id][1]
        member_entries.extend(zip(evidence.members, evidence.member_sensitivity_energies, strict=True))
    block_medians = {
        block: max(statistics.median(energy for layer, energy in member_entries if layer.block.index == block), 1e-12)
        for block in {layer.block.index for layer, _energy in member_entries}
    }
    relative = {layer: energy / block_medians[layer.block.index] for layer, energy in member_entries}
    type_medians = {
        path: max(statistics.median(value for layer, value in relative.items() if layer.path == path), 1e-12)
        for path in {layer.path for layer in relative}
    }
    importance = reconstruction.importance
    member_layers = tuple(relative)
    importance_multiplier, architecture_protected, edge_blocks = _reconstruction_importance_policy(
        member_layers, importance
    )
    member_sensitivity = {
        layer: max(value / type_medians[layer.path], 1e-12) * importance_multiplier[layer]
        for layer, value in relative.items()
    }
    kl_sensitivity = dict(kl_sensitivities)
    measured_kl_objective = (
        bool(kl_sensitivity)
        and reconstruction.kl_objective in MEASURED_UNIT_KL_OBJECTIVES
    )
    if kl_sensitivity and set(kl_sensitivity) != {unit_id for unit_id, _unit in units}:
        raise ValueError("KL sensitivity profile does not exactly cover reconstruction units")
    ordered_scores = sorted(kl_sensitivity.values() if kl_sensitivity else member_sensitivity.values())
    protected_index = min(
        len(ordered_scores) - 1,
        math.floor(reconstruction.protected_sensitivity_quantile * len(ordered_scores)),
    )
    protected_threshold = (
        ordered_scores[protected_index] if reconstruction.protect_sensitive_units else math.inf
    )
    allocation_units: list[ReconstructionAllocationUnit] = []
    uncharged_actual_bits = 0
    protected_by_unit: dict[str, tuple[LayerId, ...]] = {}
    sensitivity_by_unit: dict[str, float] = {}
    for unit_id, unit in units:
        evidence = persisted[unit_id][1]
        total_norm = sum(evidence.member_weight_norms_squared)
        proxy_unit_sensitivity = math.exp(
            sum(
                norm * math.log(member_sensitivity[layer])
                for layer, norm in zip(
                    evidence.members,
                    evidence.member_weight_norms_squared,
                    strict=True,
                )
            )
            / max(total_norm, 1e-30)
        )
        unit_sensitivity = kl_sensitivity.get(unit_id, proxy_unit_sensitivity)
        protected_members = tuple(
            layer
            for layer in evidence.members
            if (
                (
                    unit_sensitivity >= protected_threshold
                    if kl_sensitivity
                    else member_sensitivity[layer] >= protected_threshold
                )
                or layer in architecture_protected
            )
        )
        protected_by_unit[unit_id] = protected_members
        sensitivity_by_unit[unit_id] = unit_sensitivity
        out_features = unit.out_features if isinstance(unit, SharedInputGroupPlan) else unit.source_weight.spec.shape[0]
        in_features = unit.in_features if isinstance(unit, SharedInputGroupPlan) else unit.source_weight.spec.shape[1]
        fixed_bits = 0
        if unit.outliers.charge_to_budget:
            fixed_bits = unit.estimated_cost.outlier_value_bits + unit.estimated_cost.outlier_index_bits
        if request.bias_correction.charge_to_bit_budget:
            fixed_bits += unit.estimated_cost.bias_bits
        if request.low_rank_patch.charge_to_bit_budget:
            fixed_bits += unit.estimated_cost.patch_bits
        actual_fixed_bits = (
            unit.estimated_cost.outlier_value_bits
            + unit.estimated_cost.outlier_index_bits
            + unit.estimated_cost.bias_bits
            + unit.estimated_cost.patch_bits
        )
        uncharged_actual_bits += actual_fixed_bits - fixed_bits
        allocation_units.append(
            ReconstructionAllocationUnit(
                unit_id,
                out_features,
                in_features,
                evidence.baseline_rank,
                (
                    1.0
                    if measured_kl_objective
                    else evidence.weighted_normalized_squared_error
                    if (
                        kl_sensitivity
                        or reconstruction.response_source is RankResponseSource.MEASURED
                    )
                    else evidence.raw_squared_error
                ),
                unit_sensitivity,
                bool(protected_members),
                evidence.response_curve.calibrated_rank_floor_fraction,
                evidence.response_curve.calibrated_rank_ceiling_fraction,
                tuple(
                    RankResponseSegment(segment.maximum_rank_fraction, segment.beta_per_rank)
                    for segment in evidence.response_curve.segments
                ),
                fixed_bits,
            )
        )
    original_elements = sum(
        math.prod(layer.source_weight.spec.shape) for block in baseline_plan.blocks for layer in block.layers
    ) + sum(
        member.in_features * member.out_features
        for block in baseline_plan.blocks
        for group in block.shared_input_groups
        for member in group.members
    )
    allocated = allocate_reconstruction_rank_budget(
        tuple(allocation_units),
        math.floor(original_elements * request.target_bpw),
        multiple=request.rank_multiple,
        floor_fraction=request.rank_floor_fraction,
        ceiling_fraction=request.rank_ceiling_fraction,
        sensitivity_strength=reconstruction.sensitivity_strength,
        protected_rank_floor_fraction=reconstruction.protected_rank_floor_fraction,
        target_protected_error_reduction_fraction=(reconstruction.target_protected_error_reduction_fraction),
    )
    rank_trust_region: dict[str, Any] | None = None
    if reconstruction.rank_trust_reference_run is not None:
        reference_path = Path(reconstruction.rank_trust_reference_run)
        if not reference_path.is_absolute():
            repository = (
                request.launcher_path.parent.parent
                if request.launcher_path is not None
                else Path.cwd()
            )
            reference_path = repository / reference_path
        planning_reference = load_frozen_run_planning_reference(
            reference_path.resolve(),
            len(baseline_plan.blocks),
            fresh_validation=True,
        )
        reference_ranks = tuple(
            (entry.unit_id, entry.rank) for entry in planning_reference.ranks
        )
        nominal_target_bits = math.floor(original_elements * request.target_bpw)
        same_budget_target_bits = planning_reference.total_bits - uncharged_actual_bits
        trust_target_bits = min(nominal_target_bits, same_budget_target_bits)
        if trust_target_bits <= 0:
            raise ValueError("rank trust reference cannot fund candidate fixed bit costs")
        unconstrained_ranks = {
            decision.unit_id: decision.planned_rank for decision in allocated.decisions
        }
        allocated = apply_reconstruction_rank_trust_region(
            tuple(allocation_units),
            allocated,
            reference_ranks,
            trust_target_bits,
            multiple=request.rank_multiple,
            floor_fraction=request.rank_floor_fraction,
            ceiling_fraction=request.rank_ceiling_fraction,
            sensitivity_strength=reconstruction.sensitivity_strength,
            protected_rank_floor_fraction=reconstruction.protected_rank_floor_fraction,
            target_protected_error_reduction_fraction=(
                reconstruction.target_protected_error_reduction_fraction
            ),
            step_fraction=reconstruction.rank_trust_fraction,
        )
        projected_ranks = {
            decision.unit_id: decision.planned_rank for decision in allocated.decisions
        }
        rank_trust_region = {
            "reference_run": str(reference_path.resolve()),
            "step_fraction": reconstruction.rank_trust_fraction,
            "reference_total_bits": planning_reference.total_bits,
            "candidate_uncharged_actual_bits": uncharged_actual_bits,
            "nominal_target_bits": nominal_target_bits,
            "trust_target_bits": trust_target_bits,
            "reference_ranks": dict(reference_ranks),
            "unconstrained_ranks": unconstrained_ranks,
            "projected_ranks": projected_ranks,
        }
    allocation_by_unit = {decision.unit_id: decision for decision in allocated.decisions}
    decisions = tuple(
        ReconstructionRankDecision(
            unit_id,
            persisted[unit_id][1].members,
            allocation_by_unit[unit_id].baseline_rank,
            allocation_by_unit[unit_id].planned_rank,
            allocation_by_unit[unit_id].baseline_squared_error,
            allocation_by_unit[unit_id].predicted_squared_error,
            sensitivity_by_unit[unit_id],
            protected_by_unit[unit_id],
        )
        for unit_id, _unit in units
    )
    profile_payload = {
        "schema_version": 2,
        "producer": {"name": "reconstruction-rank-planner", "version": "2"},
        "probe_plan": to_dict(probe_plan),
        "unit_results": [to_dict(persisted[unit_id][0]) for unit_id, _unit in units],
        "decisions": to_dict(decisions),
        "allocation": {
            "spent_bits": allocated.spent_bits,
            "remaining_bits": allocated.remaining_bits,
            "protected_baseline_objective": allocated.protected_baseline_objective,
            "protected_planned_objective": allocated.protected_planned_objective,
        },
        "rank_trust_region": rank_trust_region,
        "importance": {
            "sensitivity_source": "kl_budget_profile" if kl_sensitivity else "activation_weight_proxy",
            "allocation_error_measure": (
                "same_run_relative_weighted_error"
                if measured_kl_objective
                else "weighted_normalized_squared_error"
                if (
                    kl_sensitivity
                    or reconstruction.response_source is RankResponseSource.MEASURED
                )
                else "raw_squared_error"
            ),
            "allocation_objective": (
                "unit_kl_proportional_score"
                if measured_kl_objective
                else "tempered_reconstruction_proxy"
            ),
            "policy": to_dict(importance),
            "edge_blocks": sorted(edge_blocks),
            "member_multipliers": {
                f"{layer.block.index}:{layer.path}": importance_multiplier[layer] for layer in member_layers
            },
        },
    }
    with artifacts.begin_write(ArtifactTypes.RECONSTRUCTION_RANK_PROFILE) as writer:
        (writer.path / "reconstruction-rank-profile.json").write_text(
            json.dumps(profile_payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    profile = ArtifactRef(ArtifactTypes.RECONSTRUCTION_RANK_PROFILE, descriptor.artifact_id, 1)
    cast(Any, events).emit(
        "resident-quantization",
        "info",
        "rank_probe.profile_committed",
        artifact_id=profile.artifact_id,
        unit_count=len(units),
        spent_bits=allocated.spent_bits,
        remaining_bits=allocated.remaining_bits,
        protected_baseline_objective=allocated.protected_baseline_objective,
        protected_planned_objective=allocated.protected_planned_objective,
    )
    return profile, decisions


def _active_preprocessing_path(output: Path) -> Path:
    return output / "state" / "preprocessing.json"


def _read_active_preprocessing_state(
    request: ResidentQuantizationRequest,
) -> _ActivePreprocessingState | None:
    path = _active_preprocessing_path(request.output)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = from_dict(_ActivePreprocessingState, payload, path="active_preprocessing")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"active preprocessing state is invalid: {path}") from exc
    if state.schema_version != 1:
        raise ValueError(f"unsupported active preprocessing schema: {state.schema_version}")
    if state.resident_config_hash != _resident_config_hash(request):
        return None
    return state


def _recover_preprocessing_references_from_journal(
    request: ResidentQuantizationRequest,
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef] | None:
    """Recover the preprocessing identity of runs created before the active pointer.

    The journal supplies the authoritative semantic config and plan identity.  The
    event stream only supplies the calibration/objective links for that exact plan;
    the normal artifact loader subsequently validates every descriptor and link.
    """

    journal_path = request.output / "state" / "journal.jsonl"
    events_path = request.output / "events.jsonl"
    if not journal_path.exists() or not events_path.exists():
        return None
    expected_config_hash = _resident_config_hash(request)
    plan_artifact: str | None = None
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        try:
            identity = json.loads(line)["identity"]
            if identity["config_hash"] == expected_config_hash:
                plan_artifact = str(identity["plan_hash"])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            break
    if plan_artifact is None:
        return None
    recovered: tuple[ArtifactRef, ArtifactRef, ArtifactRef] | None = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
            fields = event["fields"]
            if event["name"] != "preprocessing.selected" or fields["plan_artifact"] != plan_artifact:
                continue
            recovered = (
                ArtifactRef("calibration-stats", str(fields["calibration_artifact"]), 1),
                ArtifactRef("objective-specs", str(fields["objectives_artifact"]), 1),
                ArtifactRef(ArtifactTypes.QUANTIZATION_PLAN, plan_artifact, 1),
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return recovered


def _resolve_preprocessing_references(
    request: ResidentQuantizationRequest,
) -> tuple[tuple[ArtifactRef, ArtifactRef, ArtifactRef] | None, str]:
    explicit = (
        request.precomputed_calibration,
        request.precomputed_objectives,
        request.precomputed_plan,
    )
    if any(reference is not None for reference in explicit):
        if any(reference is None for reference in explicit):
            raise ValueError("precomputed calibration, objectives, and plan must be supplied together")
        return cast(tuple[ArtifactRef, ArtifactRef, ArtifactRef], explicit), "explicit"
    active = _read_active_preprocessing_state(request)
    if active is not None:
        return (active.calibration, active.objectives, active.plan), "active_state"
    recovered = _recover_preprocessing_references_from_journal(request)
    if recovered is not None:
        return recovered, "journal_recovery"
    return None, "computed"


def _write_active_preprocessing_state(
    request: ResidentQuantizationRequest,
    calibration: ArtifactRef,
    objectives: ArtifactRef,
    plan: ArtifactRef,
) -> None:
    atomic_write_json(
        _active_preprocessing_path(request.output),
        to_dict(
            _ActivePreprocessingState(
                1,
                _resident_config_hash(request),
                calibration,
                objectives,
                plan,
            )
        ),
    )


def _load_precomputed_preprocessing(
    request: ResidentQuantizationRequest,
    artifacts: LocalArtifactStore,
    inventory: ModelInventory,
    dataset: DatasetIdentity,
    total_tokens: int,
    references: tuple[ArtifactRef, ArtifactRef, ArtifactRef] | None = None,
) -> tuple[PersistedCalibration, PersistedObjectives, PersistedPlan] | None:
    selected_references: tuple[ArtifactRef | None, ArtifactRef | None, ArtifactRef | None] = (
        references
        if references is not None
        else (
            request.precomputed_calibration,
            request.precomputed_objectives,
            request.precomputed_plan,
        )
    )
    if all(reference is None for reference in selected_references):
        return None
    if any(reference is None for reference in selected_references):
        raise ValueError("precomputed calibration, objectives, and plan must be supplied together")
    calibration_ref, objectives_ref, plan_ref = cast(tuple[ArtifactRef, ArtifactRef, ArtifactRef], selected_references)
    for reference in selected_references:
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
            from_dict(ObjectiveSpec, item, path=f"objectives[{index}]") for index, item in enumerate(objective_payload)
        ),
    )
    plan_payload = json.loads((artifacts.path_for(plan_ref.artifact_id) / "plan.json").read_text(encoding="utf-8"))
    persisted_plan = PersistedPlan(plan_ref, from_dict(QuantizationPlan, plan_payload, path="plan"))
    profile_reference = persisted_plan.plan.reconstruction_profile
    if request.allocation_strategy is AllocationStrategy.RECONSTRUCTION_AWARE and profile_reference is None:
        raise ValueError("reconstruction-aware precomputed plan has no rank profile")
    if profile_reference is not None:
        descriptor = artifacts.validate(profile_reference.artifact_id)
        if descriptor.artifact_type != ArtifactTypes.RECONSTRUCTION_RANK_PROFILE:
            raise ValueError("reconstruction rank profile has the wrong artifact type")
        profile_payload = json.loads(
            (artifacts.path_for(profile_reference.artifact_id) / "reconstruction-rank-profile.json").read_text(
                encoding="utf-8"
            )
        )
        probe_plan_payload = profile_payload.get("probe_plan")
        unit_results_payload = profile_payload.get("unit_results")
        if not isinstance(probe_plan_payload, dict) or not isinstance(unit_results_payload, list):
            raise ValueError("reconstruction rank profile is malformed")
        probe_plan_reference = from_dict(ArtifactRef, probe_plan_payload, path="profile.probe_plan")
        probe_descriptor = artifacts.validate(probe_plan_reference.artifact_id)
        if probe_descriptor.artifact_type != ArtifactTypes.RANK_PROBE_PLAN:
            raise ValueError("rank probe plan has the wrong artifact type")
        for index, value in enumerate(unit_results_payload):
            if not isinstance(value, dict):
                raise ValueError("rank probe result reference is malformed")
            result_reference = from_dict(ArtifactRef, value, path=f"profile.unit_results[{index}]")
            result_descriptor = artifacts.validate(result_reference.artifact_id)
            if result_descriptor.artifact_type != ArtifactTypes.RANK_PROBE_RESULT:
                raise ValueError("rank probe result has the wrong artifact type")
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
    proposed = _resident_manifest(request, "resident-quantization")
    with open_run_session(
        request.output,
        manifest=proposed,
        observability=request.observability,
        registry_root=request.registry_root,
        console=True,
        maximum_wddm_shared_bytes=request.maximum_wddm_shared_bytes,
    ) as session:
        manifest = _start_resident_manifest(session.manifest, proposed)
        _write_resident_manifest(request.output, manifest)
        session.events.emit(
            "run",
            "info",
            "run.resumed" if session.resumed else "run.started",
            stored_config_hash=session.manifest.config_hash,
            current_config_hash=proposed.config_hash,
            previous_run_id=session.previous_run_id,
            component="resident-quantization",
            device=request.device,
            calibration_method=request.calibration_method,
            calibration_samples=len(request.token_ids),
            block_forward_batch_size=request.block_forward_batch_size,
            factorized_tuning_epochs=request.factorized_tuning_epochs,
            nonfactorized_tuning_epochs=request.nonfactorized_tuning_epochs,
            post_block_refit_epochs=request.post_block_refit_epochs,
            activation_retention=request.activation_retention,
            memory_plan_artifact=(
                None if request.memory_plan_reference is None else request.memory_plan_reference.artifact_id
            ),
            memory_plan_revision=(None if request.memory_plan is None else request.memory_plan.revision),
            memory_plan_mode=(None if request.memory_plan is None else request.memory_plan.mode),
            planned_peak_gpu_bytes=(None if request.memory_plan is None else request.memory_plan.peak_gpu_bytes),
            planned_peak_host_bytes=(None if request.memory_plan is None else request.memory_plan.peak_host_bytes),
        )
        if request.memory_plan is not None:
            cast(Any, session.events).emit(
                "memory",
                "info",
                "memory.plan_created",
                artifact_id=(
                    None if request.memory_plan_reference is None else request.memory_plan_reference.artifact_id
                ),
                revision=request.memory_plan.revision,
                mode=request.memory_plan.mode,
                profile=request.memory_plan.profile,
                executor=request.memory_plan.executor,
                activation_tier=request.memory_plan.activation_tier,
                activation_gpu_cache=request.memory_plan.activation_gpu_cache,
                peak_gpu_bytes=request.memory_plan.peak_gpu_bytes,
                peak_host_bytes=request.memory_plan.peak_host_bytes,
                peak_pinned_host_bytes=request.memory_plan.peak_pinned_host_bytes,
                peak_temporary_disk_bytes=request.memory_plan.peak_temporary_disk_bytes,
                warnings=list(request.memory_plan.warnings),
            )
            if request.memory_plan.revision > 1:
                plan_revision = load_memory_plan_revision(request.output, request.memory_plan.revision)
                revision_reason = "persisted memory plan revision"
                revision_fields: dict[str, object] = {}
                if plan_revision is not None:
                    revision_reason = plan_revision.reason
                    revision_fields = {
                        "parent_revision": plan_revision.parent_revision,
                        "revised_stage": plan_revision.stage,
                        "action": plan_revision.action,
                        "previous_batch_size": plan_revision.previous_batch_size,
                        "next_batch_size": plan_revision.next_batch_size,
                    }
                cast(Any, session.events).emit(
                    "memory",
                    "warning" if revision_reason == "out_of_memory" else "info",
                    "memory.plan_revised",
                    revision=request.memory_plan.revision,
                    reason=revision_reason,
                    algorithm_changed=(
                        False if plan_revision is None else plan_revision.algorithm_changed
                    ),
                    **revision_fields,
                )
            for stage_plan in request.memory_plan.stages:
                cast(Any, session.events).emit(
                    "memory",
                    "info",
                    "memory.stage_resized" if stage_plan.resized else "memory.stage_admitted",
                    plan_revision=request.memory_plan.revision,
                    stage_name=stage_plan.stage,
                    signature=stage_plan.signature,
                    batch_size=stage_plan.batch_size,
                    prefetch_batches=stage_plan.prefetch_batches,
                    predicted_gpu_bytes=stage_plan.predicted_gpu_bytes,
                    predicted_host_bytes=stage_plan.predicted_host_bytes,
                    predicted_pinned_host_bytes=stage_plan.predicted_pinned_host_bytes,
                    gpu_capacity_bytes=stage_plan.gpu_capacity_bytes,
                    host_capacity_bytes=stage_plan.host_capacity_bytes,
                    uncertainty_bytes=stage_plan.uncertainty_bytes,
                )
        session_started = time.perf_counter()
        with profiled_run(
            request.profiling,
            request.output,
            session.events,
            run_id=session.run_id,
        ) as recorder:
            try:
                with recorder.phase("run"):
                    with recorder.phase("pipeline"):
                        result = _run_resident_quantization_impl(request, session.events, recorder, session.run_id)
            except (KeyboardInterrupt, InterruptedError) as exc:
                session.events.emit(
                    "run",
                    "warning",
                    "run.interrupted",
                    component="resident-quantization",
                    device=request.device,
                    wall_seconds=time.perf_counter() - session_started,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                manifest = transition(manifest, RunStatus.INTERRUPTED)
                _write_resident_manifest(request.output, manifest)
                raise
            except BaseException as exc:
                session.events.emit(
                    "run",
                    "error",
                    "run.failed",
                    component="resident-quantization",
                    device=request.device,
                    wall_seconds=time.perf_counter() - session_started,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    cause_type=(
                        None
                        if exc.__cause__ is None and exc.__context__ is None
                        else type(exc.__cause__ or exc.__context__).__name__
                    ),
                    cause=(
                        None
                        if exc.__cause__ is None and exc.__context__ is None
                        else str(exc.__cause__ or exc.__context__)
                    ),
                )
                manifest = transition(
                    manifest,
                    RunStatus.FAILED,
                    failure={"type": type(exc).__name__, "message": str(exc)},
                )
                _write_resident_manifest(request.output, manifest)
                raise
        if request.defer_run_completion:
            session.events.emit(
                "run",
                "info",
                "run.stage_completed",
                component="resident-quantization",
                artifact_id=result.report.artifact_id,
            )
            manifest = replace(manifest, artifacts=(result.report.artifact_id,))
        else:
            session.events.emit("run", "info", "run.completed", artifact_id=result.report.artifact_id)
            manifest = transition(manifest, RunStatus.COMPLETED, artifacts=(result.report.artifact_id,))
        _write_resident_manifest(request.output, manifest)
        return result


def _validate_resident_request(request: ResidentQuantizationRequest) -> None:
    kl_selected = request.allocation_strategy is AllocationStrategy.KL_CALIBRATED
    if kl_selected != bool(request.kl_profile_artifact) or kl_selected != bool(request.kl_profile_key):
        raise ValueError("KL-calibrated resident planning requires exactly one KL profile artifact and key")
    if (
        not kl_selected
        and request.kl_sensitivity_granularity
        is not KlSensitivityGranularity.EXACT_OR_TYPE_BLOCK
    ):
        raise ValueError("a non-default KL sensitivity granularity requires KL-calibrated planning")
    reconstruction = request.reconstruction_rank_planning
    trust_reference = reconstruction.rank_trust_reference_run
    if not math.isfinite(reconstruction.rank_trust_fraction) or not (
        0 <= reconstruction.rank_trust_fraction <= 1
    ):
        raise ValueError("resident rank trust fraction must be in [0, 1]")
    if (reconstruction.rank_trust_fraction == 1) != (trust_reference is None):
        raise ValueError(
            "resident rank trust reference must be set exactly when its fraction is below one"
        )
    if trust_reference is not None and (not kl_selected or not trust_reference.strip()):
        raise ValueError("resident rank trust reference requires KL-calibrated planning")
    if reconstruction.kl_objective in MEASURED_UNIT_KL_OBJECTIVES:
        if (
            not kl_selected
            or request.kl_sensitivity_granularity is not KlSensitivityGranularity.EXACT
            or reconstruction.response_source is not RankResponseSource.MEASURED
            or reconstruction.objective_mode != "calibration_weighted"
            or reconstruction.sensitivity_strength != 1
            or trust_reference is not None
        ):
            raise ValueError(
                "measured-unit KL requires exact arms, current weighted response probes, "
                "untempered sensitivity, and no imported rank reference"
            )
    if request.executor not in {ExecutorKind.RESIDENT, ExecutorKind.CPU_OFFLOAD}:
        raise ValueError(f"unsupported resident composition executor: {request.executor.value}")
    if request.executor is ExecutorKind.CPU_OFFLOAD:
        if request.restore_completed_blocks:
            raise ValueError("cpu_offload requires restore_completed_blocks=False")
        if request.evaluate_inline_quality:
            raise ValueError("cpu_offload requires inline quality evaluation to be disabled")
    if request.activation_gpu_reserve_bytes < 0:
        raise ValueError("activation GPU cache reserve must not be negative")
    if request.block_forward_batch_size <= 0:
        raise ValueError("resident quantization block forward batch size must be positive")
    if request.factorized_tuning_epoch_cooldown_seconds < 0:
        raise ValueError("factorized tuning epoch cooldown must be non-negative")
    if request.nonfactorized_tuning_epoch_cooldown_seconds < 0:
        raise ValueError("non-factorized tuning epoch cooldown must be non-negative")
    if request.post_block_refit_epoch_cooldown_seconds < 0:
        raise ValueError("post-block refit epoch cooldown must be non-negative")
    if request.initial_cooldown_seconds < 0:
        raise ValueError("resident initial cooldown must be non-negative")
    if (
        request.interrupt_after_factorized_tuning_epoch_commits is not None
        and request.interrupt_after_factorized_tuning_epoch_commits <= 0
    ):
        raise ValueError("factorized tuning epoch interruption count must be positive")
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
    if request.post_block_refit_microbatch_size is not None and request.post_block_refit_microbatch_size <= 0:
        raise ValueError("resident quantization post-block refit microbatch size must be positive")
    if request.tuning_epoch_loss_mode not in ("full_evaluation", "legacy_training"):
        raise ValueError(f"unsupported tuning epoch loss mode: {request.tuning_epoch_loss_mode}")
    if request.tuning_epoch_loss_mode == "legacy_training" and request.restore_best_tuning_state:
        raise ValueError("legacy training loss mode cannot restore best tuning state")
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


@dataclass(frozen=True, slots=True)
class _ResidentEnvironment:
    source: SafetensorsModelSource
    checkpoint: CheckpointInventory
    adapter: TransformersModelAdapter
    inventory: ModelInventory
    tokens: torch.Tensor
    quality_tokens: torch.Tensor
    model: nn.Module
    decoder_layers: nn.ModuleList


@dataclass(frozen=True, slots=True)
class _ResidentResumeState:
    config_hash: str
    identity: CommitIdentity
    journal: ProgressJournal
    discovered_records: tuple[JournalRecord, ...]
    committed_blocks: list[tuple[ArtifactRef, BlockResult]]
    budget: BudgetState
    teacher_inputs: torch.Tensor
    compressed_inputs: torch.Tensor
    completed_block_indexes: set[int]
    layer_container: nn.ModuleList
    partial_layer_records: dict[tuple[int, str | None], JournalRecord]
    partial_group_records: dict[tuple[int, str | None], JournalRecord]


@dataclass(frozen=True, slots=True)
class _ResidentBlockWork:
    started: float
    peak_window: PeakWindow
    index: int
    metadata: dict[str, object]
    tuning_forward: Callable[[nn.Module, torch.Tensor], torch.Tensor]
    working_block: nn.Module
    teacher_outputs: torch.Tensor
    output_importance: torch.Tensor
    loss_recorder: BlockLossRecorder
    deferred_slice: bool


def _setup_resident_environment(
    request: ResidentQuantizationRequest,
    events: EventSink,
    recorder: PhaseRecorder,
) -> _ResidentEnvironment:
    """Load immutable source metadata, inputs, and the mutable model shell."""

    with _logged_operation(
        events,
        "inventory",
        source=request.source,
        revision=request.revision,
        verify_hashes=request.verify_hashes,
    ):
        with recorder.phase("setup"):
            with recorder.phase("inventory"):
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
    model_device = _model_placement_device(request)
    with recorder.phase("setup"):
        with recorder.phase("inputs"):
            tokens = _token_tensor(request.token_ids, model_device)
            quality_tokens = _token_tensor(
                request.token_ids if request.quality_token_ids is None else request.quality_token_ids,
                model_device,
            )
    if request.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(request.device)
    with _logged_operation(
        events,
        "model_load",
        device=model_device,
        compute_device=request.device,
        executor=request.executor.value,
        dtype=str(_checkpoint_dtype(checkpoint.config)),
        attention_implementation=adapter.attention_implementation,
    ):
        with recorder.phase("setup"):
            with recorder.phase("model_load"):
                model = load_causal_language_model(
                    request.snapshot,
                    torch_dtype=_checkpoint_dtype(checkpoint.config),
                    attention_implementation=adapter.attention_implementation,
                ).to(model_device)
    model.eval()
    if request.memory_plan is not None:
        observed = sample_device_memory()
        planned = request.memory_plan.stage("model_load")
        cast(Any, events).emit(
            "memory",
            "info",
            "memory.observation_recorded",
            stage_name="model_load",
            plan_revision=request.memory_plan.revision,
            predicted_gpu_bytes=planned.predicted_gpu_bytes,
            predicted_host_bytes=planned.predicted_host_bytes,
            **observed,
        )
        allocated = observed.get("cuda.allocated_bytes", 0)
        if allocated > planned.gpu_capacity_bytes:
            raise ResourceAdmissionError(
                f"RES001 model load observed {allocated} CUDA bytes but safe capacity is "
                f"{planned.gpu_capacity_bytes}"
            )
    return _ResidentEnvironment(
        source,
        checkpoint,
        adapter,
        inventory,
        tokens,
        quality_tokens,
        model,
        adapter.get_decoder_layers(model),
    )


def _restore_committed_state(
    request: ResidentQuantizationRequest,
    events: EventSink,
    recorder: PhaseRecorder,
    run_id: str,
    artifacts: LocalArtifactStore,
    tensors: LocalTensorStore,
    environment: _ResidentEnvironment,
    plan: QuantizationPlan,
    persisted_plan: PersistedPlan,
    initial_inputs: torch.Tensor,
) -> _ResidentResumeState:
    """Discover durable progress, restore the model shell, and load activations."""

    config_hash = _resident_config_hash(request)
    identity = CommitIdentity(
        config_hash,
        environment.inventory.model.config_hash,
        persisted_plan.reference.artifact_id,
    )
    journal = ProgressJournal(request.output / "state", run_id, artifacts)
    with _logged_operation(events, "resume_discovery"):
        with recorder.phase("resume"):
            with recorder.phase("discover"):
                discovery = journal.discover(plan, identity)
    discovered_records = (*discovery.valid_records, *discovery.orphan_records)
    events.emit(
        "resident-quantization",
        "info",
        "resume.discovery_completed",
        valid_records=len(discovery.valid_records),
        orphan_records=len(discovery.orphan_records),
        first_incomplete_block=(None if discovery.first_incomplete is None else discovery.first_incomplete.block),
        first_incomplete_layer=(None if discovery.first_incomplete is None else discovery.first_incomplete.layer),
    )
    block_records = sorted(
        (record for record in discovered_records if record.kind == "block"),
        key=lambda record: record.block,
    )
    with _logged_operation(events, "resume_commit_load", blocks=len(block_records)):
        with recorder.phase("resume"):
            with recorder.phase("load_commits"):
                committed_blocks = [
                    (
                        ArtifactRef(ArtifactTypes.BLOCK_RESULT, record.artifact_id, 1),
                        load_committed_block(
                            ArtifactRef(ArtifactTypes.BLOCK_RESULT, record.artifact_id, 1),
                            artifacts,
                            identity,
                        ).result,
                    )
                    for record in block_records
                ]
    if request.activation_retention == "rolling":
        for _reference, old_block in committed_blocks[:-1]:
            retire_block_activations(old_block, artifacts)
    accepted_bits = sum(layer.actual_bit_cost.total for _, block in committed_blocks for layer in block.layers) + sum(
        group.actual_bit_cost.total for _, block in committed_blocks for group in block.shared_input_groups
    )
    retry_bits_spent = sum(layer.extra_retry_bits for _, block in committed_blocks for layer in block.layers) + sum(
        group.extra_retry_bits for _, block in committed_blocks for group in block.shared_input_groups
    )
    budget = BudgetState(plan.planned_cost.total, accepted_bits, retry_bits_spent)
    with _logged_operation(
        events,
        "resume_activation_load",
        resumed=bool(committed_blocks),
        source_block=(None if not committed_blocks else committed_blocks[-1][1].block.index),
    ):
        with recorder.phase("resume"):
            with recorder.phase("activations"):
                if committed_blocks:
                    teacher_inputs, compressed_inputs = load_block_activations(
                        committed_blocks[-1][0], artifacts, "cpu"
                    )
                else:
                    teacher_inputs = initial_inputs
                    compressed_inputs = initial_inputs
    completed_block_indexes = {block.block.index for _, block in committed_blocks}
    layer_container = environment.adapter.get_decoder_layers(environment.model)
    if request.restore_completed_blocks:
        with _logged_operation(events, "resume_model_restore", blocks=len(committed_blocks)):
            with recorder.phase("resume"):
                with recorder.phase("restore"):
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
                        for group_state in completed_block.frozen_state.shared_input_groups:
                            frozen_group = SharedInputGroupFreezer().load(
                                group_state,
                                tensors,
                                device=request.device,
                                dtype=compressed_inputs.dtype,
                                backend="factorized",
                            )
                            BlockEditor().install_frozen_group(restored_block, frozen_group)
                        restore_block_auxiliary_parameters(
                            restored_block,
                            completed_block.frozen_state.auxiliary_parameters,
                            tensors,
                            device=request.device,
                        )
    released_decoder_blocks = _release_uncompleted_decoder_blocks(
        layer_container,
        completed_block_indexes if request.restore_completed_blocks else set(),
    )
    if request.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if released_decoder_blocks:
        events.emit(
            "resident-quantization",
            "info",
            "decoder_blocks_released",
            released_blocks=released_decoder_blocks,
        )
    partial_layer_records = {
        (record.block, record.layer): record
        for record in discovered_records
        if record.kind == "layer" and record.block not in completed_block_indexes
    }
    partial_group_records = {
        (record.block, record.layer): record
        for record in discovered_records
        if record.kind == "group" and record.block not in completed_block_indexes
    }
    return _ResidentResumeState(
        config_hash,
        identity,
        journal,
        discovered_records,
        committed_blocks,
        budget,
        teacher_inputs,
        compressed_inputs,
        completed_block_indexes,
        layer_container,
        partial_layer_records,
        partial_group_records,
    )


def _process_resident_block(
    request: ResidentQuantizationRequest,
    events: EventSink,
    recorder: PhaseRecorder,
    micro_recorder: PhaseRecorder,
    environment: _ResidentEnvironment,
    block_plan: BlockPlan,
    calibration: PersistedCalibration,
    tensors: LocalTensorStore,
    teacher_inputs: torch.Tensor,
    compressed_inputs: torch.Tensor,
    captured_metadata: dict[str, object],
    completed_block_indexes: set[int],
) -> _ResidentBlockWork:
    """Prepare one block's isolated source, teacher targets, and entry loss."""

    block_started = time.perf_counter()
    block_window = PeakWindow(request.device).start()
    deferred_slice = request.defer_layer_loss_snapshots
    block_index = block_plan.block.index
    metadata = _forward_metadata_to_device(_clone_forward_metadata(captured_metadata), request.device)
    events.emit(
        "resident-quantization",
        "info",
        "block.started",
        block=block_index,
        layers=len(block_plan.layer_order),
        factor_owners=len(block_plan.layers) + len(block_plan.shared_input_groups),
        completed_blocks=len(completed_block_indexes),
    )

    def tuning_forward(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        return environment.adapter.run_block(module, value, **metadata)

    with _logged_operation(events, "block_prepare", block=block_index, device=request.device):
        with _profile_block_phase(recorder, block_index, "prepare"):
            working_block = environment.adapter.load_block(environment.source, block_plan.block, request.device)
            working_block.eval()
    with _logged_operation(
        events,
        "block_teacher_forward",
        block=block_index,
        samples=int(teacher_inputs.shape[0]),
        batch_size=request.block_forward_batch_size,
        destination="cpu",
    ):
        with _profile_block_phase(recorder, block_index, "teacher_forward"):
            with torch.no_grad():
                teacher_outputs = _run_block_batched(
                    environment.adapter,
                    working_block,
                    teacher_inputs,
                    metadata,
                    request.block_forward_batch_size,
                    "cpu",
                    recorder=micro_recorder,
                ).detach()
    block_output_stats = next(
        (
            item
            for item in calibration.stats.layers
            if item.layer.block.index == block_index and item.layer.path == "mlp.down_proj"
        ),
        next(item for item in calibration.stats.layers if item.layer.block.index == block_index),
    )
    with _logged_operation(
        events,
        "block_entry_loss",
        block=block_index,
        deferred=deferred_slice,
        samples=int(compressed_inputs.shape[0]),
    ):
        with _profile_block_phase(recorder, block_index, "entry_loss"):
            with tensors.read(block_output_stats.output_importance, request.device) as value:
                block_output_importance = value.clone()
            loss_recorder = BlockLossRecorder()
            loss_recorder.record_target_weighted_mean_square(
                _weighted_target_mean_square(teacher_outputs, block_output_importance)
            )
            loss_recorder.record_source_reference(
                _self_reference_weighted_mse(teacher_outputs, block_output_importance)
            )
            loss_recorder.record_block_entry(
                0.0
                if deferred_slice
                else _block_loss(
                    environment.adapter,
                    working_block,
                    compressed_inputs,
                    teacher_outputs,
                    block_output_importance,
                    metadata,
                    request.block_forward_batch_size,
                    micro_recorder,
                )
            )
    if request.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return _ResidentBlockWork(
        block_started,
        block_window,
        block_index,
        metadata,
        tuning_forward,
        working_block,
        teacher_outputs,
        block_output_importance,
        loss_recorder,
        deferred_slice,
    )


def _run_resident_quantization_impl(
    request: ResidentQuantizationRequest,
    events: EventSink,
    recorder: PhaseRecorder,
    run_id: str,
) -> ResidentQuantizationResult:
    """Quantize all decoder linears while the source model remains resident on one device."""
    started = time.perf_counter()
    _validate_resident_request(request)
    micro_recorder = recorder if request.profiling.level is ProfilingLevel.MICRO else NULL_RECORDER
    artifacts = LocalArtifactStore(request.output / "artifacts", recorder=micro_recorder)
    tensors = LocalTensorStore(artifacts)
    executor = ResidentExecutor()
    context = StageContext(run_id, executor, artifacts, tensors, events, Cancellation(), recorder)
    environment = _setup_resident_environment(request, events, recorder)
    source = environment.source
    checkpoint = environment.checkpoint
    adapter = environment.adapter
    inventory = environment.inventory
    tokens = environment.tokens
    quality_tokens = environment.quality_tokens
    model = environment.model
    decoder_layers = environment.decoder_layers
    reference_logits = None
    if request.evaluate_inline_quality:
        with _logged_operation(
            events,
            "reference_quality",
            samples=int(quality_tokens.shape[0]),
            input_elements=int(quality_tokens.numel()),
        ):
            with recorder.phase("setup"):
                with recorder.phase("reference_quality"):
                    # Held on CPU for the whole run: on-device reference logits
                    # would otherwise occupy sample*sequence*vocabulary bytes of
                    # VRAM through every block's factorization and tuning.
                    reference_logits = _run_quality_logits_batched(adapter, model, quality_tokens, "cpu")
    with _logged_operation(events, "prefix_metadata_capture", samples=1):
        with recorder.phase("setup"):
            with recorder.phase("prefix_capture"):
                capture = capture_prefix_invocations(
                    decoder_layers[0],
                    (lambda: adapter.run_decoder_forward(model, tokens[:1]),),
                )[0]
    captured_input = capture.positional[0]
    if not isinstance(captured_input, torch.Tensor):
        raise TypeError("captured first-block hidden state is not a tensor")
    with _logged_operation(
        events,
        "prefix_activation_capture",
        samples=int(tokens.shape[0]),
        batch_size=request.block_forward_batch_size,
        destination="cpu",
    ):
        with recorder.phase("setup"):
            with recorder.phase("prefix_capture"):
                initial_inputs = _run_prefix_batched(
                    adapter,
                    model,
                    tokens,
                    request.block_forward_batch_size,
                    "cpu",
                ).detach()
    if not torch.equal(initial_inputs[:1], captured_input.detach().cpu()):
        raise ValueError("adapter prefix does not match the model's first-block input")
    captured_metadata = capture.keyword
    if request.device.startswith("cuda"):
        # Prefix capture traverses the full embedding/prefix path in large
        # no-grad batches. Its workspaces are dead once activations are on CPU.
        torch.cuda.empty_cache()
    selected_forward_batch = _autotune_block_forward_batch(
        request,
        adapter,
        source,
        inventory,
        decoder_layers,
        initial_inputs,
        captured_metadata,
        events,
    )
    selected_tuning_microbatch, selected_refit_microbatch = _autotune_tuning_microbatch(
        request,
        adapter,
        source,
        inventory,
        decoder_layers,
        initial_inputs,
        captured_metadata,
        events,
    )
    tuning_enabled = (
        request.factorized_tuning_epochs > 0
        or request.nonfactorized_tuning_epochs > 0
        or any(request.nonfactorized_tuning_epochs_by_layer)
    )
    throughput_selections = tuple(
        (stage, selected)
        for stage, selected in (
            ("block_forward", selected_forward_batch),
            ("tuning", selected_tuning_microbatch),
            ("post_block_refit", selected_refit_microbatch),
        )
        if selected is not None
        and (
            stage == "block_forward"
            or (stage == "tuning" and tuning_enabled)
            or request.post_block_refit_epochs > 0
        )
        and request.memory_plan is not None
        and request.memory_plan.mode == "adaptive"
        and not any(
            warning.startswith(f"{stage} selected measured-throughput")
            for warning in request.memory_plan.warnings
        )
    )
    memory_plan = request.memory_plan
    memory_plan_reference = request.memory_plan_reference
    if throughput_selections and memory_plan is not None:
        memory_plan, revision, memory_plan_reference = revise_resident_memory_plan_for_throughput(
            memory_plan,
            request.output,
            throughput_selections,
        )
        cast(Any, events).emit(
            "memory",
            "info",
            "memory.plan_revised",
            revision=revision.revision,
            parent_revision=revision.parent_revision,
            reason=revision.reason,
            action=revision.action,
            algorithm_changed=revision.algorithm_changed,
            selections=dict(throughput_selections),
            artifact_id=memory_plan_reference.artifact_id,
        )
    request = replace(
        request,
        block_forward_batch_size=selected_forward_batch,
        tuning_microbatch_size=selected_tuning_microbatch,
        post_block_refit_microbatch_size=selected_refit_microbatch,
        memory_plan=memory_plan,
        memory_plan_reference=memory_plan_reference,
    )
    with _logged_operation(events, "preprocessing_lookup"):
        with recorder.phase("setup"):
            with recorder.phase("preprocessing"):
                token_bytes = tokens.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                dataset = DatasetIdentity(
                    "sha256:" + hashlib.sha256(token_bytes).hexdigest(),
                    ("deterministic-token-fixture",),
                    ("1",),
                    checkpoint.tokenizer_hash,
                    "raw-token-ids-v1",
                )
                preprocessing_references, preprocessing_source = _resolve_preprocessing_references(request)
                preprocessed = _load_precomputed_preprocessing(
                    request,
                    artifacts,
                    inventory,
                    dataset,
                    tokens.numel(),
                    preprocessing_references,
                )

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
        causal_batch_count = (tokens.shape[0] + request.calibration_batch_size - 1) // request.calibration_batch_size
        causal_progress_total = causal_batch_count * (2 if request.calibration_method == "two_phase_fisher" else 1)
        cast(Any, events).emit(
            "resident-quantization",
            "info",
            "calibration.progress_initialized",
            method=request.calibration_method,
            total_batches=causal_progress_total,
        )

        def causal_progress(completed: int, total: int) -> None:
            cast(Any, events).emit(
                "resident-quantization",
                "info",
                "calibration.progress_updated",
                method=request.calibration_method,
                completed_batches=completed,
                total_batches=total,
            )

        with _logged_operation(
            events,
            "calibration",
            method=request.calibration_method,
            layers=len(causal_layers),
            samples=int(tokens.shape[0]),
            batch_size=request.calibration_batch_size,
        ):
            with recorder.phase("calibrate", method=request.calibration_method):
                stats = calibrate_causal_model(
                    model,
                    tuple(
                        tokens[start : start + request.calibration_batch_size]
                        for start in range(0, tokens.shape[0], request.calibration_batch_size)
                    ),
                    tuple(causal_layers),
                    method=request.calibration_method,
                    shrinkage=request.calibration_shrinkage,
                    progress_callback=causal_progress,
                    recorder=micro_recorder,
                )
        cast(Any, events).emit(
            "resident-quantization",
            "info",
            "calibration.progress_completed",
            method=request.calibration_method,
            completed_batches=causal_progress_total,
            total_batches=causal_progress_total,
        )
        calibration_values.extend((causal_ids[item.path], item) for item in stats)
    elif request.calibration_method == "forward_only":
        calibration_inputs = initial_inputs
        forward_batch_count = (
            calibration_inputs.shape[0] + request.block_forward_batch_size - 1
        ) // request.block_forward_batch_size
        forward_progress_total = len(inventory.blocks) * forward_batch_count
        cast(Any, events).emit(
            "resident-quantization",
            "info",
            "calibration.progress_initialized",
            method="forward_only",
            total_batches=forward_progress_total,
            total_blocks=len(inventory.blocks),
        )
        for calibration_block_position, (block_inventory, resident_block) in enumerate(
            zip(inventory.blocks, decoder_layers, strict=True)
        ):
            streamed_block = request.executor is ExecutorKind.CPU_OFFLOAD
            if streamed_block:
                with _logged_operation(
                    events,
                    "calibration_block_prepare",
                    block=block_inventory.block.index,
                    device=request.device,
                ):
                    block = adapter.load_block(source, block_inventory.block, request.device)
                    block.eval()
            else:
                block = resident_block
            block_parameter = next(iter(block.parameters()), None)
            block_device = calibration_inputs.device if block_parameter is None else block_parameter.device
            metadata = _forward_metadata_to_device(_clone_forward_metadata(captured_metadata), str(block_device))
            paths = tuple(layer.path for layer in adapter.quantizable_layers(block, block_inventory.block))

            def calibration_runner(
                module: nn.Module,
                value: torch.Tensor,
                block_metadata: dict[str, object] = metadata,
            ) -> torch.Tensor:
                parameter = next(iter(module.parameters()), None)
                device = value.device if parameter is None else parameter.device
                return adapter.run_block(module, value.to(device, non_blocking=True), **block_metadata)

            def forward_progress(
                completed: int,
                _total: int,
                offset: int = calibration_block_position * forward_batch_count,
                block_index: int = block_inventory.block.index,
            ) -> None:
                cast(Any, events).emit(
                    "resident-quantization",
                    "info",
                    "calibration.progress_updated",
                    method="forward_only",
                    completed_batches=offset + completed,
                    total_batches=forward_progress_total,
                    block=block_index,
                )

            with _logged_operation(
                events,
                "calibration_block",
                method="forward_only",
                block=block_inventory.block.index,
                layers=len(paths),
                samples=int(calibration_inputs.shape[0]),
                batch_size=request.block_forward_batch_size,
                device=str(block_device),
                streamed_source=streamed_block,
            ):
                with recorder.phase("calibrate", block=block_inventory.block.index, method="forward_only"):
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
                        progress_callback=forward_progress,
                        recorder=micro_recorder,
                    )
            calibration_values.extend((LayerId(block_inventory.block, item.path), item) for item in stats)
            with torch.no_grad():
                calibration_inputs = _run_block_batched(
                    adapter,
                    block,
                    calibration_inputs,
                    metadata,
                    request.block_forward_batch_size,
                    recorder=micro_recorder,
                ).detach()
            if streamed_block:
                del block
                if request.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        cast(Any, events).emit(
            "resident-quantization",
            "info",
            "calibration.progress_completed",
            method="forward_only",
            completed_batches=forward_progress_total,
            total_batches=forward_progress_total,
            total_blocks=len(inventory.blocks),
        )
    else:
        raise ValueError(f"unsupported resident calibration method: {request.calibration_method}")
    if preprocessed is None:
        with _logged_operation(
            events,
            "calibration_persist",
            layers=len(calibration_values),
            input_elements=int(tokens.numel()),
        ):
            with recorder.phase("calibrate"):
                with recorder.phase("persist"):
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
        with _logged_operation(events, "objective_build", layers=len(calibration_values)):
            with recorder.phase("plan"):
                with recorder.phase("objectives"):
                    objectives = build_objectives(calibration, ObjectiveConfig(), artifacts)
        resolved_groups = _resolve_shared_input_groups(adapter, inventory, request.shared_input_groups)
        with _logged_operation(
            events,
            "sensitivity_analysis",
            enabled=request.allocation_strategy in {
                AllocationStrategy.SENSITIVITY,
                AllocationStrategy.KL_CALIBRATED,
            },
            alpha=request.rank_sensitivity_alpha,
        ):
            with recorder.phase("plan"):
                with recorder.phase("sensitivity"):
                    if request.allocation_strategy is AllocationStrategy.SENSITIVITY:
                        sensitivity_profile = _legacy_sensitivity_profile(
                            inventory,
                            calibration,
                            source,
                            tensors,
                            alpha=request.rank_sensitivity_alpha,
                            device=request.device,
                        )
                    elif request.allocation_strategy is AllocationStrategy.KL_CALIBRATED:
                        sensitivity_profile = _load_requested_kl_sensitivities(
                            request,
                            inventory,
                            resolved_groups,
                        )
                    else:
                        sensitivity_profile = ()
        allocation = RankAllocationConfig(
            target_bpw=request.target_bpw,
            strategy=request.allocation_strategy,
            sensitivity_alpha=request.rank_sensitivity_alpha,
            kl_profile_artifact=request.kl_profile_artifact,
            kl_profile_key=request.kl_profile_key,
            kl_sensitivity_granularity=request.kl_sensitivity_granularity,
            maximum_rank_layer_patterns=request.maximum_rank_layer_patterns,
            layer_budget_multipliers=request.layer_budget_multipliers,
            bounds=RankBoundsConfig(
                multiple=request.rank_multiple,
                floor_fraction_of_uniform=request.rank_floor_fraction,
                ceiling_fraction_of_uniform=request.rank_ceiling_fraction,
                edge_block_boost=request.rank_edge_boost,
            ),
            retry=request.rank_retry,
            reconstruction=request.reconstruction_rank_planning,
        )
        with _logged_operation(
            events,
            "rank_planning",
            strategy=request.allocation_strategy.value,
            target_bpw=request.target_bpw,
            rank_multiple=request.rank_multiple,
            maximum_rank_layer_patterns=request.maximum_rank_layer_patterns,
            layer_budget_multipliers=request.layer_budget_multipliers,
        ):
            with recorder.phase("plan"):
                with recorder.phase("ranks"):
                    reconstruction_profile = None
                    reconstruction_decisions: tuple[ReconstructionRankDecision, ...] = ()
                    if request.allocation_strategy in {
                        AllocationStrategy.RECONSTRUCTION_AWARE,
                        AllocationStrategy.KL_CALIBRATED,
                    }:
                        baseline_allocation = replace(
                            allocation,
                            strategy=AllocationStrategy.UNIFORM,
                            maximum_rank_layer_patterns=(),
                            layer_budget_multipliers=(),
                            retry=replace(allocation.retry, enabled=False),
                            reconstruction=ReconstructionRankPlanningConfig(),
                        )
                        baseline_plan = build_quantization_plan(
                            PlanningRequest(
                                inventory,
                                calibration.stats,
                                calibration.reference,
                                objectives.objectives,
                                baseline_allocation,
                                request.outliers,
                                (),
                                resolved_groups,
                                bias_correction=request.bias_correction,
                                low_rank_patch=request.low_rank_patch,
                            )
                        )
                        probe_plan = _persist_rank_probe_plan(
                            request,
                            baseline_plan,
                            artifacts,
                        )
                        cast(Any, events).emit(
                            "resident-quantization",
                            "info",
                            "rank_probe.plan_committed",
                            artifact_id=probe_plan.artifact_id,
                            unit_count=len(_rank_probe_units(baseline_plan)),
                        )
                        reconstruction_profile, reconstruction_decisions = _run_reconstruction_rank_probes(
                            request,
                            baseline_plan,
                            probe_plan,
                            source,
                            calibration,
                            tensors,
                            artifacts,
                            events,
                            recorder,
                            sensitivity_profile
                            if request.allocation_strategy is AllocationStrategy.KL_CALIBRATED
                            else (),
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
                            resolved_groups,
                            reconstruction_profile,
                            reconstruction_decisions,
                            request.bias_correction,
                            request.low_rank_patch,
                        )
                    )
                    persisted_plan = persist_plan(plan, artifacts)
    else:
        calibration, objectives, persisted_plan = preprocessed
        plan = persisted_plan.plan
    with recorder.phase("plan"):
        with recorder.phase("activate"):
            _write_active_preprocessing_state(
                request,
                calibration.reference,
                objectives.reference,
                persisted_plan.reference,
            )
            events.emit(
                "resident-quantization",
                "info",
                "preprocessing.selected",
                reused=preprocessed is not None,
                source=preprocessing_source,
                calibration_artifact=calibration.reference.artifact_id,
                objectives_artifact=objectives.reference.artifact_id,
                plan_artifact=persisted_plan.reference.artifact_id,
                blocks=len(plan.blocks),
                planned_bits=plan.planned_cost.total,
            )
    resume = _restore_committed_state(
        request,
        events,
        recorder,
        run_id,
        artifacts,
        tensors,
        environment,
        plan,
        persisted_plan,
        initial_inputs,
    )
    config_hash = resume.config_hash
    identity = resume.identity
    journal = resume.journal
    discovered_records = resume.discovered_records
    committed_blocks = resume.committed_blocks
    budget = resume.budget
    teacher_inputs = resume.teacher_inputs
    compressed_inputs = resume.compressed_inputs
    completed_block_indexes = resume.completed_block_indexes
    layer_container = resume.layer_container
    partial_layer_records = resume.partial_layer_records
    partial_group_records = resume.partial_group_records
    completed_wall_seconds = sum(block.wall_seconds for _reference, block in committed_blocks)
    cast(Any, events).emit(
        "resident-quantization",
        "info",
        "compression.progress_initialized",
        total_blocks=len(plan.blocks),
        completed_blocks=len(completed_block_indexes),
        completed_wall_seconds=completed_wall_seconds,
        mean_block_seconds=(completed_wall_seconds / len(committed_blocks) if committed_blocks else None),
    )
    live_layers = {
        (layer.layer.block.index, layer.layer.path): layer
        for _reference, block in committed_blocks
        for layer in block.layers
    }
    live_groups = {
        (group.block.index, group.name): group
        for _reference, block in committed_blocks
        for group in block.shared_input_groups
    }
    for (partial_block, partial_path), partial_record in partial_layer_records.items():
        if partial_path is None:
            raise ValueError("partial layer journal record is missing its layer path")
        partial_layer = load_committed_layer(
            ArtifactRef(ArtifactTypes.LAYER_RESULT, partial_record.artifact_id, 1),
            artifacts,
            identity,
        ).result
        live_layers[(partial_block, partial_path)] = partial_layer
    for (partial_block, partial_name), partial_record in partial_group_records.items():
        if partial_name is None:
            raise ValueError("partial shared-input group journal record is missing its name")
        partial_group = load_committed_shared_input_group(
            ArtifactRef(
                ArtifactTypes.SHARED_INPUT_GROUP_RESULT,
                partial_record.artifact_id,
                1,
            ),
            artifacts,
            identity,
        ).result
        live_groups[(partial_block, partial_name)] = partial_group
    live_blocks = {block.block.index: block for _reference, block in committed_blocks}
    live_layer_order: tuple[str, ...] = ()
    if plan.blocks:
        first_block_groups = {group.name: group for group in plan.blocks[0].shared_input_groups}
        ordered_paths: list[str] = []
        unit_order = plan.blocks[0].unit_order or tuple(layer.layer.path for layer in plan.blocks[0].layers)
        for unit in unit_order:
            group = first_block_groups.get(unit)
            if group is None:
                ordered_paths.append(unit)
            else:
                ordered_paths.extend(member.layer.path for member in group.members)
        live_layer_order = tuple(ordered_paths)
    with recorder.phase("resume"):
        with recorder.phase("live_report"):
            update_live_weight_error_report(
                request.output,
                tuple(live_layers.values()),
                tuple(live_blocks.values()),
                groups=tuple(live_groups.values()),
                expected_blocks=len(plan.blocks),
                layer_order=live_layer_order,
            )
    del initial_inputs
    del decoder_layers
    peak_device_bytes = 0
    factorization_wall_seconds = 0.0
    new_layer_commits = 0
    new_block_commits = 0
    new_factorized_tuning_epoch_commits = 0
    factor_stage = FactorizationAttemptStage(
        request.admm,
        device=request.device,
        recorder=micro_recorder,
        record_admm_steps=request.observability.record_admm_steps,
        reset_peak_memory=False,
    )
    outlier_stage = OutlierSelectionStage(
        device=request.device,
        residual_probe_iterations=request.outliers.residual_probe.iterations,
        residual_probe_inner_iterations=request.admm.inner_iterations,
        transpose_wide=request.admm.transpose_wide,
    )
    scale_stage = ScaleFitStage(request.scale_fit, device=request.device)
    bias_stage = BiasCorrectionStage(request.bias_correction, device=request.device)
    patch_stage = LowRankPatchStage(request.low_rank_patch, device=request.device)
    bias_storage_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[request.bias_correction.storage_dtype.value]
    patch_storage_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[request.low_rank_patch.storage_dtype.value]

    for block_plan in plan.blocks:
        if block_plan.block.index in completed_block_indexes:
            continue
        block_work = _process_resident_block(
            request,
            events,
            recorder,
            micro_recorder,
            environment,
            block_plan,
            calibration,
            tensors,
            teacher_inputs,
            compressed_inputs,
            captured_metadata,
            completed_block_indexes,
        )
        block_started = block_work.started
        block_window = block_work.peak_window
        deferred_slice = block_work.deferred_slice
        block_index = block_work.index
        metadata = block_work.metadata
        tuning_forward = block_work.tuning_forward
        working_block = block_work.working_block
        teacher_outputs = block_work.teacher_outputs
        block_output_importance = block_work.output_importance
        loss_recorder = block_work.loss_recorder
        block_target_weighted_mean_square = loss_recorder.target_weighted_mean_square
        if block_target_weighted_mean_square is None:
            raise AssertionError("resident block is missing teacher activation power")
        layer_results: list[LayerResult] = []
        group_results: list[SharedInputGroupResult] = []
        frozen_states: list[FrozenNanoQuantState] = []
        frozen_group_states: list[FrozenSharedInputGroupState] = []
        quantization_targets: dict[str, TensorRef] = {}
        group_member_targets: dict[str, tuple[TensorRef, ...]] = {}
        tuning_recorder = micro_recorder
        del block_work
        # Admit caches only after the active source/working block and its
        # teacher forward have established the real per-block free-memory
        # envelope.  Checking before block materialization would overestimate
        # capacity and could turn an optional cache into an OOM.
        if request.activation_gpu_cache in {
            ActivationGpuCacheMode.INPUTS,
            ActivationGpuCacheMode.BOTH,
            ActivationGpuCacheMode.AUTO,
        }:
            compressed_inputs = _cache_activation_tensor(
                compressed_inputs,
                request,
                events,
                role="compressed_inputs",
                required=request.activation_gpu_cache is not ActivationGpuCacheMode.AUTO,
            )
        if request.activation_gpu_cache in {
            ActivationGpuCacheMode.BOTH,
            ActivationGpuCacheMode.AUTO,
        }:
            teacher_outputs = _cache_activation_tensor(
                teacher_outputs,
                request,
                events,
                role="teacher_outputs",
                required=request.activation_gpu_cache is ActivationGpuCacheMode.BOTH,
            )

        unit_schedule = block_plan.unit_order or (
            tuple(group.name for group in block_plan.shared_input_groups)
            + tuple(layer.layer.path for layer in block_plan.layers)
        )
        for unit_position, unit_name in enumerate(unit_schedule):
            for group_plan in block_plan.shared_input_groups:
                if group_plan.name != unit_name:
                    continue
                group_started = time.perf_counter()
                group_position = unit_position
                nonfactorized_epochs = _nonfactorized_epochs(request, group_position)
                events.emit(
                    "resident-quantization",
                    "info",
                    "shared_input_group.started",
                    block=block_index,
                    group=group_plan.name,
                    members=tuple(member.layer.path for member in group_plan.members),
                    position=group_position,
                    planned_rank=group_plan.rank,
                    outlier_columns=group_plan.outliers.count,
                )
                if nonfactorized_epochs > 0:
                    with _profile_layer_phase(recorder, block_index, group_plan.name, "nonfactorized_tuning"):
                        tune_non_factorized(
                            working_block,
                            TuningRequest(
                                compressed_inputs,
                                teacher_outputs,
                                nonfactorized_epochs,
                                request.nonfactorized_tuning_batch_size,
                                request.nonfactorized_tuning_learning_rate,
                                early_stop_relative_tolerance=request.nonfactorized_tuning_early_stop_relative_tolerance,
                                output_importance=block_output_importance,
                                seed=_tuning_seed(request, "nonfactorized-tuning", block_index, group_plan.name),
                                microbatch_size=request.tuning_microbatch_size,
                                restore_best_state=request.restore_best_tuning_state,
                                epoch_loss_mode=request.tuning_epoch_loss_mode,
                            ),
                            tuning_forward,
                            tuning_recorder,
                        )
                with _profile_layer_phase(recorder, block_index, group_plan.name, "materialize"):
                    synthetic_plan, source_ref, member_source_refs = _materialize_shared_input_plan(
                        group_plan,
                        working_block,
                        tensors,
                        device=request.device,
                    )
                    quantization_targets[group_plan.name] = source_ref
                    group_member_targets[group_plan.name] = member_source_refs
                prior_record = partial_group_records.get((block_index, group_plan.name))
                if prior_record is not None:
                    prior_group = load_committed_shared_input_group(
                        ArtifactRef(
                            ArtifactTypes.SHARED_INPUT_GROUP_RESULT,
                            prior_record.artifact_id,
                            1,
                        ),
                        artifacts,
                        identity,
                    ).result
                    frozen_group = SharedInputGroupFreezer().load(
                        prior_group.frozen_state,
                        tensors,
                        device=request.device,
                        dtype=compressed_inputs.dtype,
                        backend="factorized",
                    )
                    BlockEditor().install_frozen_group(working_block, frozen_group)
                    frozen_group_states.append(prior_group.frozen_state)
                    group_results.append(prior_group)
                    budget = replace(
                        budget,
                        accepted_bits=budget.accepted_bits + prior_group.actual_bit_cost.total,
                        retry_bits_spent=budget.retry_bits_spent + prior_group.extra_retry_bits,
                    )
                    if not deferred_slice:
                        loss_recorder.record_after_layer(
                            LayerId(block_plan.block, group_plan.name),
                            _block_loss(
                                adapter,
                                working_block,
                                compressed_inputs,
                                teacher_outputs,
                                block_output_importance,
                                metadata,
                                request.block_forward_batch_size,
                                micro_recorder,
                            ),
                        )
                    events.emit(
                        "resident-quantization",
                        "info",
                        "shared_input_group.completed",
                        block=block_index,
                        group=group_plan.name,
                        status="reused",
                        artifact_id=prior_record.artifact_id,
                        journal_sequence=prior_record.sequence,
                        rank=prior_group.frozen_state.rank,
                        accepted_bits=budget.accepted_bits,
                        retry_bits_spent=budget.retry_bits_spent,
                        wall_seconds=time.perf_counter() - group_started,
                    )
                    continue
                with _profile_layer_phase(recorder, block_index, group_plan.name, "factorize"):
                    accepted, outliers, fitted = _run_resident_factorization_attempts(
                        synthetic_plan,
                        source_ref,
                        request,
                        budget,
                        context,
                        config_hash,
                        factor_stage,
                        outlier_stage,
                        scale_stage,
                        recorder,
                    )
                factorized = accepted.result
                bias_correction = _run_bias_correction(
                    synthetic_plan,
                    source_ref,
                    factorized,
                    outliers,
                    request,
                    context,
                    bias_stage,
                )
                peak_device_bytes = max(peak_device_bytes, accepted.peak_workspace_bytes)
                factorization_wall_seconds += accepted.wall_seconds
                scales = factorized.factors.scales
                if scales.mid is None:
                    raise AssertionError("shared-input factorizer omitted required mid scale")
                outlier_indices = outlier_values = outlier_scales = None
                bias = None
                if bias_correction is not None:
                    with tensors.read(bias_correction.bias, request.device) as value:
                        bias = value.clone()
                if group_plan.outliers.count:
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
                    tensors.read(scales.mid, request.device) as scale_mid,
                    tensors.read(scales.post, request.device) as scale_post,
                ):
                    trainable_group = TrainableSharedInputFactorGroup(
                        left,
                        right,
                        scale_pre,
                        scale_mid,
                        scale_post,
                        bias=bias,
                        outlier_indices=outlier_indices,
                        outlier_values=outlier_values,
                        outlier_scales=outlier_scales,
                    ).to(device=request.device, dtype=compressed_inputs.dtype)
                member_ids = tuple(member.layer for member in group_plan.members)
                member_widths = tuple(member.out_features for member in group_plan.members)
                editor = BlockEditor()
                editor.install_trainable_group(
                    working_block,
                    group_plan.name,
                    member_ids,
                    member_widths,
                    trainable_group,
                )
                tuning = None
                if request.factorized_tuning_epochs > 0:
                    owner_path = "_nanoquant_shared_input_groups." + BlockEditor._group_key(group_plan.name)
                    with _profile_layer_phase(recorder, block_index, group_plan.name, "factorized_tuning"):
                        tuning = tune_factorized(
                            working_block,
                            owner_path,
                            TuningRequest(
                                compressed_inputs,
                                teacher_outputs,
                                request.factorized_tuning_epochs,
                                request.factorized_tuning_batch_size,
                                request.factorized_tuning_learning_rate,
                                output_importance=block_output_importance,
                                seed=_tuning_seed(request, "factorized-tuning", block_index, group_plan.name),
                                microbatch_size=request.tuning_microbatch_size,
                                restore_best_state=request.restore_best_tuning_state,
                                epoch_loss_mode=request.tuning_epoch_loss_mode,
                            ),
                            tuning_forward,
                            tuning_recorder,
                        )
                with _profile_layer_phase(recorder, block_index, group_plan.name, "freeze"):
                    frozen_group = SharedInputGroupFreezer().freeze(
                        member_ids,
                        group_plan.name,
                        member_widths,
                        trainable_group,
                        tensors,
                        backend="factorized",
                        bias_storage_dtype=bias_storage_dtype,
                    )
                    editor.install_frozen_group(working_block, frozen_group)
                    frozen_group_states.append(frozen_group.state)
                    with (
                        tensors.read(source_ref, request.device) as source_value,
                        tensors.read(synthetic_plan.objective.input_importance, request.device) as input_importance,
                        tensors.read(synthetic_plan.objective.output_importance, request.device) as output_importance,
                    ):
                        dense_group = frozen_group.owner.dense_weight()
                        final_metrics = reconstruction_metrics(
                            source_value,
                            dense_group,
                            input_importance,
                            output_importance,
                        )
                        member_metrics = []
                        row_start = 0
                        for member, member_source, objective in zip(
                            group_plan.members,
                            member_source_refs,
                            group_plan.objectives,
                            strict=True,
                        ):
                            row_end = row_start + member.out_features
                            with (
                                tensors.read(member_source, request.device) as member_value,
                                tensors.read(objective.input_importance, request.device) as member_input,
                                tensors.read(objective.output_importance, request.device) as member_output,
                            ):
                                member_metrics.append(
                                    (
                                        member.layer,
                                        reconstruction_metrics(
                                            member_value,
                                            dense_group[row_start:row_end],
                                            member_input,
                                            member_output,
                                        ),
                                    )
                                )
                            row_start = row_end
                    accepted_attempt = next(
                        index for index, attempt in enumerate(accepted.attempts) if attempt.accepted
                    )
                    group_result = SharedInputGroupResult(
                        1,
                        group_plan.block,
                        group_plan.name,
                        group_plan,
                        accepted.attempts,
                        accepted_attempt,
                        factorized.factors.left_binary.artifact,
                        fitted,
                        tuning,
                        frozen_group.state,
                        final_metrics,
                        tuple(member_metrics),
                        accepted.actual_bit_cost,
                        accepted.extra_retry_bits,
                        (),
                        bias_correction,
                    )
                with _logged_operation(
                    events,
                    "shared_input_group_commit",
                    block=block_index,
                    group=group_plan.name,
                    rank=group_result.frozen_state.rank,
                ):
                    with _profile_layer_phase(recorder, block_index, group_plan.name, "commit"):
                        committed_group = commit_shared_input_group(
                            group_result,
                            artifacts,
                            identity,
                        )
                        group_journal_record = journal.append(
                            "group",
                            block_index,
                            group_plan.name,
                            committed_group.reference.artifact_id,
                            identity,
                        )
                        clear_tuning_checkpoint(request.output)
                        events.emit(
                            "resident-quantization",
                            "info",
                            "shared_input_group.committed",
                            block=block_index,
                            group=group_plan.name,
                            artifact_id=committed_group.reference.artifact_id,
                            journal_sequence=group_journal_record.sequence,
                            rank=group_result.frozen_state.rank,
                            accepted_attempt=group_result.accepted_attempt,
                            actual_bits=group_result.actual_bit_cost.total,
                            extra_retry_bits=group_result.extra_retry_bits,
                            weighted_error=(group_result.final_reconstruction.export_weighted_normalized_error),
                            raw_error=group_result.final_reconstruction.raw_normalized_error,
                        )
                        live_groups[(block_index, group_plan.name)] = group_result
                        update_live_weight_error_report(
                            request.output,
                            tuple(live_layers.values()),
                            tuple(live_blocks.values()),
                            groups=tuple(live_groups.values()),
                            expected_blocks=len(plan.blocks),
                            layer_order=live_layer_order,
                        )
                new_layer_commits += 1
                if (
                    request.interrupt_after_layer_commits is not None
                    and new_layer_commits >= request.interrupt_after_layer_commits
                ):
                    raise InterruptedError(f"injected interruption after {new_layer_commits} new physical-unit commits")
                group_results.append(group_result)
                budget = accepted.budget
                if not deferred_slice:
                    loss_recorder.record_after_layer(
                        LayerId(block_plan.block, group_plan.name),
                        _block_loss(
                            adapter,
                            working_block,
                            compressed_inputs,
                            teacher_outputs,
                            block_output_importance,
                            metadata,
                            request.block_forward_batch_size,
                            micro_recorder,
                        ),
                    )
                events.emit(
                    "resident-quantization",
                    "info",
                    "shared_input_group.completed",
                    block=block_index,
                    group=group_plan.name,
                    rank=frozen_group.state.rank,
                    actual_bits=group_result.actual_bit_cost.total,
                    weighted_error=group_result.final_reconstruction.export_weighted_normalized_error,
                    raw_error=group_result.final_reconstruction.raw_normalized_error,
                    wall_seconds=time.perf_counter() - group_started,
                )

            for layer_position, layer_plan in enumerate(block_plan.layers):
                if layer_plan.layer.path != unit_name:
                    continue
                layer_position = unit_position
                layer_started = time.perf_counter()
                nonfactorized_epochs = _nonfactorized_epochs(request, layer_position)
                events.emit(
                    "resident-quantization",
                    "info",
                    "layer.started",
                    block=block_index,
                    layer=layer_plan.layer.path,
                    position=layer_position,
                    planned_rank=layer_plan.rank,
                    outlier_columns=layer_plan.outliers.count,
                    nonfactorized_tuning_epochs=nonfactorized_epochs,
                    factorized_tuning_epochs=request.factorized_tuning_epochs,
                )
                if nonfactorized_epochs > 0:
                    with _logged_operation(
                        events,
                        "nonfactorized_tuning",
                        block=block_index,
                        layer=layer_plan.layer.path,
                        epochs=nonfactorized_epochs,
                        batch_size=request.nonfactorized_tuning_batch_size,
                        microbatch_size=request.tuning_microbatch_size,
                        learning_rate=request.nonfactorized_tuning_learning_rate,
                    ):
                        with _profile_layer_phase(
                            recorder,
                            block_index,
                            layer_plan.layer.path,
                            "nonfactorized_tuning",
                        ):
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
                                        request,
                                        "nonfactorized-tuning",
                                        block_index,
                                        layer_plan.layer.path,
                                    ),
                                    microbatch_size=request.tuning_microbatch_size,
                                    epoch_observer=_epoch_cooldown_observer(
                                        request.nonfactorized_tuning_epoch_cooldown_seconds,
                                        events,
                                        tuning_kind="nonfactorized",
                                        block=block_index,
                                        layer=layer_plan.layer.path,
                                        target_weighted_mean_square=block_target_weighted_mean_square,
                                    ),
                                    restore_best_state=request.restore_best_tuning_state,
                                    epoch_loss_mode=request.tuning_epoch_loss_mode,
                                ),
                                tuning_forward,
                                tuning_recorder,
                            )
                with _profile_layer_phase(recorder, block_index, layer_plan.layer.path, "materialize"):
                    source_linear = _module_at_path(working_block, layer_plan.layer.path)
                    source_weight = getattr(source_linear, "weight", None)
                    if not isinstance(source_weight, torch.Tensor):
                        raise TypeError(f"quantization target has no materialized weight: {layer_plan.layer.path}")
                    source_ref = tensors.put("source-layer", {"weight": source_weight.detach().cpu()})["weight"]
                quantization_targets[layer_plan.layer.path] = source_ref
                prior_record = partial_layer_records.get((block_index, layer_plan.layer.path))
                if prior_record is not None:
                    prior = load_committed_layer(
                        ArtifactRef(ArtifactTypes.LAYER_RESULT, prior_record.artifact_id, 1), artifacts, identity
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
                    events.emit(
                        "resident-quantization",
                        "info",
                        "layer.reused",
                        block=block_index,
                        layer=layer_plan.layer.path,
                        artifact_id=prior_record.artifact_id,
                        journal_sequence=prior_record.sequence,
                        rank=prior.frozen_state.rank,
                    )
                    budget = replace(
                        budget,
                        accepted_bits=budget.accepted_bits + prior.actual_bit_cost.total,
                        retry_bits_spent=budget.retry_bits_spent + prior.extra_retry_bits,
                    )
                    if not deferred_slice:
                        with _profile_layer_phase(recorder, block_index, layer_plan.layer.path, "loss_snapshot"):
                            loss_recorder.record_after_layer(
                                layer_plan.layer,
                                _block_loss(
                                    adapter,
                                    working_block,
                                    compressed_inputs,
                                    teacher_outputs,
                                    block_output_importance,
                                    metadata,
                                    request.block_forward_batch_size,
                                    micro_recorder,
                                ),
                            )
                    events.emit(
                        "resident-quantization",
                        "info",
                        "layer.completed",
                        block=block_index,
                        layer=layer_plan.layer.path,
                        status="reused",
                        rank=prior.frozen_state.rank,
                        accepted_bits=budget.accepted_bits,
                        retry_bits_spent=budget.retry_bits_spent,
                        wall_seconds=time.perf_counter() - layer_started,
                    )
                    continue
                with _profile_layer_phase(recorder, block_index, layer_plan.layer.path, "factorize"):
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
                        recorder,
                    )
                with _profile_layer_phase(recorder, block_index, layer_plan.layer.path, "materialize"):
                    factorized = accepted.result
                    bias_correction = _run_bias_correction(
                        layer_plan,
                        source_ref,
                        factorized,
                        outliers,
                        request,
                        context,
                        bias_stage,
                    )
                    low_rank_patch = _run_low_rank_patch(
                        layer_plan,
                        source_ref,
                        factorized,
                        outliers,
                        bias_correction,
                        request,
                        adapter,
                        working_block,
                        compressed_inputs,
                        metadata,
                        context,
                        tensors,
                        patch_stage,
                    )
                    accepted = _account_for_patch_acceptance(accepted, low_rank_patch)
                    peak_device_bytes = max(peak_device_bytes, accepted.peak_workspace_bytes)
                    factorization_wall_seconds += accepted.wall_seconds
                    scales = factorized.factors.scales
                    mid_ref = scales.mid
                    if mid_ref is None:
                        raise AssertionError("factorizer omitted required mid scale")
                    outlier_indices = None
                    outlier_values = None
                    outlier_scales = None
                    bias = None
                    patch_left = patch_right = None
                    if bias_correction is not None:
                        with tensors.read(bias_correction.bias, request.device) as value:
                            bias = value.clone()
                    if low_rank_patch is not None and low_rank_patch.accepted:
                        with (
                            tensors.read(low_rank_patch.left, request.device) as left_value,
                            tensors.read(low_rank_patch.right, request.device) as right_value,
                        ):
                            patch_left = left_value.clone()
                            patch_right = right_value.clone()
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
                            bias=bias,
                            outlier_indices=outlier_indices,
                            outlier_values=outlier_values,
                            outlier_scales=outlier_scales,
                            patch_left=patch_left,
                            patch_right=patch_right,
                        ).to(device=request.device, dtype=compressed_inputs.dtype)
                tuning = None
                if request.factorized_tuning_epochs > 0:
                    BlockEditor().install_trainable_layer(working_block, layer_plan.layer.path, trainable)
                    tuning_checkpoint_identity = TuningCheckpointIdentity(
                        identity.config_hash,
                        identity.model_hash,
                        identity.plan_hash,
                        block_index,
                        layer_plan.layer.path,
                        "factorized",
                    )
                    active_checkpoint = active_tuning_checkpoint(request.output, tuning_checkpoint_identity)
                    events.emit(
                        "resident-quantization",
                        "info",
                        "factorized_tuning.resume_checkpoint",
                        block=block_index,
                        layer=layer_plan.layer.path,
                        found=active_checkpoint is not None,
                        completed_epochs=(
                            None if active_checkpoint is None else active_checkpoint.state.completed_epochs
                        ),
                    )

                    def checkpoint_sink(
                        state: TuningResumeState,
                        *,
                        checkpoint_identity: TuningCheckpointIdentity = tuning_checkpoint_identity,
                        checkpoint_block: int = block_index,
                        checkpoint_layer: str = layer_plan.layer.path,
                        checkpoint_target_power: float = block_target_weighted_mean_square,
                    ) -> None:
                        nonlocal new_factorized_tuning_epoch_commits
                        stored = save_tuning_checkpoint(request.output, state, checkpoint_identity)
                        new_factorized_tuning_epoch_commits += 1
                        events.emit(
                            "resident-quantization",
                            "info",
                            "factorized_tuning.epoch_checkpoint_committed",
                            block=checkpoint_block,
                            layer=checkpoint_layer,
                            completed_epochs=stored.state.completed_epochs,
                            generation=stored.generation,
                            loss=(stored.state.epoch_losses[-1] if stored.state.epoch_losses else None),
                            normalized_loss=(
                                None
                                if not stored.state.epoch_losses or stored.state.epoch_losses[-1] is None
                                else normalized_activation_error(
                                    stored.state.epoch_losses[-1],
                                    checkpoint_target_power,
                                )
                            ),
                            target_weighted_mean_square=checkpoint_target_power,
                            best_epoch=stored.state.best_epoch,
                            stopped_early=stored.state.stopped_early,
                        )
                        if request.factorized_tuning_epoch_cooldown_seconds:
                            time.sleep(request.factorized_tuning_epoch_cooldown_seconds)
                        if (
                            request.interrupt_after_factorized_tuning_epoch_commits is not None
                            and new_factorized_tuning_epoch_commits
                            >= request.interrupt_after_factorized_tuning_epoch_commits
                        ):
                            raise InterruptedError(
                                "injected interruption after "
                                f"{new_factorized_tuning_epoch_commits} factorized tuning epoch checkpoint(s)"
                            )

                    with _logged_operation(
                        events,
                        "factorized_tuning",
                        block=block_index,
                        layer=layer_plan.layer.path,
                        epochs=request.factorized_tuning_epochs,
                        completed_epochs=(0 if active_checkpoint is None else active_checkpoint.state.completed_epochs),
                        batch_size=request.factorized_tuning_batch_size,
                        microbatch_size=request.tuning_microbatch_size,
                        learning_rate=request.factorized_tuning_learning_rate,
                    ):
                        with _profile_layer_phase(
                            recorder,
                            block_index,
                            layer_plan.layer.path,
                            "factorized_tuning",
                        ):
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
                                    seed=_tuning_seed(
                                        request,
                                        "factorized-tuning",
                                        block_index,
                                        layer_plan.layer.path,
                                    ),
                                    microbatch_size=request.tuning_microbatch_size,
                                    restore_best_state=request.restore_best_tuning_state,
                                    epoch_loss_mode=request.tuning_epoch_loss_mode,
                                ),
                                tuning_forward,
                                tuning_recorder,
                                resume=None if active_checkpoint is None else active_checkpoint.state,
                                checkpoint_sink=checkpoint_sink,
                            )
                with _profile_layer_phase(recorder, block_index, layer_plan.layer.path, "freeze"):
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
                        bias_storage_dtype=bias_storage_dtype,
                        patch_storage_dtype=patch_storage_dtype,
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
                        else (
                            ("tuning_disabled",)
                            if request.scale_fit.enabled
                            else ("scale_fit_disabled", "tuning_disabled")
                        ),
                        bias_correction,
                        low_rank_patch,
                    )
                with _logged_operation(
                    events,
                    "layer_commit",
                    block=block_index,
                    layer=layer_plan.layer.path,
                    rank=layer_result.frozen_state.rank,
                ):
                    with _profile_layer_phase(recorder, block_index, layer_plan.layer.path, "commit"):
                        committed_layer = commit_layer(layer_result, artifacts, identity)
                        layer_journal_record = journal.append(
                            "layer",
                            block_index,
                            layer_plan.layer.path,
                            committed_layer.reference.artifact_id,
                            identity,
                        )
                        clear_tuning_checkpoint(request.output)
                        events.emit(
                            "resident-quantization",
                            "info",
                            "layer.committed",
                            block=block_index,
                            layer=layer_plan.layer.path,
                            artifact_id=committed_layer.reference.artifact_id,
                            journal_sequence=layer_journal_record.sequence,
                            rank=layer_result.frozen_state.rank,
                            accepted_attempt=layer_result.accepted_attempt,
                            actual_bits=layer_result.actual_bit_cost.total,
                            extra_retry_bits=layer_result.extra_retry_bits,
                            weighted_error=layer_result.final_reconstruction.export_weighted_normalized_error,
                            raw_error=layer_result.final_reconstruction.raw_normalized_error,
                        )
                        live_layers[(block_index, layer_plan.layer.path)] = layer_result
                        update_live_weight_error_report(
                            request.output,
                            tuple(live_layers.values()),
                            tuple(live_blocks.values()),
                            groups=tuple(live_groups.values()),
                            expected_blocks=len(plan.blocks),
                            layer_order=live_layer_order,
                        )
                new_layer_commits += 1
                if (
                    request.interrupt_after_layer_commits is not None
                    and new_layer_commits >= request.interrupt_after_layer_commits
                ):
                    raise InterruptedError(f"injected interruption after {new_layer_commits} new layer commits")
                layer_results.append(layer_result)
                budget = accepted.budget
                with _profile_layer_phase(recorder, block_index, layer_plan.layer.path, "loss_snapshot"):
                    loss_recorder.record_after_layer(
                        layer_plan.layer,
                        _block_loss(
                            adapter,
                            working_block,
                            compressed_inputs,
                            teacher_outputs,
                            block_output_importance,
                            metadata,
                            request.block_forward_batch_size,
                            micro_recorder,
                        ),
                    )
                events.emit(
                    "resident-quantization",
                    "info",
                    "layer.completed",
                    block=block_index,
                    layer=layer_plan.layer.path,
                    status="committed",
                    rank=layer_result.frozen_state.rank,
                    accepted_attempt=layer_result.accepted_attempt,
                    attempts=len(layer_result.attempts),
                    accepted_bits=budget.accepted_bits,
                    retry_bits_spent=budget.retry_bits_spent,
                    wall_seconds=time.perf_counter() - layer_started,
                )
        if request.post_block_refit_epochs > 0:
            trainable_by_path: dict[str, TrainableFactorizedLinear] = {}
            trainable_groups: dict[str, TrainableSharedInputFactorGroup] = {}
            for state in frozen_states:
                trainable = _rehydrate_trainable_layer(
                    state,
                    tensors,
                    device=request.device,
                    dtype=compressed_inputs.dtype,
                )
                BlockEditor().install_trainable_layer(working_block, state.layer.path, trainable)
                trainable_by_path[state.layer.path] = trainable
            for group_state in frozen_group_states:
                trainable_group = _rehydrate_trainable_group(
                    group_state,
                    tensors,
                    device=request.device,
                    dtype=compressed_inputs.dtype,
                )
                BlockEditor().install_trainable_group(
                    working_block,
                    group_state.name,
                    tuple(member.layer for member in group_state.members),
                    tuple(member.row_end - member.row_start for member in group_state.members),
                    trainable_group,
                )
                trainable_groups[group_state.name] = trainable_group
            with _logged_operation(
                events,
                "post_block_refit",
                block=block_index,
                layers=len(frozen_states),
                epochs=request.post_block_refit_epochs,
                batch_size=request.post_block_refit_batch_size,
                microbatch_size=request.post_block_refit_microbatch_size,
                learning_rate=request.post_block_refit_learning_rate,
            ):
                with _profile_block_phase(recorder, block_index, "refit"):
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
                            microbatch_size=request.post_block_refit_microbatch_size,
                            epoch_observer=_epoch_cooldown_observer(
                                request.post_block_refit_epoch_cooldown_seconds,
                                events,
                                tuning_kind="post_block_refit",
                                block=block_index,
                                target_weighted_mean_square=block_target_weighted_mean_square,
                            ),
                        ),
                        tuning_forward,
                        tuning_recorder,
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
                    bias_storage_dtype=bias_storage_dtype,
                    patch_storage_dtype=patch_storage_dtype,
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
            refitted_group_states: list[FrozenSharedInputGroupState] = []
            refitted_group_results: list[SharedInputGroupResult] = []
            for group_result in group_results:
                refitted_group = SharedInputGroupFreezer().freeze(
                    tuple(member.layer for member in group_result.frozen_state.members),
                    group_result.name,
                    tuple(member.row_end - member.row_start for member in group_result.frozen_state.members),
                    trainable_groups[group_result.name],
                    tensors,
                    backend="factorized",
                    bias_storage_dtype=bias_storage_dtype,
                )
                BlockEditor().install_frozen_group(working_block, refitted_group)
                with (
                    tensors.read(quantization_targets[group_result.name], request.device) as source_value,
                    tensors.read(group_result.plan.objectives[0].input_importance, request.device) as input_importance,
                ):
                    output_importances = []
                    for objective in group_result.plan.objectives:
                        with tensors.read(objective.output_importance, request.device) as value:
                            output_importances.append(value.clone())
                    merged_output = torch.cat(output_importances)
                    dense_group = refitted_group.owner.dense_weight()
                    metrics = reconstruction_metrics(source_value, dense_group, input_importance, merged_output)
                member_metrics = []
                for member_slice, member_source, objective in zip(
                    refitted_group.state.members,
                    group_member_targets[group_result.name],
                    group_result.plan.objectives,
                    strict=True,
                ):
                    with (
                        tensors.read(member_source, request.device) as member_value,
                        tensors.read(objective.input_importance, request.device) as member_input,
                        tensors.read(objective.output_importance, request.device) as member_output,
                    ):
                        member_metrics.append(
                            (
                                member_slice.layer,
                                reconstruction_metrics(
                                    member_value,
                                    dense_group[member_slice.row_start : member_slice.row_end],
                                    member_input,
                                    member_output,
                                ),
                            )
                        )
                refitted_group_states.append(refitted_group.state)
                refitted_group_results.append(
                    replace(
                        group_result,
                        frozen_state=refitted_group.state,
                        final_reconstruction=metrics,
                        member_reconstruction=tuple(member_metrics),
                    )
                )
            frozen_group_states = refitted_group_states
            group_results = refitted_group_results
            loss_recorder.record_post_block_refit(
                _block_loss(
                    adapter,
                    working_block,
                    compressed_inputs,
                    teacher_outputs,
                    block_output_importance,
                    metadata,
                    request.block_forward_batch_size,
                    micro_recorder,
                )
            )
        with _logged_operation(
            events,
            "block_propagation",
            block=block_index,
            samples=int(compressed_inputs.shape[0]),
            batch_size=request.block_forward_batch_size,
            destination="cpu",
        ):
            with _profile_block_phase(recorder, block_index, "propagate"):
                with torch.no_grad():
                    compressed_outputs = _run_block_batched(
                        adapter,
                        working_block,
                        compressed_inputs,
                        metadata,
                        request.block_forward_batch_size,
                        "cpu",
                        recorder=micro_recorder,
                    ).detach()
        with _profile_block_phase(recorder, block_index, "finalize"):
            loss_recorder.record_final_frozen_pre_kd(
                _weighted_mse(compressed_outputs, teacher_outputs, block_output_importance)
            )
            auxiliary_parameters = freeze_block_auxiliary_parameters(working_block, tensors)
            frozen_block = FrozenBlockState(
                block_plan.block,
                tuple(frozen_states),
                (),
                auxiliary_parameters,
                tuple(frozen_group_states),
            )
            block_window.finish()
            block_peak = max(
                block_window.peak_allocated_bytes,
                block_window.peak_reserved_bytes,
            )
            block_peak_host = peak_process_memory_bytes()
            peak_device_bytes = max(peak_device_bytes, block_peak)
        with _logged_operation(
            events,
            "block_commit",
            block=block_index,
            layers=len(layer_results),
            retention=request.activation_retention,
        ):
            with _profile_block_phase(recorder, block_index, "commit"):
                required_disk_bytes, free_disk_bytes, reserved_disk_bytes = _ensure_block_commit_disk_capacity(
                    request,
                    teacher_outputs,
                    compressed_outputs,
                )
                if request.memory_plan is not None:
                    cast(Any, events).emit(
                        "memory",
                        "info",
                        "memory.disk_commit_admitted",
                        block=block_index,
                        plan_revision=request.memory_plan.revision,
                        required_additional_bytes=required_disk_bytes,
                        free_bytes=free_disk_bytes,
                        reserve_bytes=reserved_disk_bytes,
                        safe_capacity_bytes=max(0, free_disk_bytes - reserved_disk_bytes),
                    )
                committed = commit_block(
                    block_plan.block,
                    tuple(layer_results),
                    frozen_block,
                    loss_recorder.finalize(),
                    teacher_outputs,
                    compressed_outputs,
                    budget.retry_bits_spent,
                    artifacts,
                    identity,
                    wall_seconds=time.perf_counter() - block_started,
                    peak_gpu_bytes=block_peak,
                    peak_host_bytes=block_peak_host,
                    warnings=()
                    if request.factorized_tuning_epochs > 0 and request.scale_fit.enabled
                    else (
                        ("tuning_disabled",) if request.scale_fit.enabled else ("scale_fit_disabled", "tuning_disabled")
                    ),
                    shared_input_groups=tuple(group_results),
                )
                block_journal_record = journal.append(
                    "block", block_index, None, committed.reference.artifact_id, identity
                )
                if request.activation_retention == "rolling" and committed_blocks:
                    retire_block_activations(committed_blocks[-1][1], artifacts)
        committed_blocks.append((committed.reference, committed.result))
        live_blocks[block_index] = committed.result
        for final_layer in committed.result.layers:
            live_layers[(block_index, final_layer.layer.path)] = final_layer
        for final_group in committed.result.shared_input_groups:
            live_groups[(block_index, final_group.name)] = final_group
        update_live_weight_error_report(
            request.output,
            tuple(live_layers.values()),
            tuple(live_blocks.values()),
            groups=tuple(live_groups.values()),
            expected_blocks=len(plan.blocks),
            layer_order=live_layer_order,
        )
        cast(Any, events).emit(
            "resident-quantization",
            "info",
            "block.completed",
            block=block_index,
            artifact_id=committed.reference.artifact_id,
            journal_sequence=block_journal_record.sequence,
            entry_loss=committed.result.losses.block_entry_pre_quantization,
            final_loss=committed.result.losses.final_frozen_pre_kd,
            target_weighted_mean_square=committed.result.losses.target_weighted_mean_square,
            entry_normalized_error=committed.result.losses.block_entry_normalized_error,
            final_normalized_error=committed.result.losses.final_frozen_normalized_error,
            wall_seconds=committed.result.wall_seconds,
            gpu_peak_bytes=committed.result.peak_gpu_bytes,
            host_peak_bytes=committed.result.peak_host_bytes,
            planned_device_bytes=(
                None if request.memory_plan is None else request.memory_plan.peak_gpu_bytes
            ),
            budget_utilization=(
                None
                if request.memory_plan is None or request.memory_plan.peak_gpu_bytes <= 0
                else block_peak / request.memory_plan.peak_gpu_bytes
            ),
            memory_plan_revision=(None if request.memory_plan is None else request.memory_plan.revision),
            **{
                "cuda.window_peak_allocated_bytes": block_window.peak_allocated_bytes,
                "cuda.window_peak_reserved_bytes": block_window.peak_reserved_bytes,
            },
        )
        new_block_commits += 1
        if request.device.startswith("cuda"):
            released = release_cached_host_memory()
            cast(Any, events).emit(
                "resource",
                "info",
                "host_pinned_cache.released",
                block=block_index,
                supported=released,
            )
        if (
            request.interrupt_after_block_commits is not None
            and new_block_commits >= request.interrupt_after_block_commits
        ):
            raise InterruptedError(f"injected interruption after {new_block_commits} new block commits")
        has_pending_block = any(
            candidate.block.index not in completed_block_indexes and candidate.block.index > block_index
            for candidate in plan.blocks
        )
        if has_pending_block:
            # A continued run and a crash-resumed run must consume the same
            # canonical activation generation. Reloading the just-committed
            # boundary also makes the artifact round trip part of every
            # multi-block execution rather than a resume-only code path.
            with _logged_operation(
                events,
                "block_activation_reload",
                block=block_index,
                artifact_id=committed.reference.artifact_id,
                destination="cpu",
            ):
                del teacher_outputs, compressed_outputs
                teacher_inputs, compressed_inputs = load_block_activations(
                    committed.reference,
                    artifacts,
                    "cpu",
                )
        else:
            teacher_inputs = teacher_outputs
            compressed_inputs = compressed_outputs
        retained_on_device = _place_completed_decoder_block(
            layer_container,
            block_index,
            working_block,
            retain=request.restore_completed_blocks,
        )
        events.emit(
            "resident-quantization",
            "info",
            "block.device_residency_updated",
            block=block_index,
            retained=retained_on_device,
            reason=("inline_full_model_forward" if retained_on_device else "streaming_compression"),
        )
        del working_block
        if request.device.startswith("cuda") and not retained_on_device:
            torch.cuda.empty_cache()

    # The last activation boundary may be GPU-cached, but final quality and
    # model assembly do not consume it.  Drop every local alias before those
    # stages so a cache intended for tuning cannot inflate finalization peak.
    del teacher_inputs, compressed_inputs
    try:
        del teacher_outputs, compressed_outputs
    except UnboundLocalError:
        # A fully resumed run may skip the block loop entirely.
        pass
    if request.device.startswith("cuda"):
        torch.cuda.empty_cache()

    quality_metrics: tuple[float, float, float, float] | None = None
    with _logged_operation(
        events,
        "quality_evaluation",
        enabled=request.evaluate_inline_quality,
        samples=int(quality_tokens.shape[0]),
        input_elements=int(quality_tokens.numel()),
    ):
        with recorder.phase("finalize"):
            with recorder.phase("quality"):
                if request.evaluate_inline_quality:
                    if not request.restore_completed_blocks and completed_block_indexes:
                        raise ValueError("inline quality evaluation requires completed-block restoration")
                    if reference_logits is None:
                        raise AssertionError("inline quality evaluation requires captured reference logits")
                    quality_metrics = _streamed_quality_metrics(adapter, model, quality_tokens, reference_logits)
    original_elements = sum(
        layer.in_features * layer.out_features for block in inventory.blocks for layer in block.quantizable_layers
    )
    with _logged_operation(events, "model_assembly", blocks=len(committed_blocks)):
        with recorder.phase("finalize"):
            with recorder.phase("assemble"):
                frozen_model = assemble_frozen_model(
                    inventory.model,
                    persisted_plan.reference,
                    tuple(committed_blocks),
                    (),
                    original_elements,
                )
    update_live_weight_error_report(
        request.output,
        tuple(live_layers.values()),
        tuple(live_blocks.values()),
        groups=tuple(live_groups.values()),
        expected_blocks=len(plan.blocks),
        layer_order=live_layer_order,
        status="compression complete",
    )
    with recorder.phase("finalize"):
        with recorder.phase("metrics"):
            if quality_metrics is None:
                reference_nll = compressed_nll = logit_mse = argmax_agreement = float("nan")
            else:
                reference_nll, compressed_nll, logit_mse, argmax_agreement = quality_metrics
    events.emit(
        "resident-quantization",
        "info",
        "quality.metrics_computed",
        reference_nll=reference_nll,
        compressed_nll=compressed_nll,
        logit_mse=logit_mse,
        argmax_agreement=argmax_agreement,
    )
    with recorder.phase("finalize"):
        with recorder.phase("report_prepare"):
            elapsed = time.perf_counter() - started
            peak_host_bytes = peak_process_memory_bytes()
            ranks = [
                *[layer.rank for block in plan.blocks for layer in block.layers],
                *[group.rank for block in plan.blocks for group in block.shared_input_groups],
            ]
            artifact_bytes_before_report = _artifact_bytes(artifacts.root)
            report_payload = {
                "schema_version": 1,
                "source": request.source,
                "revision": request.revision,
                "model": to_dict(inventory.model),
                "plan": persisted_plan.reference.artifact_id,
                "block_count": len(committed_blocks),
                "layer_count": sum(
                    len(block.layers) + sum(len(group.plan.members) for group in block.shared_input_groups)
                    for _, block in committed_blocks
                ),
                "factor_owner_count": sum(
                    len(block.layers) + len(block.shared_input_groups) for _, block in committed_blocks
                ),
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
    with _logged_operation(
        events,
        "report_write",
        blocks=len(committed_blocks),
        layers=sum(
            len(block.layers) + sum(len(group.plan.members) for group in block.shared_input_groups)
            for _, block in committed_blocks
        ),
    ):
        with recorder.phase("finalize"):
            with recorder.phase("report"):
                with artifacts.begin_write("resident-quantization-report") as writer:
                    (writer.path / "report.json").write_text(
                        json.dumps(report_payload, sort_keys=True, indent=2), encoding="utf-8"
                    )
                    (writer.path / "reconstruction.md").write_text(
                        render_reconstruction_tables(tuple(block for _, block in committed_blocks)), encoding="utf-8"
                    )
                    descriptor = writer.commit()
    with recorder.phase("finalize"):
        with recorder.phase("resource_summary"):
            report = ArtifactRef("resident-quantization-report", descriptor.artifact_id, descriptor.schema_version)
            artifact_bytes = _artifact_bytes(artifacts.root)
            executor.release()
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


def _run_resident_factorization_slice_impl(
    request: ResidentQuantizationRequest,
    events: EventSink,
    run_id: str,
) -> ResidentFactorizationSliceResult:
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
    if any(block.shared_input_groups for block in plan.blocks):
        raise ValueError("factor-only slices do not yet support shared-input groups; use the block-resident workflow")
    config_hash = _resident_config_hash(request)
    identity = CommitIdentity(config_hash, inventory.model.config_hash, persisted_plan.reference.artifact_id)
    journal = ProgressJournal(request.output / "state", run_id, artifacts)
    discovery = journal.discover(plan, identity)
    records = (*discovery.valid_records, *discovery.orphan_records)
    complete_blocks = {record.block for record in records if record.kind == "block"}
    completed_results = [
        load_committed_block(ArtifactRef(ArtifactTypes.BLOCK_RESULT, record.artifact_id, 1), artifacts, identity).result
        for record in records
        if record.kind == "block"
    ]
    partial_results = [
        load_committed_layer(ArtifactRef(ArtifactTypes.LAYER_RESULT, record.artifact_id, 1), artifacts, identity).result
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
    context = StageContext(run_id, executor, artifacts, tensors, events, Cancellation())
    slice_window = PeakWindow(request.device).start()
    try:
        factor_stage = FactorizationAttemptStage(
            request.admm,
            device=request.device,
            record_admm_steps=request.observability.record_admm_steps,
            reset_peak_memory=False,
        )
        outlier_stage = OutlierSelectionStage(
            device=request.device,
            residual_probe_iterations=request.outliers.residual_probe.iterations,
            residual_probe_inner_iterations=request.admm.inner_iterations,
            transpose_wide=request.admm.transpose_wide,
        )
        scale_stage = ScaleFitStage(request.scale_fit, device=request.device)
        bias_stage = BiasCorrectionStage(request.bias_correction, device=request.device)
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
        bias_correction = _run_bias_correction(
            layer_plan,
            source_ref,
            factorized,
            outliers,
            request,
            context,
            bias_stage,
        )
        scales = factorized.factors.scales
        if scales.mid is None:
            raise AssertionError("factorizer omitted required mid scale")
        outlier_indices = outlier_values = outlier_scales = None
        bias = None
        if bias_correction is not None:
            with tensors.read(bias_correction.bias, request.device) as value:
                bias = value.clone()
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
                bias=bias,
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                outlier_scales=outlier_scales,
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
            bias_storage_dtype={
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[request.bias_correction.storage_dtype.value],
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
        accepted_attempt = next(index for index, attempt in enumerate(accepted.attempts) if attempt.accepted)
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
            bias_correction,
            None,
        )
        committed = commit_layer(layer_result, artifacts, identity)
        journal_record = journal.append(
            "layer", block_index, layer_plan.layer.path, committed.reference.artifact_id, identity
        )
        events.emit(
            "resident-factorization-slice",
            "info",
            "layer.committed",
            block=block_index,
            layer=layer_plan.layer.path,
            artifact_id=committed.reference.artifact_id,
            journal_sequence=journal_record.sequence,
            rank=layer_result.frozen_state.rank,
            accepted_attempt=layer_result.accepted_attempt,
            actual_bits=layer_result.actual_bit_cost.total,
            extra_retry_bits=layer_result.extra_retry_bits,
            weighted_error=layer_result.final_reconstruction.export_weighted_normalized_error,
            raw_error=layer_result.final_reconstruction.raw_normalized_error,
        )
        slice_window.finish()
        peak = slice_window.peak_allocated_bytes
        return ResidentFactorizationSliceResult(
            layer_result,
            identity,
            time.perf_counter() - started,
            peak,
            len(pending) - 1,
        )
    finally:
        slice_window.finish()
        executor.release()


def _run_resident_factorization_slice(request: ResidentQuantizationRequest) -> ResidentFactorizationSliceResult:
    proposed = _resident_manifest(request, "resident-factorization-slice")
    with open_run_session(
        request.output,
        manifest=proposed,
        observability=request.observability,
        registry_root=request.registry_root,
        console=True,
        maximum_wddm_shared_bytes=request.maximum_wddm_shared_bytes,
    ) as session:
        manifest = _start_resident_manifest(session.manifest, proposed)
        _write_resident_manifest(request.output, manifest)
        session.events.emit(
            "run",
            "info",
            "run.resumed" if session.resumed else "run.started",
            stored_config_hash=session.manifest.config_hash,
            current_config_hash=proposed.config_hash,
            previous_run_id=session.previous_run_id,
            component="resident-factorization-slice",
            device=request.device,
            calibration_method=request.calibration_method,
            calibration_samples=len(request.token_ids),
        )
        session_started = time.perf_counter()
        try:
            result = _run_resident_factorization_slice_impl(request, session.events, session.run_id)
        except (KeyboardInterrupt, InterruptedError) as exc:
            session.events.emit(
                "run",
                "warning",
                "run.interrupted",
                component="resident-factorization-slice",
                device=request.device,
                wall_seconds=time.perf_counter() - session_started,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            manifest = transition(manifest, RunStatus.INTERRUPTED)
            _write_resident_manifest(request.output, manifest)
            raise
        except BaseException as exc:
            session.events.emit(
                "run",
                "error",
                "run.failed",
                component="resident-factorization-slice",
                device=request.device,
                wall_seconds=time.perf_counter() - session_started,
                error_type=type(exc).__name__,
                error=str(exc),
                cause_type=(
                    None
                    if exc.__cause__ is None and exc.__context__ is None
                    else type(exc.__cause__ or exc.__context__).__name__
                ),
                cause=(
                    None if exc.__cause__ is None and exc.__context__ is None else str(exc.__cause__ or exc.__context__)
                ),
            )
            manifest = transition(
                manifest,
                RunStatus.FAILED,
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
            _write_resident_manifest(request.output, manifest)
            raise
        session.events.emit(
            "run",
            "warning",
            "run.interrupted",
            reason="factorization_slice_boundary",
            remaining_layers=result.remaining_layers,
        )
        manifest = transition(manifest, RunStatus.INTERRUPTED)
        _write_resident_manifest(request.output, manifest)
        return result


def run_resident_factorization_slice(request: ResidentQuantizationRequest) -> ResidentFactorizationSliceResult:
    if request.low_rank_patch.enabled:
        raise ValueError("resident factorization slices cannot fit activation-space low-rank patches")
    if request.device.startswith("cuda"):
        with acquire_device_lease(request.device), _legacy_cuda_numerics():
            return _run_resident_factorization_slice(request)
    return _run_resident_factorization_slice(request)


def run_resident_quantization(request: ResidentQuantizationRequest) -> ResidentQuantizationResult:
    """Run with an exclusive cross-process lease for CUDA resident state."""
    if request.device.startswith("cuda"):
        with acquire_device_lease(request.device), _legacy_cuda_numerics():
            if request.initial_cooldown_seconds:
                time.sleep(request.initial_cooldown_seconds)
            return _run_resident_quantization(request)
    return _run_resident_quantization(request)


def load_completed_resident_quantization(
    request: ResidentQuantizationRequest,
) -> ResidentQuantizationResult:
    """Rehydrate one completed resident result without reopening its run session."""

    _validate_resident_request(request)
    directory = RunDirectory(request.output.parent, request.output.name)
    manifest = from_dict(RunManifest, directory.read_manifest(), path="manifest")
    if manifest.status is not RunStatus.COMPLETED:
        raise ValueError(f"resident run is not completed: {manifest.status.value}")
    proposed = _resident_manifest(request, "resident-quantization")
    if manifest.config_hash != proposed.config_hash:
        raise ValueError("completed resident run configuration differs from the current request")

    artifacts = LocalArtifactStore(
        request.output / "artifacts",
        use_persistent_validation_cache=False,
    )
    _source, _checkpoint, inventory = _factor_slice_source_inventory(request)
    records: list[dict[str, Any]] = []
    journal_path = request.output / "state" / "journal.jsonl"
    try:
        lines = journal_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"could not read completed resident journal: {journal_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"completed resident journal line {line_number} is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"completed resident journal line {line_number} is not an object")
        records.append(payload)
    identity, block_records = latest_complete_identity(records, len(inventory.blocks))
    if identity.config_hash != _resident_config_hash(request):
        raise ValueError("completed resident commit identity differs from the current algorithm request")
    if identity.model_hash != inventory.model.config_hash:
        raise ValueError("completed resident commit identity belongs to a different model")

    plan_descriptor = artifacts.validate(identity.plan_hash)
    if plan_descriptor.artifact_type != ArtifactTypes.QUANTIZATION_PLAN:
        raise ValueError("completed resident plan reference is not a quantization plan")
    plan_payload = json.loads((artifacts.path_for(identity.plan_hash) / "plan.json").read_text(encoding="utf-8"))
    plan = from_dict(QuantizationPlan, plan_payload, path="plan")
    if plan.model != inventory.model or len(plan.blocks) != len(inventory.blocks):
        raise ValueError("completed resident plan does not match the current model inventory")
    plan_reference = ArtifactRef(
        ArtifactTypes.QUANTIZATION_PLAN,
        identity.plan_hash,
        plan_descriptor.schema_version,
    )

    committed = tuple(
        (
            ArtifactRef(ArtifactTypes.BLOCK_RESULT, str(block_records[index]["artifact_id"]), 1),
            load_committed_block(
                ArtifactRef(ArtifactTypes.BLOCK_RESULT, str(block_records[index]["artifact_id"]), 1),
                artifacts,
                identity,
            ).result,
        )
        for index in range(len(inventory.blocks))
    )
    original_elements = sum(
        layer.in_features * layer.out_features for block in inventory.blocks for layer in block.quantizable_layers
    )
    frozen_model = assemble_frozen_model(
        inventory.model,
        plan_reference,
        committed,
        (),
        original_elements,
    )

    report_references: list[ArtifactRef] = []
    for artifact_id in manifest.artifacts:
        descriptor = artifacts.validate(artifact_id)
        if descriptor.artifact_type == "resident-quantization-report":
            report_references.append(ArtifactRef(descriptor.artifact_type, artifact_id, descriptor.schema_version))
    if len(report_references) != 1:
        raise ValueError("completed resident manifest must reference exactly one quantization report")
    report_reference = report_references[0]
    report_root = artifacts.path_for(report_reference.artifact_id)
    report_payload = json.loads((report_root / "report.json").read_text(encoding="utf-8"))
    if (
        report_payload.get("source") != request.source
        or report_payload.get("revision") != request.revision
        or report_payload.get("model") != to_dict(inventory.model)
        or report_payload.get("plan") != identity.plan_hash
        or int(report_payload.get("block_count", -1)) != len(committed)
        or int(report_payload.get("actual_total_bits", -1)) != frozen_model.actual_total_bits
        or float(report_payload.get("effective_bpw", -1.0)) != frozen_model.effective_bpw
    ):
        raise ValueError("completed resident report does not match its validated commits")
    report_bytes = sum(path.stat().st_size for path in report_root.rglob("*") if path.is_file())
    artifact_bytes = int(report_payload["artifact_bytes_before_report"]) + report_bytes

    return ResidentQuantizationResult(
        inventory,
        plan,
        identity,
        frozen_model,
        tuple(block for _reference, block in committed),
        report_reference,
        float(report_payload["reference_nll"]),
        float(report_payload["compressed_nll"]),
        float(report_payload["logit_mse"]),
        float(report_payload["argmax_agreement"]),
        int(report_payload["peak_device_bytes"]),
        int(report_payload["peak_host_bytes"]),
        artifact_bytes,
        float(report_payload["elapsed_seconds"]),
        int(report_payload["reused_commit_count"]),
    )
