"""Canonical configuration composition for resident quantization experiments.

The resident engines intentionally keep concrete tensors and filesystem paths in
their request objects.  This module is the single boundary that resolves those
material inputs and maps the semantic :class:`RunConfig` into the resident and
global-distillation requests used by tools, numbered runfiles, and Python callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.application.distillation import DistillationMetrics, TopKDistillationConfig
from nanoquant.config.codec import config_hash, from_dict
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ActivationStorageConfig,
    CalibrationFallbackConfig,
    CalibrationMethod,
    DistillationLoss,
    DType,
    ExecutorKind,
    MemoryPolicyMode,
    ObjectiveConfig,
    ObjectiveKind,
    ResourceLimitsConfig,
    RunConfig,
    SourceStreamingConfig,
)
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.domain.models import ArtifactRef
from nanoquant.domain.resources import ResolvedMemoryPlan
from nanoquant.domain.runs import RunManifest, RunStatus
from nanoquant.global_distillation import (
    GlobalDistillationRequest,
    GlobalDistillationRunResult,
    run_global_topk_distillation,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_memory import is_cuda_oom
from nanoquant.infrastructure.environment import load_repository_dotenv
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.hf_calibration_dataset import load_or_prepare_calibration
from nanoquant.infrastructure.resource_planning import (
    build_resident_memory_plan,
    load_memory_plan,
    persist_memory_plan,
    revise_resident_memory_plan_after_oom,
)
from nanoquant.infrastructure.runs import RunDirectory, transition
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    ResidentQuantizationResult,
    load_completed_resident_quantization,
    run_resident_quantization,
)


@dataclass(frozen=True, slots=True)
class ResolvedResidentInputs:
    """Material inputs resolved outside the canonical semantic recipe."""

    snapshot: Path
    output: Path
    registry_root: Path
    token_ids: torch.Tensor | tuple[tuple[int, ...], ...]
    quality_token_ids: torch.Tensor | tuple[tuple[int, ...], ...] | None
    launcher_path: Path | None = None
    pad_token_id: int | None = None
    precomputed_calibration: ArtifactRef | None = None
    precomputed_objectives: ArtifactRef | None = None
    precomputed_plan: ArtifactRef | None = None


@dataclass(frozen=True, slots=True)
class ResidentExecutionOptions:
    """Non-semantic interruption and operator controls for a workflow invocation."""

    initial_cooldown_seconds: float = 0.0
    distillation_initial_cooldown_seconds: float = 0.0
    factorized_tuning_epoch_cooldown_seconds: float = 0.0
    nonfactorized_tuning_epoch_cooldown_seconds: float = 0.0
    post_block_refit_epoch_cooldown_seconds: float = 0.0
    distillation_epoch_cooldown_seconds: float = 0.0
    interrupt_after_layer_commits: int | None = None
    preprocessing_reuse_run: Path | None = None
    rank_probe_reuse_run: Path | None = None
    interrupt_after_block_commits: int | None = None
    interrupt_after_factorized_tuning_epoch_commits: int | None = None
    interrupt_after_distillation_epoch_commits: int | None = None
    restore_completed_blocks: bool = True
    defer_layer_loss_snapshots: bool = False
    replace_existing_global_tuning: bool = False
    maximum_wddm_shared_bytes: int | None = None
    replan_memory: bool = False


@dataclass(frozen=True, slots=True)
class ResidentWorkflowResult:
    quantization: ResidentQuantizationResult
    distillation: GlobalDistillationRunResult | None


_DEFAULT_EXECUTION_OPTIONS = ResidentExecutionOptions()


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise ValueError(f"resident workflow does not support {path}: {message}")


def _validate_supported_recipe(config: RunConfig) -> None:
    raise_for_issues(validate(config, ValidationPhase.RESOLVED))
    _require(config.intent.name != "unnamed-run", "intent.name", "a stable output name is required")
    _require(config.model.load_dtype is DType.BFLOAT16, "model.load_dtype", "resident model loading is pinned to BF16")
    _require(not config.model.trust_remote_code, "model.trust_remote_code", "remote model code is not allowed")
    _require(
        config.model.tokenizer_source in {None, config.model.source},
        "model.tokenizer_source",
        "a separate tokenizer source is not yet mapped",
    )
    _require(
        config.model.tokenizer_revision == config.model.revision,
        "model.tokenizer_revision",
        "a separate tokenizer revision is not yet mapped",
    )
    _require(config.reproducibility.deterministic, "reproducibility.deterministic", "must be true")
    _require(
        not config.reproducibility.allow_nondeterministic_kernels,
        "reproducibility.allow_nondeterministic_kernels",
        "must be false",
    )
    _require(config.calibration.sample_count > 0, "calibration.sample_count", "must be positive")
    _require(
        config.calibration.method in {CalibrationMethod.ONLINE_FISHER, CalibrationMethod.FORWARD_ONLY},
        "calibration.method",
        "only online_fisher and forward_only are implemented by the resident engine",
    )
    _require(
        config.calibration.fallback == CalibrationFallbackConfig(on_cuda_oom=("fail",)),
        "calibration.fallback",
        "automatic calibration fallback is not yet composed",
    )
    _require(
        config.calibration.objective == ObjectiveConfig(kind=ObjectiveKind.DIAGONAL),
        "calibration.objective",
        "the resident engine currently builds the diagonal objective",
    )
    _require(
        config.calibration.accumulation_dtype is DType.FLOAT32,
        "calibration.accumulation_dtype",
        "resident calibration accumulation is FP32",
    )
    _require(
        config.allocation.utility_profile_artifact is None,
        "allocation.utility_profile_artifact",
        "external utility profiles are not yet mapped",
    )
    _require(
        config.factorization.implementation == "nanoquant_admm",
        "factorization.implementation",
        "only nanoquant_admm is implemented",
    )
    _require(
        config.factorization.compute_dtype is DType.BFLOAT16,
        "factorization.compute_dtype",
        "resident factor execution is BF16",
    )
    _require(
        config.factorization.solve_dtype is DType.FLOAT32,
        "factorization.solve_dtype",
        "resident ADMM solves are FP32",
    )
    nonfactorized = config.block_tuning.non_factorized
    _require(nonfactorized.optimizer.name == "adamw", "block_tuning.non_factorized.optimizer.name", "must be adamw")
    _require(
        nonfactorized.optimizer.weight_decay == 0.0,
        "block_tuning.non_factorized.optimizer.weight_decay",
        "weight decay is not exposed by the resident parity optimizer",
    )
    _require(
        not nonfactorized.epochs_by_layer_position or nonfactorized.loop.enabled,
        "block_tuning.non_factorized.epochs_by_layer_position",
        "a disabled loop cannot have a per-layer epoch schedule",
    )
    factorized = config.block_tuning.factorized
    _require(
        factorized.loop.early_stop_relative_tolerance is None,
        "block_tuning.factorized.loop.early_stop_relative_tolerance",
        "factorized early stopping is not exposed by the resident engine",
    )
    _require(
        factorized.skip_if_relative_loss_jump_below is None,
        "block_tuning.factorized.skip_if_relative_loss_jump_below",
        "conditional factorized tuning is not exposed by the resident engine",
    )
    factor_lrs = factorized.learning_rates
    _require(
        factor_lrs.binary == factor_lrs.scale == factor_lrs.bias and factor_lrs.outlier in {None, factor_lrs.scale},
        "block_tuning.factorized.learning_rates",
        "the resident parity optimizer currently uses one learning rate",
    )
    refit = config.block_tuning.post_block_refit
    refit_lr = factor_lrs.scale if refit.scale_learning_rate is None else refit.scale_learning_rate
    _require(
        refit.outlier_learning_rate in {None, refit_lr} and refit.bias_learning_rate in {None, refit_lr},
        "block_tuning.post_block_refit",
        "the resident refit currently uses one learning rate",
    )
    _require(
        config.runtime.executor in {ExecutorKind.AUTO, ExecutorKind.RESIDENT, ExecutorKind.CPU_OFFLOAD},
        "runtime.executor",
        "this composition root supports resident and cpu_offload execution",
    )
    if config.runtime.executor is ExecutorKind.CPU_OFFLOAD:
        _require(
            not config.evaluation.inline_quality,
            "evaluation.inline_quality",
            "cpu_offload requires inline quality to be disabled",
        )
        _require(
            not config.distillation.enabled,
            "distillation.enabled",
            "cpu_offload requires model-level distillation to be disabled until teacher streaming is implemented",
        )
    expected_activations = ActivationStorageConfig(
        gpu_cache=config.runtime.activations.gpu_cache,
        gpu_reserve_gib=config.runtime.activations.gpu_reserve_gib,
    )
    _require(
        config.runtime.activations == expected_activations,
        "runtime.activations",
        "only the activation GPU cache and its reserve are currently configurable",
    )
    expected_streaming = SourceStreamingConfig(
        verify_tensor_hashes=config.runtime.source_streaming.verify_tensor_hashes
    )
    _require(
        config.runtime.source_streaming == expected_streaming,
        "runtime.source_streaming",
        "only source hash verification is currently configurable",
    )
    checkpoints = config.runtime.checkpoints
    _require(checkpoints.enabled, "runtime.checkpoints.enabled", "resident commits are mandatory")
    _require(checkpoints.commit_granularity == "layer", "runtime.checkpoints.commit_granularity", "must be layer")
    _require(not checkpoints.keep_attempt_artifacts, "runtime.checkpoints.keep_attempt_artifacts", "must be false")
    _require(checkpoints.verify_on_resume, "runtime.checkpoints.verify_on_resume", "must be true")
    supported_oom_actions = {
        "reduce_batch_size",
        "reduce_stage_batch_size",
        "move_activations_down_one_tier",
        "move_activation_store_to_pageable_ram",
        "fail",
    }
    _require(
        bool(config.runtime.on_cuda_oom)
        and config.runtime.on_cuda_oom[-1] == "fail"
        and set(config.runtime.on_cuda_oom) <= supported_oom_actions,
        "runtime.on_cuda_oom",
        "actions must be supported, finite, and terminate with fail",
    )
    if config.runtime.memory_policy.mode is not MemoryPolicyMode.ADAPTIVE:
        _require(
            config.runtime.on_cuda_oom == ("fail",),
            "runtime.on_cuda_oom",
            "automatic OOM recovery requires runtime.memory_policy.mode=adaptive",
        )
    _require(config.output.artifact_root == "artifacts", "output.artifact_root", "resident artifacts are run-local")
    if config.distillation.enabled:
        _require(config.distillation.loss is DistillationLoss.TOP_K, "distillation.loss", "only top_k is implemented")
        _require(
            config.distillation.teacher_targets_artifact is None,
            "distillation.teacher_targets_artifact",
            "external teacher targets are not yet mapped",
        )


def resident_request_from_config(
    config: RunConfig,
    inputs: ResolvedResidentInputs,
    options: ResidentExecutionOptions = _DEFAULT_EXECUTION_OPTIONS,
    *,
    memory_plan: ResolvedMemoryPlan | None = None,
    memory_plan_reference: ArtifactRef | None = None,
) -> ResidentQuantizationRequest:
    """Map one validated canonical recipe to the resident engine request."""

    _validate_supported_recipe(config)
    if config.runtime.executor is ExecutorKind.CPU_OFFLOAD and config.calibration.method in {
        CalibrationMethod.ONLINE_FISHER,
        CalibrationMethod.TWO_PHASE_FISHER,
    }:
        precomputed = (
            inputs.precomputed_calibration,
            inputs.precomputed_objectives,
            inputs.precomputed_plan,
        )
        _require(
            all(reference is not None for reference in precomputed),
            "calibration.method",
            "cpu_offload Fisher calibration requires complete precomputed calibration, objectives, and plan; "
            "use forward_only when preprocessing must run in-process",
        )
    token_count = len(inputs.token_ids)
    if token_count != config.calibration.sample_count:
        raise ValueError(
            "resolved calibration sample count does not match config: "
            f"{token_count} != {config.calibration.sample_count}"
        )
    if config.evaluation.inline_quality and inputs.quality_token_ids is None:
        raise ValueError("inline quality is enabled but no quality tokens were resolved")
    nonfactorized = config.block_tuning.non_factorized
    factorized = config.block_tuning.factorized
    refit = config.block_tuning.post_block_refit
    nonfactorized_schedule = nonfactorized.epochs_by_layer_position
    nonfactorized_epochs = 0 if nonfactorized_schedule else nonfactorized.loop.epochs
    factorized_lr = factorized.learning_rates.scale
    refit_lr = factorized_lr if refit.scale_learning_rate is None else refit.scale_learning_rate
    executor = (
        ExecutorKind(memory_plan.executor)
        if memory_plan is not None
        else (ExecutorKind.RESIDENT if config.runtime.executor is ExecutorKind.AUTO else config.runtime.executor)
    )
    if executor is ExecutorKind.CPU_OFFLOAD:
        _require(
            not config.evaluation.inline_quality and not config.distillation.enabled,
            "runtime.executor",
            "resolved cpu_offload execution cannot run inline quality or model-level distillation",
        )
    effective_restore_completed_blocks = options.restore_completed_blocks and executor is ExecutorKind.RESIDENT
    effective_forward_batch = (
        config.runtime.block_forward_batch_size
        if memory_plan is None
        else memory_plan.stage("block_forward").batch_size
    )
    effective_calibration_batch = (
        config.calibration.batch_size
        if memory_plan is None
        else memory_plan.stage("calibration").batch_size
    )
    effective_tuning_microbatch = (
        config.block_tuning.microbatch_size
        if memory_plan is None
        else memory_plan.stage("tuning").batch_size
    )
    effective_refit_microbatch = (
        config.block_tuning.microbatch_size
        if memory_plan is None
        else memory_plan.stage("post_block_refit").batch_size
    )
    effective_cache = (
        config.runtime.activations.gpu_cache
        if memory_plan is None
        else ActivationGpuCacheMode(memory_plan.activation_gpu_cache)
    )
    return ResidentQuantizationRequest(
        snapshot=inputs.snapshot,
        output=inputs.output,
        source=config.model.source,
        revision=str(config.model.revision),
        token_ids=inputs.token_ids,
        quality_token_ids=inputs.quality_token_ids if config.evaluation.inline_quality else None,
        device=config.runtime.compute_device,
        executor=executor,
        verify_hashes=config.runtime.source_streaming.verify_tensor_hashes,
        target_bpw=config.allocation.target_bpw,
        rank_multiple=config.allocation.bounds.multiple,
        allocation_strategy=config.allocation.strategy,
        rank_floor_fraction=config.allocation.bounds.floor_fraction_of_uniform,
        rank_ceiling_fraction=config.allocation.bounds.ceiling_fraction_of_uniform,
        rank_sensitivity_alpha=config.allocation.sensitivity_alpha,
        rank_edge_boost=config.allocation.bounds.edge_block_boost,
        maximum_rank_layer_patterns=config.allocation.maximum_rank_layer_patterns,
        layer_budget_multipliers=config.allocation.layer_budget_multipliers,
        rank_retry=config.allocation.retry,
        reconstruction_rank_planning=config.allocation.reconstruction,
        kl_profile_artifact=config.allocation.kl_profile_artifact,
        kl_profile_key=config.allocation.kl_profile_key,
        kl_sensitivity_granularity=config.allocation.kl_sensitivity_granularity,
        layer_order=config.block_tuning.layer_order,
        shared_input_groups=config.factorization.shared_input.groups,
        admm=config.factorization.admm,
        outliers=config.outliers,
        scale_fit=config.factorization.scale_fit,
        bias_correction=config.factorization.bias_correction,
        low_rank_patch=config.factorization.low_rank_patch,
        factorized_tuning_epochs=factorized.loop.epochs if factorized.loop.enabled else 0,
        factorized_tuning_batch_size=factorized.loop.batch_size,
        factorized_tuning_learning_rate=factorized_lr,
        factorized_tuning_epoch_cooldown_seconds=options.factorized_tuning_epoch_cooldown_seconds,
        initial_cooldown_seconds=options.initial_cooldown_seconds,
        nonfactorized_tuning_epochs=nonfactorized_epochs if nonfactorized.loop.enabled else 0,
        nonfactorized_tuning_epochs_by_layer=(nonfactorized_schedule if nonfactorized.loop.enabled else ()),
        nonfactorized_tuning_batch_size=nonfactorized.loop.batch_size,
        nonfactorized_tuning_learning_rate=nonfactorized.optimizer.learning_rate,
        nonfactorized_tuning_epoch_cooldown_seconds=options.nonfactorized_tuning_epoch_cooldown_seconds,
        nonfactorized_tuning_early_stop_relative_tolerance=(nonfactorized.loop.early_stop_relative_tolerance),
        post_block_refit_epochs=refit.epochs if refit.enabled else 0,
        post_block_refit_batch_size=refit.batch_size or factorized.loop.batch_size,
        post_block_refit_learning_rate=refit_lr,
        post_block_refit_epoch_cooldown_seconds=options.post_block_refit_epoch_cooldown_seconds,
        tuning_microbatch_size=effective_tuning_microbatch,
        post_block_refit_microbatch_size=effective_refit_microbatch,
        legacy_tuning_seed_reset=config.block_tuning.reset_seed_each_stage,
        restore_best_tuning_state=config.block_tuning.restore_best_state,
        tuning_epoch_loss_mode=config.block_tuning.epoch_loss_mode.value,
        activation_retention=config.runtime.checkpoints.activation_retention.value,
        activation_gpu_cache=effective_cache,
        activation_gpu_reserve_bytes=int(config.runtime.activations.gpu_reserve_gib * 2**30),
        seed=config.reproducibility.seed,
        interrupt_after_layer_commits=options.interrupt_after_layer_commits,
        preprocessing_reuse_run=options.preprocessing_reuse_run,
        rank_probe_reuse_run=options.rank_probe_reuse_run,
        interrupt_after_block_commits=options.interrupt_after_block_commits,
        interrupt_after_factorized_tuning_epoch_commits=(options.interrupt_after_factorized_tuning_epoch_commits),
        block_forward_batch_size=effective_forward_batch,
        calibration_method=config.calibration.method.value,
        calibration_shrinkage=config.calibration.shrinkage,
        calibration_batch_size=effective_calibration_batch,
        precomputed_calibration=inputs.precomputed_calibration,
        precomputed_objectives=inputs.precomputed_objectives,
        precomputed_plan=inputs.precomputed_plan,
        restore_completed_blocks=effective_restore_completed_blocks,
        evaluate_inline_quality=config.evaluation.inline_quality and effective_restore_completed_blocks,
        defer_layer_loss_snapshots=options.defer_layer_loss_snapshots,
        profiling=config.profiling,
        observability=config.observability,
        maximum_wddm_shared_bytes=options.maximum_wddm_shared_bytes,
        registry_root=inputs.registry_root,
        run_config=config,
        launcher_path=inputs.launcher_path,
        defer_run_completion=config.distillation.enabled,
        memory_plan=memory_plan,
        memory_plan_reference=memory_plan_reference,
    )


def _token_shape(value: torch.Tensor | tuple[tuple[int, ...], ...]) -> tuple[int, int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 2:
            raise ValueError("resident calibration tokens must be a rank-two tensor")
        return int(value.shape[0]), int(value.shape[1])
    if not value or not value[0] or any(len(row) != len(value[0]) for row in value):
        raise ValueError("resident calibration token rows must be non-empty and rectangular")
    return len(value), len(value[0])


def _memory_planning_requested(config: RunConfig) -> bool:
    return (
        config.runtime.memory_policy.mode is MemoryPolicyMode.ADAPTIVE
        or config.runtime.resources != ResourceLimitsConfig()
    )


def _resolve_workflow_memory_plan(
    config: RunConfig,
    inputs: ResolvedResidentInputs,
    options: ResidentExecutionOptions,
) -> tuple[ResolvedMemoryPlan | None, ArtifactRef | None]:
    if not _memory_planning_requested(config):
        return None, None
    request_hash = config_hash(config)
    if not options.replan_memory:
        loaded = load_memory_plan(inputs.output, request_hash)
        if loaded is not None:
            return loaded
    plan = build_resident_memory_plan(
        config,
        inputs.snapshot,
        inputs.output,
        _token_shape(inputs.token_ids),
        retain_completed_blocks=config.evaluation.inline_quality and options.restore_completed_blocks,
    )
    return plan, persist_memory_plan(plan, inputs.output)


def distillation_request_from_config(
    config: RunConfig,
    inputs: ResolvedResidentInputs,
    options: ResidentExecutionOptions = _DEFAULT_EXECUTION_OPTIONS,
) -> GlobalDistillationRequest:
    """Map the canonical model-level KD recipe to its durable execution request."""

    _validate_supported_recipe(config)
    if not config.distillation.enabled:
        raise ValueError("distillation is disabled in the canonical run config")
    if len(inputs.token_ids) != config.calibration.sample_count:
        raise ValueError("resolved distillation sample count does not match config")
    distillation = config.distillation
    return GlobalDistillationRequest(
        run_output=inputs.output,
        snapshot=inputs.snapshot,
        source=config.model.source,
        revision=str(config.model.revision),
        token_ids=inputs.token_ids,
        config=TopKDistillationConfig(
            epochs=distillation.epochs,
            batch_size=distillation.batch_size,
            learning_rate=distillation.learning_rate,
            temperature=distillation.temperature,
            top_k=distillation.top_k,
            vocabulary_chunk_size=distillation.vocabulary_chunk_size,
            token_chunk_size=distillation.token_chunk_size,
            maximum_tokens_per_batch=distillation.maximum_tokens_per_batch,
            gradient_checkpointing=distillation.gradient_checkpointing,
            weight_decay=distillation.weight_decay,
            seed=config.reproducibility.seed,
            optimizer_version=distillation.optimizer_version,
            sampling_version=distillation.sampling_version,
        ),
        device=config.runtime.compute_device,
        pad_token_id=inputs.pad_token_id,
        verify_hashes=config.runtime.source_streaming.verify_tensor_hashes,
        replace_existing_global_tuning=options.replace_existing_global_tuning,
        interrupt_after_epoch_commits=options.interrupt_after_distillation_epoch_commits,
        initial_cooldown_seconds=options.distillation_initial_cooldown_seconds,
        epoch_cooldown_seconds=options.distillation_epoch_cooldown_seconds,
        profiling=config.profiling,
        block_snapshot_samples=config.observability.block_snapshot_samples,
        block_snapshot_tokens=config.observability.block_snapshot_tokens,
        block_snapshot_denominator_floor=config.observability.loss_denominator_floor,
        maximum_wddm_shared_bytes=options.maximum_wddm_shared_bytes,
    )


def execute_resident_workflow(
    config: RunConfig,
    inputs: ResolvedResidentInputs,
    options: ResidentExecutionOptions = _DEFAULT_EXECUTION_OPTIONS,
) -> ResidentWorkflowResult:
    """Execute quantization and, when enabled, model-level KD in legacy order."""

    memory_plan, memory_plan_reference = _resolve_workflow_memory_plan(config, inputs, options)
    memory_retries = 0
    while True:
        try:
            quantization = run_resident_quantization(
                resident_request_from_config(
                    config,
                    inputs,
                    options,
                    memory_plan=memory_plan,
                    memory_plan_reference=memory_plan_reference,
                )
            )
            break
        except BaseException as exc:
            fallback_authorized = any(action != "fail" for action in config.runtime.on_cuda_oom)
            if (
                not is_cuda_oom(exc)
                or memory_plan is None
                or memory_plan.mode != "adaptive"
                or not fallback_authorized
                or memory_retries >= config.runtime.memory_policy.maximum_stage_retries
            ):
                raise
            memory_plan, _revision, memory_plan_reference = revise_resident_memory_plan_after_oom(
                memory_plan,
                config,
                inputs.output,
                stage=getattr(exc, "nanoquant_operation", None),
            )
            memory_retries += 1
    distillation = None
    if config.distillation.enabled:
        try:
            distillation = run_global_topk_distillation(distillation_request_from_config(config, inputs, options))
        except (KeyboardInterrupt, InterruptedError) as exc:
            _transition_workflow_manifest(
                inputs,
                RunStatus.INTERRUPTED,
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        except BaseException as exc:
            _transition_workflow_manifest(
                inputs,
                RunStatus.FAILED,
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        _transition_workflow_manifest(
            inputs,
            RunStatus.COMPLETED,
            artifact_id=distillation.reference.artifact_id,
        )
    return ResidentWorkflowResult(quantization, distillation)


def load_completed_resident_workflow(
    config: RunConfig,
    inputs: ResolvedResidentInputs,
    options: ResidentExecutionOptions = _DEFAULT_EXECUTION_OPTIONS,
) -> ResidentWorkflowResult | None:
    """Load a terminal completed workflow, or return ``None`` for an active/new run."""

    directory = RunDirectory(inputs.output.parent, inputs.output.name)
    manifest_path = inputs.output / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = from_dict(RunManifest, directory.read_manifest(), path="manifest")
    if manifest.status is not RunStatus.COMPLETED:
        return None
    memory_plan, memory_plan_reference = _resolve_workflow_memory_plan(config, inputs, options)
    quantization = load_completed_resident_quantization(
        resident_request_from_config(
            config,
            inputs,
            options,
            memory_plan=memory_plan,
            memory_plan_reference=memory_plan_reference,
        ),
        allow_historical_algorithm=True,
    )
    distillation = None
    if config.distillation.enabled:
        reference = active_global_tuning(inputs.output)
        if reference is None or reference.artifact_id not in manifest.artifacts:
            raise ValueError("completed resident workflow has no active global tuning result")
        artifacts = LocalArtifactStore(
            inputs.output / "artifacts",
            use_persistent_validation_cache=False,
        )
        committed = load_global_tuning(reference, artifacts)
        result = committed.result
        metrics = DistillationMetrics(
            result.epoch_losses,
            result.steps_completed,
            result.selected_parameter_count,
            result.teacher_cache_bytes,
        )
        distillation = GlobalDistillationRunResult(reference, result, metrics)
    return ResidentWorkflowResult(quantization, distillation)


def _transition_workflow_manifest(
    inputs: ResolvedResidentInputs,
    status: RunStatus,
    *,
    artifact_id: str | None = None,
    failure: dict[str, object] | None = None,
) -> None:
    directory = RunDirectory(inputs.output.parent, inputs.output.name)
    manifest = from_dict(RunManifest, directory.read_manifest(), path="manifest")
    artifacts = manifest.artifacts
    if artifact_id is not None and artifact_id not in artifacts:
        artifacts = (*artifacts, artifact_id)
    directory.write_manifest(transition(manifest, status, artifacts=artifacts, failure=failure))


def resolve_resident_experiment_inputs(config: RunConfig, *, launcher_path: str | Path) -> ResolvedResidentInputs:
    """Resolve a zero-argument runfile's model and run-local calibration tokens."""

    _validate_supported_recipe(config)
    launcher = Path(launcher_path).resolve()
    repository_root = launcher.parent.parent
    load_repository_dotenv(repository_root)
    model_path = Path(config.model.source)
    if model_path.exists():
        snapshot = model_path.resolve()
    else:
        snapshot = Path(snapshot_download(repo_id=config.model.source, revision=str(config.model.revision))).resolve()
    registry_root = Path(config.output.run_root)
    if not registry_root.is_absolute():
        registry_root = repository_root / registry_root
    output = registry_root / config.intent.name
    calibration = load_or_prepare_calibration(
        snapshot,
        output,
        sample_count=config.calibration.sample_count,
        sequence_length=config.model.sequence_length,
        seed=config.dataset.selection_seed,
        preparation_id=config_hash(config),
    )
    tokens = calibration.input_ids
    quality_tokens = None
    if config.evaluation.inline_quality:
        quality_samples = config.evaluation.inline_quality_samples
        quality_length = config.evaluation.inline_quality_tokens
        if quality_samples > tokens.shape[0] or quality_length > tokens.shape[1]:
            raise ValueError("inline quality selection is outside the generated calibration tokens")
        quality_tokens = tokens[:quality_samples, :quality_length]
    pad_token_id = None
    if config.distillation.enabled:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=False)
        pad_token_id = tokenizer.pad_token_id
    return ResolvedResidentInputs(
        snapshot=snapshot,
        output=output,
        registry_root=registry_root,
        token_ids=tokens,
        quality_token_ids=quality_tokens,
        launcher_path=launcher,
        pad_token_id=pad_token_id,
    )


def run_resident_experiment(config: RunConfig, *, launcher_path: str | Path) -> int:
    """Zero-argument numbered-runfile adapter over the shared resident workflow."""

    inputs = resolve_resident_experiment_inputs(config, launcher_path=launcher_path)
    execute_resident_workflow(config, inputs)
    return 0


__all__ = [
    "ResolvedResidentInputs",
    "ResidentExecutionOptions",
    "ResidentWorkflowResult",
    "distillation_request_from_config",
    "execute_resident_workflow",
    "load_completed_resident_workflow",
    "resident_request_from_config",
    "resolve_resident_experiment_inputs",
    "run_resident_experiment",
]
