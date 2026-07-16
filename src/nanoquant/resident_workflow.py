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

from nanoquant.application.distillation import TopKDistillationConfig
from nanoquant.config.codec import from_dict
from nanoquant.config.schema import (
    ActivationStorageConfig,
    CalibrationFallbackConfig,
    CalibrationMethod,
    DistillationLoss,
    DType,
    ExecutorKind,
    ObjectiveConfig,
    ObjectiveKind,
    ResourceLimitsConfig,
    RunConfig,
    SourceStreamingConfig,
)
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.domain.models import ArtifactRef
from nanoquant.domain.runs import RunManifest, RunStatus
from nanoquant.global_distillation import (
    GlobalDistillationRequest,
    GlobalDistillationRunResult,
    run_global_topk_distillation,
)
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.runs import RunDirectory, transition
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    ResidentQuantizationResult,
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
    interrupt_after_block_commits: int | None = None
    interrupt_after_factorized_tuning_epoch_commits: int | None = None
    interrupt_after_distillation_epoch_commits: int | None = None
    restore_completed_blocks: bool = True
    defer_layer_loss_snapshots: bool = False
    replace_existing_global_tuning: bool = False
    maximum_wddm_shared_bytes: int | None = None


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
    _require(
        config.runtime.resources == ResourceLimitsConfig(),
        "runtime.resources",
        "explicit limits are not yet enforced",
    )
    _require(
        config.runtime.activations == ActivationStorageConfig(),
        "runtime.activations",
        "explicit activation-store selection is not yet mapped",
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
    _require(config.runtime.on_cuda_oom == ("fail",), "runtime.on_cuda_oom", "automatic OOM fallback is not yet mapped")
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
) -> ResidentQuantizationRequest:
    """Map one validated canonical recipe to the resident engine request."""

    _validate_supported_recipe(config)
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
    return ResidentQuantizationRequest(
        snapshot=inputs.snapshot,
        output=inputs.output,
        source=config.model.source,
        revision=str(config.model.revision),
        token_ids=inputs.token_ids,
        quality_token_ids=inputs.quality_token_ids if config.evaluation.inline_quality else None,
        device=config.runtime.compute_device,
        executor=(
            ExecutorKind.RESIDENT
            if config.runtime.executor is ExecutorKind.AUTO
            else config.runtime.executor
        ),
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
        layer_order=config.block_tuning.layer_order,
        admm=config.factorization.admm,
        outliers=config.outliers,
        scale_fit=config.factorization.scale_fit,
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
        tuning_microbatch_size=config.block_tuning.microbatch_size,
        legacy_tuning_seed_reset=config.block_tuning.reset_seed_each_stage,
        restore_best_tuning_state=config.block_tuning.restore_best_state,
        tuning_epoch_loss_mode=config.block_tuning.epoch_loss_mode.value,
        activation_retention=config.runtime.checkpoints.activation_retention.value,
        seed=config.reproducibility.seed,
        interrupt_after_layer_commits=options.interrupt_after_layer_commits,
        interrupt_after_block_commits=options.interrupt_after_block_commits,
        interrupt_after_factorized_tuning_epoch_commits=(options.interrupt_after_factorized_tuning_epoch_commits),
        block_forward_batch_size=config.runtime.block_forward_batch_size,
        calibration_method=config.calibration.method.value,
        calibration_shrinkage=config.calibration.shrinkage,
        calibration_batch_size=config.calibration.batch_size,
        precomputed_calibration=inputs.precomputed_calibration,
        precomputed_objectives=inputs.precomputed_objectives,
        precomputed_plan=inputs.precomputed_plan,
        restore_completed_blocks=options.restore_completed_blocks,
        evaluate_inline_quality=config.evaluation.inline_quality and options.restore_completed_blocks,
        defer_layer_loss_snapshots=options.defer_layer_loss_snapshots,
        profiling=config.profiling,
        observability=config.observability,
        maximum_wddm_shared_bytes=options.maximum_wddm_shared_bytes,
        registry_root=inputs.registry_root,
        run_config=config,
        launcher_path=inputs.launcher_path,
        defer_run_completion=config.distillation.enabled,
    )


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

    quantization = run_resident_quantization(resident_request_from_config(config, inputs, options))
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
    """Resolve a zero-argument runfile's pinned model and prepared token artifact."""

    _validate_supported_recipe(config)
    launcher = Path(launcher_path).resolve()
    repository_root = launcher.parent.parent
    model_path = Path(config.model.source)
    if model_path.exists():
        snapshot = model_path.resolve()
    else:
        snapshot = Path(snapshot_download(repo_id=config.model.source, revision=str(config.model.revision))).resolve()
    prepared_root = config.dataset.prepared_root
    prepared_artifact = config.dataset.prepared_artifact
    if prepared_root is None or prepared_artifact is None:
        raise ValueError("resident experiments require a pinned prepared dataset artifact")
    calibration_root = Path(prepared_root)
    if not calibration_root.is_absolute():
        calibration_root = repository_root / calibration_root
    calibration = load_pinned_calibration(
        calibration_root,
        ArtifactRef("calibration-dataset-manifest", prepared_artifact, 1),
    )
    if config.calibration.sample_count > calibration.input_ids.shape[0]:
        raise ValueError("calibration sample count is outside the prepared dataset")
    tokens = calibration.input_ids[: config.calibration.sample_count]
    quality_tokens = None
    if config.evaluation.inline_quality:
        quality_samples = config.evaluation.inline_quality_samples
        quality_length = config.evaluation.inline_quality_tokens
        if quality_samples > tokens.shape[0] or quality_length > tokens.shape[1]:
            raise ValueError("inline quality selection is outside the prepared calibration tokens")
        quality_tokens = tokens[:quality_samples, :quality_length]
    pad_token_id = None
    if config.distillation.enabled:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        pad_token_id = tokenizer.pad_token_id
    registry_root = Path(config.output.run_root)
    if not registry_root.is_absolute():
        registry_root = repository_root / registry_root
    return ResolvedResidentInputs(
        snapshot=snapshot,
        output=registry_root / config.intent.name,
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
    "resident_request_from_config",
    "resolve_resident_experiment_inputs",
    "run_resident_experiment",
]
