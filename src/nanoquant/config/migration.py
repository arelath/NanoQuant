"""Total migration from the frozen legacy flat configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .codec import ConfigDecodeError, apply_overrides
from .schema import ModelConfig, RunConfig

Transform = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class MigrationEntry:
    legacy_field: str
    destination: str | None
    disposition: str
    note: str = ""


def _identity(value: Any) -> Any:
    return value


MAPPINGS: dict[str, tuple[str, Transform]] = {
    "model_id": ("model.source", _identity),
    "bits": ("allocation.target_bpw", _identity),
    "seed": ("reproducibility.seed", _identity),
    "num_calib_samples": ("calibration.sample_count", _identity),
    "calib_shrinkage": ("calibration.shrinkage", _identity),
    "calib_strategy": (
        "calibration.method",
        lambda value: {"online": "online_fisher", "two_phase": "two_phase_fisher", "forward": "forward_only"}.get(
            value, value
        ),
    ),
    "calib_oom_fallback": (
        "calibration.fallback.on_cuda_oom",
        lambda value: [value, "fail"] if value != "fail" else ["fail"],
    ),
    "hessian_max_tokens": ("calibration.objective.sampling.max_tokens_per_layer", _identity),
    "hessian_max_sequences": (
        "calibration.objective.sampling.max_sequences",
        lambda value: None if value == 0 else value,
    ),
    "hessian_batch_size": ("calibration.objective.sampling.batch_size", _identity),
    "hessian_reuse_siblings": ("calibration.objective.sampling.reuse_sibling_inputs", _identity),
    "hessian_damp_percent": ("calibration.objective.regularization.diagonal_damp_fraction", _identity),
    "hessian_shrinkage": ("calibration.objective.regularization.identity_shrinkage", _identity),
    "hessian_diagonal_blend": ("calibration.objective.regularization.diagonal_blend", _identity),
    "seqlen": ("model.sequence_length", _identity),
    "block_forward_batch_size": ("runtime.block_forward_batch_size", lambda value: 8 if value == 0 else value),
    "pin_cpu_activation_max_gib": ("runtime.resources.pinned_memory_gib", _identity),
    "loss_pct_floor": ("observability.loss_denominator_floor", _identity),
    "quant_layer_order": ("block_tuning.layer_order", lambda value: [] if value == "default" else value.split(",")),
    "rank_allocation_strategy": ("allocation.strategy", _identity),
    "rank_sensitivity_alpha": ("allocation.sensitivity_alpha", _identity),
    "rank_edge_boost": ("allocation.bounds.edge_block_boost", _identity),
    "rank_floor_frac": ("allocation.bounds.floor_fraction_of_uniform", _identity),
    "rank_ceil_frac": ("allocation.bounds.ceiling_fraction_of_uniform", _identity),
    "rank_retry_norm_error_threshold": (
        "allocation.retry.thresholds.weighted_normalized_error",
        lambda value: None if value == 0 else value,
    ),
    "rank_retry_raw_norm_error_threshold": (
        "allocation.retry.thresholds.raw_normalized_error",
        lambda value: None if value == 0 else value,
    ),
    "rank_retry_allow_above_cap": ("allocation.retry.allow_above_allocator_cap", _identity),
    "rank_retry_bump_frac": ("allocation.retry.rank_increase_fraction", _identity),
    # Legacy names this value "max attempts" but treats it as retries *after*
    # the initial solve. The canonical policy counts every attempted solve.
    "rank_retry_max_attempts": ("allocation.retry.maximum_attempts", lambda value: int(value) + 1),
    "rank_retry_bits_budget_frac": ("allocation.retry.extra_bit_budget_fraction", _identity),
    "rank_utility_profile_path": ("allocation.utility_profile_artifact", _identity),
    "outlier_frac": ("outliers.fraction", _identity),
    "outlier_dtype": (
        "outliers.storage_dtype",
        lambda value: {"bf16": "bfloat16", "fp16": "float16"}.get(value, value),
    ),
    "outlier_layers": ("outliers.layer_patterns", lambda value: ["*"] if value in {"all", "*"} else value.split(",")),
    "outlier_metric": ("outliers.selector", _identity),
    "outlier_budget_compensate": ("outliers.charge_to_bit_budget", _identity),
    "outlier_count_multiple": ("outliers.count_multiple", _identity),
    "outlier_i_norm_mode": ("outliers.removed_column_importance", _identity),
    "outlier_residual_probe_iters": ("outliers.residual_probe.iterations", _identity),
    "outlier_residual_chunk_rows": ("outliers.residual_probe.chunk_rows", _identity),
    "embed_tokens_weight_bits": ("packing.embeddings.bits", _identity),
    "tune_nonfact": ("block_tuning.non_factorized.loop.enabled", _identity),
    "nonfact_lr": ("block_tuning.non_factorized.optimizer.learning_rate", _identity),
    "nonfact_batch_size": ("block_tuning.non_factorized.loop.batch_size", _identity),
    "nonfact_epochs": ("block_tuning.non_factorized.loop.epochs", _identity),
    "nonfact_early_stop_rel_tol": (
        "block_tuning.non_factorized.loop.early_stop_relative_tolerance",
        lambda value: None if value == 0 else value,
    ),
    "nonfact_epoch_schedule": (
        "block_tuning.non_factorized.epochs_by_layer_position",
        lambda value: [] if not value else [int(item) for item in value.split(",")],
    ),
    "admm_type": ("factorization.implementation", lambda value: {"nanoquant": "nanoquant_admm"}.get(value, value)),
    "admm_outer_iters": ("factorization.admm.outer_iterations", _identity),
    "admm_inner_iters": ("factorization.admm.inner_iterations", _identity),
    "admm_reg": ("factorization.admm.regularization", _identity),
    "admm_penalty_scheduler": ("factorization.admm.penalty_schedule", _identity),
    "admm_print_steps": ("observability.record_admm_steps", _identity),
    "ls_scale_fit": ("factorization.scale_fit.enabled", _identity),
    "ls_scale_fit_iters": ("factorization.scale_fit.alternating_passes", _identity),
    "ls_scale_fit_eps": ("factorization.scale_fit.epsilon", _identity),
    "ls_scale_fit_chunk_rows": ("factorization.scale_fit.chunk_rows", _identity),
    "tune_fact": ("block_tuning.factorized.loop.enabled", _identity),
    "fact_binary_lr": ("block_tuning.factorized.learning_rates.binary", _identity),
    "fact_scale_lr": ("block_tuning.factorized.learning_rates.scale", _identity),
    "fact_outlier_lr": ("block_tuning.factorized.learning_rates.outlier", _identity),
    "fact_bias_lr": ("block_tuning.factorized.learning_rates.bias", _identity),
    "fact_batch_size": ("block_tuning.factorized.loop.batch_size", _identity),
    "fact_epochs": ("block_tuning.factorized.loop.epochs", _identity),
    "fact_early_stop_rel_tol": (
        "block_tuning.factorized.loop.early_stop_relative_tolerance",
        lambda value: None if value == 0 else value,
    ),
    "fact_skip_jump_frac": (
        "block_tuning.factorized.skip_if_relative_loss_jump_below",
        lambda value: None if value == 0 else value,
    ),
    "post_block_scale_epochs": ("block_tuning.post_block_refit.epochs", _identity),
    "post_block_scale_lr": ("block_tuning.post_block_refit.scale_learning_rate", _identity),
    "post_block_outlier_lr": ("block_tuning.post_block_refit.outlier_learning_rate", _identity),
    "post_block_bias_lr": ("block_tuning.post_block_refit.bias_learning_rate", _identity),
    "post_block_scale_batch_size": (
        "block_tuning.post_block_refit.batch_size",
        lambda value: None if value == 0 else value,
    ),
    "tune_model": ("distillation.enabled", _identity),
    "model_kd_lr": ("distillation.learning_rate", _identity),
    "model_kd_batch_size": ("distillation.batch_size", _identity),
    "model_kd_epochs": ("distillation.epochs", _identity),
    "model_kd_gradient_checkpointing": ("distillation.gradient_checkpointing", _identity),
    "model_kd_loss": ("distillation.loss", lambda value: {"topk": "top_k", "full": "full_kl"}.get(value, value)),
    "model_kd_temperature": ("distillation.temperature", _identity),
    "model_kd_topk": ("distillation.top_k", _identity),
    "model_kd_vocab_chunk_size": ("distillation.vocabulary_chunk_size", _identity),
    "model_kd_token_chunk_size": ("distillation.token_chunk_size", _identity),
    "model_kd_max_tokens_per_batch": ("distillation.maximum_tokens_per_batch", _identity),
}

REMOVED: dict[str, str] = {
    "calib_dataset": "replaced by versioned dataset sources; migrate manually when mixtures are encoded in one string",
    "device_map": "replaced by executor/resource planning",
    "eval_block_ppl": "replaced by evaluator registry suites",
    "block_activation_device": "replaced by runtime.activations.kind and automatic tiering",
    "block_activation_gpu_cache": "replaced by activation tier planning",
    "block_activation_gpu_reserve_gib": "replaced by runtime.resources safety margins",
    "pin_cpu_activations": "replaced by runtime.activations.kind",
    "cleanup_per_layer": "allocator cleanup policy is executor-owned",
    "weight_error_log_path": "structured metrics are artifacts; report paths are output policy",
    "weight_error_table_path": "structured metrics are artifacts; report paths are output policy",
    "rank_utility_log_path": "utility results are content-addressed artifacts",
    "tune_eval_summaries": "structured tuning events are always available",
}


def migration_inventory() -> tuple[MigrationEntry, ...]:
    mapped = [MigrationEntry(name, destination, "mapped") for name, (destination, _) in MAPPINGS.items()]
    mapped.append(MigrationEntry("hessian_whitening", "calibration.objective.kind", "mapped"))
    removed = [MigrationEntry(name, None, "removed", note) for name, note in REMOVED.items()]
    return tuple(sorted((*mapped, *removed), key=lambda entry: entry.legacy_field))


def migrate_legacy(legacy: dict[str, Any]) -> tuple[RunConfig, tuple[MigrationEntry, ...]]:
    unknown = sorted(set(legacy) - set(MAPPINGS) - set(REMOVED) - {"hessian_whitening"})
    if unknown:
        raise ConfigDecodeError(f"legacy.{unknown[0]}", "legacy field has no migration disposition")
    source = legacy.get("model_id")
    if not isinstance(source, str) or not source:
        raise ConfigDecodeError("legacy.model_id", "required for migration")
    config = RunConfig(model=ModelConfig(source=source))
    overrides: dict[str, Any] = {}
    for name, value in legacy.items():
        if name in MAPPINGS and name != "model_id":
            destination, transform = MAPPINGS[name]
            overrides[destination] = transform(value)
    if "hessian_whitening" in legacy:
        overrides["calibration.objective.kind"] = "dense_hessian" if legacy["hessian_whitening"] else "diagonal"
    if legacy.get("post_block_scale_epochs", 0) > 0:
        overrides["block_tuning.post_block_refit.enabled"] = True
    return apply_overrides(config, overrides), migration_inventory()
