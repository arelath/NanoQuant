"""Stable, phased configuration validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from nanoquant.ports.event_sink import Severity

from .schema import (
    MEASURED_UNIT_KL_OBJECTIVES,
    AllocationStrategy,
    DType,
    KlSensitivityGranularity,
    ObjectiveKind,
    OutlierSelector,
    RankResponseSource,
    RunConfig,
)


class ValidationPhase(str, Enum):
    PRE_RESOLUTION = "pre_resolution"
    RESOLVED = "resolved"
    PLANNED = "planned"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    severity: str = "error"


def validate(config: RunConfig, phase: ValidationPhase = ValidationPhase.PRE_RESOLUTION) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []

    def require(condition: bool, code: str, path: str, message: str) -> None:
        if not condition:
            issues.append(ValidationIssue(code, path, message))

    require(config.schema_version == 1, "CFG001", "schema_version", "only schema version 1 is supported")
    require(bool(config.model.source.strip()), "CFG002", "model.source", "model source must not be empty")
    require(config.model.sequence_length > 0, "CFG003", "model.sequence_length", "must be positive")
    require(config.calibration.sample_count >= 0, "CFG004", "calibration.sample_count", "must not be negative")
    require(config.calibration.batch_size > 0, "CFG017", "calibration.batch_size", "must be positive")
    require(0 <= config.calibration.shrinkage <= 1, "CFG005", "calibration.shrinkage", "must be in [0, 1]")
    behavior_slices = config.dataset.behavior_slices
    behavior_names = tuple(item.name for item in behavior_slices)
    require(
        all(bool(name.strip()) for name in behavior_names),
        "CFG088",
        "dataset.behavior_slices",
        "slice names must not be empty",
    )
    require(
        len(set(behavior_names)) == len(behavior_names),
        "CFG089",
        "dataset.behavior_slices",
        "slice names must be unique",
    )
    require(
        all(
            item.source.revision
            and math.isfinite(item.target_valid_token_fraction)
            and item.target_valid_token_fraction > 0
            for item in behavior_slices
        ),
        "CFG090",
        "dataset.behavior_slices",
        "every slice requires a pinned source and a positive finite token fraction",
    )
    require(
        not behavior_slices
        or math.isclose(
            sum(item.target_valid_token_fraction for item in behavior_slices),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "CFG091",
        "dataset.behavior_slices",
        "target valid-token fractions must sum to one",
    )
    require(
        all(item.record_format in {"raw_text", "ultrachat_messages", "openr1_generations"} for item in behavior_slices),
        "CFG092",
        "dataset.behavior_slices",
        "record format is unsupported",
    )
    trace_slices = tuple(item for item in behavior_slices if item.teacher_trace_generation is not None)
    require(
        all(
            item.mode.value in {"thinking", "non_thinking"}
            and item.record_format == "ultrachat_messages"
            for item in trace_slices
        ),
        "CFG101",
        "dataset.behavior_slices",
        "teacher outputs require a chat-mode UltraChat message slice",
    )
    require(
        all(
            bool(trace.implementation.strip())
            and trace.minimum_new_tokens > 0
            and trace.maximum_new_tokens >= trace.minimum_new_tokens
            and trace.maximum_attempt_multiplier > 0
            for item in trace_slices
            for trace in (item.teacher_trace_generation,)
            if trace is not None
        ),
        "CFG102",
        "dataset.behavior_slices",
        "teacher-trace generation limits and implementation must be valid",
    )
    require(
        not trace_slices or bool(config.model.revision),
        "CFG103",
        "model.revision",
        "teacher-trace generation requires a pinned teacher model revision",
    )
    require(
        all(item.partition in {"train", "quick", "final"} for item in behavior_slices),
        "CFG099",
        "dataset.behavior_slices",
        "partition must be train, quick, or final",
    )
    require(
        all(
            item.minimum_valid_tokens is None or item.minimum_valid_tokens > 0
            for item in behavior_slices
        ),
        "CFG100",
        "dataset.behavior_slices",
        "minimum valid-token floors must be positive when configured",
    )
    require(
        all(
            math.isfinite(item.assistant_target_weight)
            and item.assistant_target_weight >= 0
            and math.isfinite(item.prompt_target_weight)
            and item.prompt_target_weight >= 0
            for item in behavior_slices
        ),
        "CFG093",
        "dataset.behavior_slices",
        "distillation weights must be finite and non-negative",
    )
    reasoning_modes = config.evaluation.reasoning_modes
    require(
        len(set(reasoning_modes)) == len(reasoning_modes),
        "CFG094",
        "evaluation.reasoning_modes",
        "reasoning modes must be unique",
    )
    require(
        all(mode.value in {"thinking", "non_thinking"} for mode in reasoning_modes),
        "CFG095",
        "evaluation.reasoning_modes",
        "raw is not a chat reasoning evaluation mode",
    )
    require(
        not reasoning_modes or config.evaluation.reasoning_samples_per_mode > 0,
        "CFG096",
        "evaluation.reasoning_samples_per_mode",
        "must be positive",
    )
    require(
        not reasoning_modes
        or 2 <= config.evaluation.reasoning_sequence_length <= config.model.sequence_length,
        "CFG097",
        "evaluation.reasoning_sequence_length",
        "must be between two and the model sequence length",
    )
    require(
        not reasoning_modes
        or (
            math.isfinite(config.evaluation.maximum_thinking_degradation_ratio)
            and config.evaluation.maximum_thinking_degradation_ratio >= 1
        ),
        "CFG098",
        "evaluation.maximum_thinking_degradation_ratio",
        "must be finite and at least one",
    )
    require(config.allocation.target_bpw > 0, "CFG006", "allocation.target_bpw", "must be positive")
    require(config.allocation.bounds.multiple > 0, "CFG007", "allocation.bounds.multiple", "must be positive")
    kl_selected = config.allocation.strategy is AllocationStrategy.KL_CALIBRATED
    require(
        kl_selected == bool(config.allocation.kl_profile_artifact),
        "CFG076",
        "allocation.kl_profile_artifact",
        "must be set exactly when strategy is kl_calibrated",
    )
    require(
        kl_selected == bool(config.allocation.kl_profile_key),
        "CFG086",
        "allocation.kl_profile_key",
        "must be set exactly when strategy is kl_calibrated",
    )
    require(
        kl_selected
        or config.allocation.kl_sensitivity_granularity
        is KlSensitivityGranularity.EXACT_OR_TYPE_BLOCK,
        "CFG087",
        "allocation.kl_sensitivity_granularity",
        "a non-default KL sensitivity granularity requires strategy kl_calibrated",
    )
    maximum_rank_patterns = config.allocation.maximum_rank_layer_patterns
    require(
        all(bool(pattern.strip()) for pattern in maximum_rank_patterns),
        "CFG039",
        "allocation.maximum_rank_layer_patterns",
        "patterns must not be empty",
    )
    require(
        len(set(maximum_rank_patterns)) == len(maximum_rank_patterns),
        "CFG040",
        "allocation.maximum_rank_layer_patterns",
        "patterns must be unique",
    )
    budget_multipliers = config.allocation.layer_budget_multipliers
    budget_patterns = tuple(item.pattern for item in budget_multipliers)
    require(
        all(bool(pattern.strip()) for pattern in budget_patterns),
        "CFG041",
        "allocation.layer_budget_multipliers",
        "patterns must not be empty",
    )
    require(
        len(set(budget_patterns)) == len(budget_patterns),
        "CFG042",
        "allocation.layer_budget_multipliers",
        "patterns must be unique",
    )
    require(
        all(math.isfinite(item.multiplier) and item.multiplier > 1 for item in budget_multipliers),
        "CFG043",
        "allocation.layer_budget_multipliers",
        "multipliers must be finite and greater than one",
    )
    require(
        config.allocation.bounds.floor_fraction_of_uniform <= config.allocation.bounds.ceiling_fraction_of_uniform,
        "CFG008",
        "allocation.bounds",
        "floor must not exceed ceiling",
    )
    require(
        math.isfinite(config.allocation.bounds.overcomplete_rank_ceiling_fraction)
        and config.allocation.bounds.overcomplete_rank_ceiling_fraction >= 1,
        "CFG132",
        "allocation.bounds.overcomplete_rank_ceiling_fraction",
        "must be finite and at least one",
    )
    reconstruction = config.allocation.reconstruction
    reconstruction_selected = config.allocation.strategy in {
        AllocationStrategy.RECONSTRUCTION_AWARE,
        AllocationStrategy.KL_CALIBRATED,
    }
    require(
        reconstruction.enabled == reconstruction_selected,
        "CFG049",
        "allocation.reconstruction.enabled",
        "must be enabled exactly when strategy is reconstruction_aware or kl_calibrated",
    )
    require(
        0 <= reconstruction.sensitivity_strength <= 1,
        "CFG050",
        "allocation.reconstruction.sensitivity_strength",
        "must be in [0, 1]",
    )
    require(
        0 <= reconstruction.protected_sensitivity_quantile <= 1,
        "CFG051",
        "allocation.reconstruction.protected_sensitivity_quantile",
        "must be in [0, 1]",
    )
    require(
        math.isfinite(reconstruction.protected_rank_floor_fraction)
        and reconstruction.protected_rank_floor_fraction >= 1,
        "CFG052",
        "allocation.reconstruction.protected_rank_floor_fraction",
        "must be finite and at least one",
    )
    require(
        0 <= reconstruction.target_protected_error_reduction_fraction < 1,
        "CFG053",
        "allocation.reconstruction.target_protected_error_reduction_fraction",
        "must be in [0, 1)",
    )
    require(
        reconstruction.protect_sensitive_units
        or reconstruction.target_protected_error_reduction_fraction == 0,
        "CFG097",
        "allocation.reconstruction.target_protected_error_reduction_fraction",
        "must be zero when sensitive-unit protection is disabled",
    )
    require(
        math.isfinite(reconstruction.rank_trust_fraction)
        and 0 <= reconstruction.rank_trust_fraction <= 1,
        "CFG088",
        "allocation.reconstruction.rank_trust_fraction",
        "must be finite and in [0, 1]",
    )
    trust_reference = reconstruction.rank_trust_reference_run
    require(
        trust_reference is None or bool(trust_reference.strip()),
        "CFG089",
        "allocation.reconstruction.rank_trust_reference_run",
        "must be null or a non-empty run path",
    )
    require(
        (reconstruction.rank_trust_fraction == 1) == (trust_reference is None),
        "CFG090",
        "allocation.reconstruction.rank_trust_reference_run",
        "must be set exactly when rank_trust_fraction is below one",
    )
    require(
        kl_selected or trust_reference is None,
        "CFG091",
        "allocation.reconstruction.rank_trust_reference_run",
        "rank trust regions currently require strategy kl_calibrated",
    )
    importance = reconstruction.importance
    importance_patterns = tuple(item.pattern for item in importance.layer_multipliers)
    require(
        all(bool(pattern.strip()) for pattern in importance_patterns),
        "CFG065",
        "allocation.reconstruction.importance.layer_multipliers",
        "patterns must not be empty",
    )
    require(
        len(set(importance_patterns)) == len(importance_patterns),
        "CFG066",
        "allocation.reconstruction.importance.layer_multipliers",
        "patterns must be unique",
    )
    require(
        all(math.isfinite(item.multiplier) and item.multiplier >= 1 for item in importance.layer_multipliers),
        "CFG067",
        "allocation.reconstruction.importance.layer_multipliers",
        "multipliers must be finite and at least one",
    )
    protected_patterns = importance.protected_layer_patterns
    require(
        all(bool(pattern.strip()) for pattern in protected_patterns),
        "CFG068",
        "allocation.reconstruction.importance.protected_layer_patterns",
        "patterns must not be empty",
    )
    require(
        len(set(protected_patterns)) == len(protected_patterns),
        "CFG069",
        "allocation.reconstruction.importance.protected_layer_patterns",
        "patterns must be unique",
    )
    require(
        math.isfinite(importance.edge_block_multiplier) and importance.edge_block_multiplier >= 1,
        "CFG070",
        "allocation.reconstruction.importance.edge_block_multiplier",
        "must be finite and at least one",
    )
    require(
        importance.protected_edge_block_count >= 0,
        "CFG071",
        "allocation.reconstruction.importance.protected_edge_block_count",
        "must not be negative",
    )
    curve_patterns = tuple(curve.unit_pattern for curve in reconstruction.response_curves)
    require(
        len(curve_patterns) == len(set(curve_patterns)) and all(bool(pattern.strip()) for pattern in curve_patterns),
        "CFG054",
        "allocation.reconstruction.response_curves",
        "unit patterns must be non-empty and unique",
    )
    for index, curve in enumerate(reconstruction.response_curves):
        curve_path = f"allocation.reconstruction.response_curves[{index}]"
        require(
            0 < curve.calibrated_rank_floor_fraction <= 1 <= curve.calibrated_rank_ceiling_fraction,
            "CFG055",
            curve_path,
            "calibrated rank range must contain baseline fraction one",
        )
        boundaries = tuple(segment.maximum_rank_fraction for segment in curve.segments)
        require(
            bool(boundaries)
            and all(math.isfinite(value) for value in boundaries)
            and all(left < right for left, right in zip(boundaries, boundaries[1:], strict=False)),
            "CFG056",
            f"{curve_path}.segments",
            "segment boundaries must be finite and strictly increasing",
        )
        require(
            bool(boundaries)
            and boundaries[-1] == curve.calibrated_rank_ceiling_fraction
            and boundaries[0] > curve.calibrated_rank_floor_fraction,
            "CFG057",
            f"{curve_path}.segments",
            "segments must cover the complete calibrated rank range",
        )
        require(
            all(math.isfinite(segment.beta_per_rank) and segment.beta_per_rank > 0 for segment in curve.segments),
            "CFG058",
            f"{curve_path}.segments",
            "response slopes must be finite and positive",
        )
    if reconstruction_selected:
        require(
            reconstruction.objective_mode in {"unit_frobenius", "calibration_weighted"},
            "CFG059",
            "allocation.reconstruction.objective_mode",
            "must be unit_frobenius or calibration_weighted",
        )
        require(
            reconstruction.probe_admm is not None,
            "CFG060",
            "allocation.reconstruction.probe_admm",
            "an explicit full probe protocol is required",
        )
        if reconstruction.response_source is RankResponseSource.CONFIGURED:
            require(
                bool(reconstruction.response_curves),
                "CFG061",
                "allocation.reconstruction.response_curves",
                "configured response mode requires at least one response curve",
            )
            require(
                bool(reconstruction.response_profile_provenance.strip()),
                "CFG062",
                "allocation.reconstruction.response_profile_provenance",
                "configured response mode requires measured provenance",
            )
            for index, curve in enumerate(reconstruction.response_curves):
                require(
                    config.allocation.bounds.floor_fraction_of_uniform >= curve.calibrated_rank_floor_fraction
                    and config.allocation.bounds.ceiling_fraction_of_uniform <= curve.calibrated_rank_ceiling_fraction,
                    "CFG063",
                    f"allocation.reconstruction.response_curves[{index}]",
                    "allocation bounds must stay within the calibrated response range",
                )
        else:
            require(
                not reconstruction.response_curves,
                "CFG092",
                "allocation.reconstruction.response_curves",
                "measured response mode derives per-unit curves and forbids configured curves",
            )
            require(
                reconstruction.objective_mode == "calibration_weighted",
                "CFG093",
                "allocation.reconstruction.objective_mode",
                "measured response mode requires calibration_weighted probes",
            )
        if reconstruction.kl_objective in MEASURED_UNIT_KL_OBJECTIVES:
            require(
                kl_selected and config.allocation.kl_sensitivity_granularity is KlSensitivityGranularity.EXACT,
                "CFG094",
                "allocation.kl_sensitivity_granularity",
                "measured_unit_kl requires KL allocation with complete exact physical-unit arms",
            )
            require(
                reconstruction.response_source is RankResponseSource.MEASURED
                and reconstruction.sensitivity_strength == 1,
                "CFG095",
                "allocation.reconstruction",
                "measured_unit_kl requires same-run measured responses and untempered sensitivity",
            )
            require(
                reconstruction.rank_trust_reference_run is None
                and reconstruction.rank_trust_fraction == 1,
                "CFG096",
                "allocation.reconstruction.rank_trust_reference_run",
                "measured_unit_kl forbids rank values imported from another run",
            )
        if reconstruction.probe_admm is not None:
            require(
                reconstruction.probe_admm.outer_iterations > 0
                and reconstruction.probe_admm.inner_iterations > 0
                and reconstruction.probe_admm.convergence_check_interval > 0,
                "CFG064",
                "allocation.reconstruction.probe_admm",
                "probe iteration settings must be positive",
            )
    require(0 <= config.outliers.fraction < 1, "CFG009", "outliers.fraction", "must be in [0, 1)")
    require(
        not (config.outliers.selector is OutlierSelector.NONE and config.outliers.fraction > 0),
        "CFG010",
        "outliers",
        "positive fraction requires an enabled selector",
    )
    require(
        config.runtime.block_forward_batch_size > 0, "CFG011", "runtime.block_forward_batch_size", "must be positive"
    )
    require(
        math.isfinite(config.runtime.activations.gpu_reserve_gib) and config.runtime.activations.gpu_reserve_gib >= 0,
        "CFG044",
        "runtime.activations.gpu_reserve_gib",
        "must be finite and non-negative",
    )
    resource_limits = config.runtime.resources
    for path, resource_value in (
        ("gpu_memory_gib", resource_limits.gpu_memory_gib),
        ("cpu_memory_gib", resource_limits.cpu_memory_gib),
        ("temporary_disk_gib", resource_limits.temporary_disk_gib),
        ("workspace_memory_gib", resource_limits.workspace_memory_gib),
    ):
        require(
            resource_value is None or (math.isfinite(resource_value) and resource_value > 0),
            "CFG072",
            f"runtime.resources.{path}",
            "must be finite and positive when provided",
        )
    require(
        math.isfinite(resource_limits.pinned_memory_gib) and resource_limits.pinned_memory_gib >= 0,
        "CFG073",
        "runtime.resources.pinned_memory_gib",
        "must be finite and non-negative",
    )
    memory_policy = config.runtime.memory_policy
    for path, reserve_value in (
        ("gpu_reserve_gib", memory_policy.gpu_reserve_gib),
        ("host_reserve_gib", memory_policy.host_reserve_gib),
        ("temporary_disk_reserve_gib", memory_policy.temporary_disk_reserve_gib),
    ):
        require(
            math.isfinite(reserve_value) and reserve_value >= 0,
            "CFG074",
            f"runtime.memory_policy.{path}",
            "must be finite and non-negative",
        )
    require(
        memory_policy.maximum_stage_retries >= 0,
        "CFG075",
        "runtime.memory_policy.maximum_stage_retries",
        "must not be negative",
    )
    require(
        config.factorization.admm.outer_iterations > 0,
        "CFG012",
        "factorization.admm.outer_iterations",
        "must be positive",
    )
    shared = config.factorization.shared_input
    group_names = tuple(group.name for group in shared.groups)
    require(
        shared.enabled == bool(shared.groups),
        "CFG045",
        "factorization.shared_input",
        "enabled grouping requires at least one group and configured groups require enabled=true",
    )
    require(
        len(group_names) == len(set(group_names)) and all(bool(name.strip()) for name in group_names),
        "CFG046",
        "factorization.shared_input.groups",
        "group names must be non-empty and unique",
    )
    group_members = [member for group in shared.groups for member in group.members]
    require(
        all(len(group.members) >= 2 and len(group.members) == len(set(group.members)) for group in shared.groups),
        "CFG047",
        "factorization.shared_input.groups",
        "each group requires at least two unique members",
    )
    require(
        len(group_members) == len(set(group_members)) and all(bool(member.strip()) for member in group_members),
        "CFG048",
        "factorization.shared_input.groups",
        "member paths must be non-empty and may belong to only one group",
    )
    multiplier_entries = [entry for group in shared.groups for entry in group.member_multipliers]
    require(
        all(
            len({entry.member for entry in group.member_multipliers}) == len(group.member_multipliers)
            and all(entry.member in group.members for entry in group.member_multipliers)
            for group in shared.groups
        ),
        "CFG077",
        "factorization.shared_input.groups.member_multipliers",
        "multiplier members must be unique members of their configured group",
    )
    require(
        all(math.isfinite(entry.multiplier) and entry.multiplier > 0 for entry in multiplier_entries),
        "CFG078",
        "factorization.shared_input.groups.member_multipliers",
        "multipliers must be finite and positive",
    )
    binary_search = config.factorization.binary_search
    require(
        bool(binary_search.layer_patterns)
        and len(set(binary_search.layer_patterns)) == len(binary_search.layer_patterns)
        and all(bool(pattern.strip()) for pattern in binary_search.layer_patterns),
        "CFG133",
        "factorization.binary_search.layer_patterns",
        "must contain unique non-empty patterns",
    )
    require(
        binary_search.scale_passes > 0
        and binary_search.control_outer_passes > 0
        and binary_search.one_bit_passes > 0
        and binary_search.max_one_bit_vectors > 0
        and binary_search.variable_depth_passes > 0
        and binary_search.variable_depth_length > 0
        and binary_search.tabu_outer_passes > 0
        and binary_search.tabu_passes > 0
        and binary_search.tabu_steps > 0
        and binary_search.tabu_tenure > 0
        and binary_search.tabu_tenure_jitter >= 0,
        "CFG134",
        "factorization.binary_search",
        "enabled protocol depths and bounds must be positive",
    )
    require(
        math.isfinite(binary_search.one_bit_fraction)
        and 0 < binary_search.one_bit_fraction <= 1,
        "CFG135",
        "factorization.binary_search.one_bit_fraction",
        "must be finite and in (0, 1]",
    )
    bias = config.factorization.bias_correction
    require(
        bias.storage_dtype in {DType.FLOAT16, DType.BFLOAT16},
        "CFG079",
        "factorization.bias_correction.storage_dtype",
        "must be float16 or bfloat16",
    )
    patch = config.factorization.low_rank_patch
    require(patch.rank > 0, "CFG080", "factorization.low_rank_patch.rank", "must be positive")
    require(
        bool(patch.layer_patterns)
        and len(set(patch.layer_patterns)) == len(patch.layer_patterns)
        and all(bool(pattern.strip()) for pattern in patch.layer_patterns),
        "CFG081",
        "factorization.low_rank_patch.layer_patterns",
        "must contain unique non-empty patterns",
    )
    require(
        patch.storage_dtype in {DType.FLOAT16, DType.BFLOAT16},
        "CFG082",
        "factorization.low_rank_patch.storage_dtype",
        "must be float16 or bfloat16",
    )
    require(
        math.isfinite(patch.ridge_fraction) and patch.ridge_fraction > 0,
        "CFG083",
        "factorization.low_rank_patch.ridge_fraction",
        "must be finite and positive",
    )
    require(
        patch.fit_tokens > 0,
        "CFG084",
        "factorization.low_rank_patch.fit_tokens",
        "must be positive",
    )
    require(
        patch.held_out_tokens > 0,
        "CFG085",
        "factorization.low_rank_patch.held_out_tokens",
        "must be positive",
    )
    require(config.profiling.cuda_sample_every > 0, "CFG015", "profiling.cuda_sample_every", "must be positive")
    require(
        config.profiling.raw_samples_per_phase > 0,
        "CFG016",
        "profiling.raw_samples_per_phase",
        "must be positive",
    )
    for path, loop in (
        ("block_tuning.non_factorized.loop", config.block_tuning.non_factorized.loop),
        ("block_tuning.factorized.loop", config.block_tuning.factorized.loop),
    ):
        require(loop.epochs >= 0, "CFG018", f"{path}.epochs", "must not be negative")
        require(loop.batch_size > 0, "CFG019", f"{path}.batch_size", "must be positive")
        require(not loop.enabled or loop.epochs > 0, "CFG020", path, "enabled loop requires positive epochs")
    microbatch = config.block_tuning.microbatch_size
    require(microbatch is None or microbatch > 0, "CFG021", "block_tuning.microbatch_size", "must be positive")
    refit = config.block_tuning.post_block_refit
    require(refit.epochs >= 0, "CFG022", "block_tuning.post_block_refit.epochs", "must not be negative")
    require(
        not refit.enabled or refit.epochs > 0,
        "CFG023",
        "block_tuning.post_block_refit",
        "enabled refit requires positive epochs",
    )
    require(
        refit.batch_size is None or refit.batch_size > 0,
        "CFG024",
        "block_tuning.post_block_refit.batch_size",
        "must be positive",
    )
    post_refit_covariance = config.block_tuning.post_refit_covariance_refinement
    require(
        post_refit_covariance.enabled
        == bool(
            post_refit_covariance.block_indices
            and post_refit_covariance.shared_input_groups
        ),
        "CFG104",
        "block_tuning.post_refit_covariance_refinement",
        "enabled refinement requires blocks and shared-input groups; configured selections require enabled=true",
    )
    require(
        not post_refit_covariance.enabled or refit.enabled,
        "CFG105",
        "block_tuning.post_refit_covariance_refinement",
        "post-refit covariance refinement requires post-block refit",
    )
    require(
        len(post_refit_covariance.block_indices)
        == len(set(post_refit_covariance.block_indices))
        and all(index >= 0 for index in post_refit_covariance.block_indices),
        "CFG106",
        "block_tuning.post_refit_covariance_refinement.block_indices",
        "block indices must be unique and non-negative",
    )
    require(
        len(post_refit_covariance.shared_input_groups)
        == len(set(post_refit_covariance.shared_input_groups))
        and all(name.strip() for name in post_refit_covariance.shared_input_groups),
        "CFG107",
        "block_tuning.post_refit_covariance_refinement.shared_input_groups",
        "shared-input group names must be non-empty and unique",
    )
    require(
        post_refit_covariance.sampling.max_tokens_per_layer > 0,
        "CFG108",
        "block_tuning.post_refit_covariance_refinement.sampling.max_tokens_per_layer",
        "must be positive",
    )
    require(config.distillation.epochs > 0, "CFG025", "distillation.epochs", "must be positive")
    require(config.distillation.batch_size > 0, "CFG026", "distillation.batch_size", "must be positive")
    require(config.distillation.learning_rate > 0, "CFG027", "distillation.learning_rate", "must be positive")
    require(config.distillation.temperature > 0, "CFG028", "distillation.temperature", "must be positive")
    require(config.distillation.top_k > 0, "CFG029", "distillation.top_k", "must be positive")
    require(
        config.distillation.vocabulary_chunk_size > 0,
        "CFG030",
        "distillation.vocabulary_chunk_size",
        "must be positive",
    )
    require(config.distillation.token_chunk_size > 0, "CFG031", "distillation.token_chunk_size", "must be positive")
    require(
        config.distillation.maximum_tokens_per_batch is None or config.distillation.maximum_tokens_per_batch > 0,
        "CFG032",
        "distillation.maximum_tokens_per_batch",
        "must be positive when provided",
    )
    require(
        config.distillation.maximum_batches_per_epoch is None
        or config.distillation.maximum_batches_per_epoch > 0,
        "CFG109",
        "distillation.maximum_batches_per_epoch",
        "must be positive when provided",
    )
    require(
        math.isfinite(config.distillation.tail_mass_weight)
        and config.distillation.tail_mass_weight > 0,
        "CFG110",
        "distillation.tail_mass_weight",
        "must be finite and positive",
    )
    require(config.distillation.weight_decay >= 0, "CFG033", "distillation.weight_decay", "must not be negative")
    correction = config.distillation.mass_floor_correction
    require(
        not correction.enabled or config.distillation.enabled,
        "CFG120",
        "distillation.mass_floor_correction.enabled",
        "requires global distillation",
    )
    require(
        not correction.enabled
        or (
            isinstance(correction.expected_initializer_protocol_hash, str)
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                correction.expected_initializer_protocol_hash,
            )
            is not None
        ),
        "CFG130",
        "distillation.mass_floor_correction.expected_initializer_protocol_hash",
        "must bind an exact sha256 protocol when correction is enabled",
    )
    require(
        not correction.enabled
        or (
            correction.expected_initializer_steps is not None
            and correction.expected_initializer_steps > 0
        ),
        "CFG131",
        "distillation.mass_floor_correction.expected_initializer_steps",
        "must bind a positive completed-step count when correction is enabled",
    )
    require(
        correction.epochs > 0,
        "CFG121",
        "distillation.mass_floor_correction.epochs",
        "must be positive",
    )
    require(
        math.isfinite(correction.learning_rate) and correction.learning_rate > 0,
        "CFG122",
        "distillation.mass_floor_correction.learning_rate",
        "must be finite and positive",
    )
    require(
        correction.maximum_batches_per_epoch > 0,
        "CFG123",
        "distillation.mass_floor_correction.maximum_batches_per_epoch",
        "must be positive",
    )
    require(
        correction.scheduler_total_steps > 0,
        "CFG128",
        "distillation.mass_floor_correction.scheduler_total_steps",
        "must be positive",
    )
    require(
        correction.scheduler_total_steps
        >= correction.epochs * correction.maximum_batches_per_epoch,
        "CFG129",
        "distillation.mass_floor_correction.scheduler_total_steps",
        "must cover every configured correction training step",
    )
    require(
        math.isfinite(correction.minimum_teacher_mass_ratio)
        and 0 < correction.minimum_teacher_mass_ratio <= 1,
        "CFG124",
        "distillation.mass_floor_correction.minimum_teacher_mass_ratio",
        "must be finite and in (0, 1]",
    )
    require(
        math.isfinite(correction.mass_loss_weight) and correction.mass_loss_weight > 0,
        "CFG125",
        "distillation.mass_floor_correction.mass_loss_weight",
        "must be finite and positive",
    )
    final_norm = config.distillation.final_norm_calibration
    require(
        not final_norm.enabled or config.distillation.enabled,
        "CFG126",
        "distillation.final_norm_calibration.enabled",
        "requires global distillation",
    )
    require(
        math.isfinite(final_norm.scale) and final_norm.scale > 0,
        "CFG127",
        "distillation.final_norm_calibration.scale",
        "must be finite and positive",
    )
    foldable = config.distillation.foldable_mlp_multipliers
    require(
        not foldable.enabled or config.distillation.enabled,
        "CFG109",
        "distillation.foldable_mlp_multipliers.enabled",
        "requires global distillation",
    )
    require(
        foldable.steps > 0,
        "CFG110",
        "distillation.foldable_mlp_multipliers.steps",
        "must be positive",
    )
    require(
        math.isfinite(foldable.learning_rate) and foldable.learning_rate > 0,
        "CFG111",
        "distillation.foldable_mlp_multipliers.learning_rate",
        "must be finite and positive",
    )
    require(
        math.isfinite(foldable.identity_penalty) and foldable.identity_penalty >= 0,
        "CFG112",
        "distillation.foldable_mlp_multipliers.identity_penalty",
        "must be finite and non-negative",
    )
    require(
        math.isfinite(foldable.gradient_clip) and foldable.gradient_clip > 0,
        "CFG113",
        "distillation.foldable_mlp_multipliers.gradient_clip",
        "must be finite and positive",
    )
    require(
        math.isfinite(foldable.multiplier_limit) and foldable.multiplier_limit > 1,
        "CFG114",
        "distillation.foldable_mlp_multipliers.multiplier_limit",
        "must be finite and greater than one",
    )
    require(
        foldable.checkpoint_interval_steps > 0,
        "CFG115",
        "distillation.foldable_mlp_multipliers.checkpoint_interval_steps",
        "must be positive",
    )
    require(
        (foldable.initializer_artifact is None) == (foldable.initializer_sha256 is None),
        "CFG116",
        "distillation.foldable_mlp_multipliers.initializer_artifact",
        "must be paired with initializer_sha256",
    )
    require(
        foldable.initializer_artifact is None or bool(foldable.initializer_artifact.strip()),
        "CFG117",
        "distillation.foldable_mlp_multipliers.initializer_artifact",
        "must be non-empty when provided",
    )
    require(
        foldable.initializer_sha256 is None
        or (
            len(foldable.initializer_sha256) == 64
            and all(character in "0123456789abcdef" for character in foldable.initializer_sha256)
        ),
        "CFG118",
        "distillation.foldable_mlp_multipliers.initializer_sha256",
        "must be a lowercase SHA-256 digest",
    )
    require(
        math.isfinite(foldable.initializer_multiplier_limit)
        and foldable.initializer_multiplier_limit > 1,
        "CFG119",
        "distillation.foldable_mlp_multipliers.initializer_multiplier_limit",
        "must be finite and greater than one",
    )
    require(
        config.evaluation.inline_quality_samples > 0, "CFG034", "evaluation.inline_quality_samples", "must be positive"
    )
    require(
        config.evaluation.inline_quality_tokens > 0, "CFG035", "evaluation.inline_quality_tokens", "must be positive"
    )
    require(
        config.observability.block_snapshot_samples > 0,
        "CFG036",
        "observability.block_snapshot_samples",
        "must be positive",
    )
    require(
        config.observability.block_snapshot_tokens > 0,
        "CFG037",
        "observability.block_snapshot_tokens",
        "must be positive",
    )
    levels: dict[str, Severity] = {}
    for path, value in (
        ("observability.console_level", config.observability.console_level),
        ("observability.event_level", config.observability.event_level),
    ):
        try:
            levels[path] = Severity.parse(value)
        except ValueError:
            require(False, "OBS001", path, "must be one of debug, info, warning, error")
    console_level = levels.get("observability.console_level")
    event_level = levels.get("observability.event_level")
    if console_level is not None and event_level is not None:
        require(
            event_level.rank <= console_level.rank,
            "OBS002",
            "observability.event_level",
            "must be at least as verbose as observability.console_level",
        )
        require(
            not config.observability.record_admm_steps or event_level is Severity.DEBUG,
            "OBS003",
            "observability.record_admm_steps",
            "requires observability.event_level=debug",
        )
    resource_interval = config.observability.record_resource_interval_seconds
    require(
        math.isfinite(resource_interval),
        "OBS004",
        "observability.record_resource_interval_seconds",
        "must be finite; values at or below zero disable resource sampling",
    )
    if math.isfinite(resource_interval) and 0 < resource_interval < 1:
        issues.append(
            ValidationIssue(
                "OBS004",
                "observability.record_resource_interval_seconds",
                "intervals below one second may create excessive event volume",
                "warning",
            )
        )
    if config.calibration.objective.kind is ObjectiveKind.BLOCK_DIAGONAL:
        require(
            bool(config.calibration.objective.block_size and config.calibration.objective.block_size > 0),
            "CFG013",
            "calibration.objective.block_size",
            "is required for block-diagonal objectives",
        )
    if config.calibration.objective.kind is ObjectiveKind.LOW_RANK_DIAGONAL:
        require(
            bool(config.calibration.objective.low_rank and config.calibration.objective.low_rank > 0),
            "CFG014",
            "calibration.objective.low_rank",
            "is required for low-rank-diagonal objectives",
        )
    if phase in (ValidationPhase.RESOLVED, ValidationPhase.PLANNED):
        require(
            config.model.revision is not None, "RES001", "model.revision", "resolved config requires a pinned revision"
        )
        require(
            config.model.tokenizer_revision is not None,
            "RES002",
            "model.tokenizer_revision",
            "resolved config requires a pinned tokenizer revision",
        )
    return tuple(issues)


def raise_for_issues(issues: tuple[ValidationIssue, ...]) -> None:
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        rendered = "\n".join(f"{item.code} {item.path}: {item.message}" for item in errors)
        raise ValueError(rendered)
