import math
from dataclasses import FrozenInstanceError, replace

import pytest

from nanoquant.config.codec import (
    ConfigDecodeError,
    apply_overrides,
    canonical_json,
    config_hash,
    from_dict,
    semantic_hash,
    to_dict,
)
from nanoquant.config.migration import migrate_legacy, migration_inventory
from nanoquant.config.resolution import resolve_config
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ActivationStorageConfig,
    ActivationStoreKind,
    AllocationStrategy,
    BehaviorSliceConfig,
    BinaryFactorSearchConfig,
    DatasetConfig,
    DatasetSourceConfig,
    DistillationLoss,
    DType,
    FoldableMlpMultiplierTuningConfig,
    KlSensitivityGranularity,
    LayerRankBudgetConfig,
    ModelConfig,
    ObjectiveKind,
    ObservabilityConfig,
    PostRefitCovarianceRefinementConfig,
    ProfilingConfig,
    ProfilingLevel,
    ReasoningMode,
    RunConfig,
    TeacherTraceGenerationConfig,
)
from nanoquant.config.validation import ValidationPhase, validate


def test_round_trip_decodes_nested_enums_tuples_and_optionals() -> None:
    raw = {
        "model": {"source": "local/tiny", "load_dtype": "float16"},
        "dataset": {"sources": [{"name": "fixture", "revision": None}], "shuffle": False},
        "allocation": {
            "layer_budget_multipliers": [{"pattern": "self_attn.q_proj", "multiplier": 1.25}],
            "kl_sensitivity_granularity": "type_block",
        },
        "runtime": {"activations": {"kind": "mmap", "gpu_cache": "auto", "gpu_reserve_gib": 1.5}},
        "calibration": {"objective": {"kind": "low_rank_diagonal", "low_rank": 4}},
        "profiling": {"level": "micro", "trace_blocks": [3, 7]},
    }
    config = from_dict(RunConfig, raw)
    assert config.model.load_dtype is DType.FLOAT16
    assert config.dataset.sources == (DatasetSourceConfig(name="fixture"),)
    assert config.allocation.layer_budget_multipliers == (
        LayerRankBudgetConfig("self_attn.q_proj", 1.25),
    )
    assert config.allocation.kl_sensitivity_granularity is KlSensitivityGranularity.TYPE_BLOCK
    assert config.runtime.activations.kind is ActivationStoreKind.MMAP
    assert config.runtime.activations.gpu_cache is ActivationGpuCacheMode.AUTO
    assert config.runtime.activations.gpu_reserve_gib == 1.5
    assert config.calibration.objective.kind is ObjectiveKind.LOW_RANK_DIAGONAL
    assert config.profiling.level is ProfilingLevel.MICRO
    assert config.profiling.trace_blocks == (3, 7)
    assert from_dict(RunConfig, to_dict(config)) == config


def test_post_refit_covariance_selection_requires_refit_blocks_and_groups() -> None:
    base = RunConfig(ModelConfig("x"))
    selected = PostRefitCovarianceRefinementConfig(
        enabled=True,
        block_indices=(5, 11, 24, 25),
        shared_input_groups=("self_attn.attn_qkv",),
    )
    missing_refit = replace(
        base,
        block_tuning=replace(
            base.block_tuning,
            post_refit_covariance_refinement=selected,
        ),
    )
    enabled_refit = replace(
        missing_refit,
        block_tuning=replace(
            missing_refit.block_tuning,
            post_block_refit=replace(
                missing_refit.block_tuning.post_block_refit,
                enabled=True,
                epochs=1,
            ),
        ),
    )

    assert {issue.code for issue in validate(missing_refit)} == {"CFG105"}
    assert not validate(enabled_refit)


def test_behavior_slices_and_reasoning_modes_round_trip_and_validate() -> None:
    source = DatasetSourceConfig("fixture/reasoning", revision="pinned")
    config = replace(
        RunConfig(ModelConfig("x")),
        dataset=DatasetConfig(
            behavior_slices=(
                BehaviorSliceConfig(
                    "thinking",
                    ReasoningMode.THINKING,
                    source,
                    "openr1_generations",
                    0.5,
                ),
                BehaviorSliceConfig(
                    "non-thinking",
                    ReasoningMode.NON_THINKING,
                    source,
                    "ultrachat_messages",
                    0.5,
                ),
            )
        ),
        evaluation=replace(
            RunConfig(ModelConfig("x")).evaluation,
            reasoning_modes=(ReasoningMode.THINKING, ReasoningMode.NON_THINKING),
        ),
    )

    assert not validate(config)
    assert from_dict(RunConfig, to_dict(config)) == config


def test_teacher_trace_generation_is_pinned_and_only_valid_for_thinking_ultrachat() -> None:
    source = DatasetSourceConfig("fixture/ultrachat", revision="pinned")
    base = RunConfig(ModelConfig("Qwen/Qwen3", revision="teacher-revision"))
    trace = TeacherTraceGenerationConfig(maximum_new_tokens=128, minimum_new_tokens=8)
    valid = replace(
        base,
        dataset=DatasetConfig(
            behavior_slices=(
                BehaviorSliceConfig(
                    "thinking",
                    ReasoningMode.THINKING,
                    source,
                    "ultrachat_messages",
                    1.0,
                    teacher_trace_generation=trace,
                ),
            )
        ),
    )

    assert validate(valid) == ()
    assert from_dict(RunConfig, to_dict(valid)) == valid
    wrong_mode = replace(
        valid,
        dataset=replace(
            valid.dataset,
            behavior_slices=(
                replace(valid.dataset.behavior_slices[0], mode=ReasoningMode.RAW),
            ),
        ),
    )
    unpinned = replace(valid, model=replace(valid.model, revision=None))
    assert {issue.code for issue in validate(wrong_mode)} == {"CFG101"}
    assert {issue.code for issue in validate(unpinned)} == {"CFG103"}


def test_unknown_path_has_full_path_and_suggestion() -> None:
    with pytest.raises(ConfigDecodeError, match=r"config\.calibration\.sampl_count.*sample_count"):
        from_dict(RunConfig, {"model": {"source": "x"}, "calibration": {"sampl_count": 3}})
    with pytest.raises(ConfigDecodeError, match=r"allocation\.target_bpwz.*target_bpw"):
        apply_overrides(RunConfig(ModelConfig("x")), {"allocation.target_bpwz": 1})


def test_canonical_serialization_is_deterministic_and_config_is_frozen() -> None:
    config = RunConfig(ModelConfig("x"))
    assert canonical_json(config) == canonical_json(from_dict(RunConfig, to_dict(config)))
    with pytest.raises(FrozenInstanceError):
        config.schema_version = 2  # type: ignore[misc]


def test_sparse_overrides_use_schema_types() -> None:
    config = apply_overrides(
        RunConfig(ModelConfig("x")),
        {"runtime.activations.kind": "ram", "intent.tags": ["a", "b"], "model.revision": None},
    )
    assert config.runtime.activations.kind is ActivationStoreKind.RAM
    assert config.intent.tags == ("a", "b")


def test_validation_phases_have_stable_codes() -> None:
    config = RunConfig(ModelConfig("x"))
    assert validate(config) == ()
    assert {issue.code for issue in validate(config, ValidationPhase.RESOLVED)} == {"RES001", "RES002"}
    invalid = RunConfig(
        ModelConfig("x"),
        profiling=ProfilingConfig(cuda_sample_every=0, raw_samples_per_phase=0),
    )
    assert {issue.code for issue in validate(invalid)} == {"CFG015", "CFG016"}


def test_mass_floor_correction_requires_primary_distillation_and_valid_policy() -> None:
    base = RunConfig(ModelConfig("x"))
    correction = replace(
        base.distillation.mass_floor_correction,
        enabled=True,
        expected_initializer_protocol_hash="invalid",
        expected_initializer_steps=0,
        epochs=0,
        learning_rate=math.nan,
        maximum_batches_per_epoch=0,
        scheduler_total_steps=0,
        minimum_teacher_mass_ratio=1.1,
        mass_loss_weight=0.0,
    )
    invalid = replace(
        base,
        distillation=replace(base.distillation, mass_floor_correction=correction),
    )

    assert {issue.code for issue in validate(invalid)} == {
        "CFG120",
        "CFG121",
        "CFG122",
        "CFG123",
        "CFG124",
        "CFG125",
        "CFG128",
        "CFG130",
        "CFG131",
    }

    missing_regime = replace(
        base,
        distillation=replace(
            base.distillation,
            enabled=True,
            mass_floor_correction=replace(
                base.distillation.mass_floor_correction,
                enabled=True,
            ),
        ),
    )
    assert {issue.code for issue in validate(missing_regime)} == {
        "CFG130",
        "CFG131",
    }

    bad_final_norm = replace(
        base,
        distillation=replace(
            base.distillation,
            final_norm_calibration=replace(
                base.distillation.final_norm_calibration,
                enabled=True,
                scale=math.nan,
            ),
        ),
    )
    assert {issue.code for issue in validate(bad_final_norm)} == {"CFG126", "CFG127"}


def test_activation_gpu_cache_reserve_must_be_finite_and_non_negative() -> None:
    base = RunConfig(ModelConfig("x"))

    for reserve in (-1.0, math.inf, math.nan):
        invalid = replace(
            base,
            runtime=replace(
                base.runtime,
                activations=ActivationStorageConfig(gpu_reserve_gib=reserve),
            ),
        )
        assert {issue.code for issue in validate(invalid)} == {"CFG044"}


def test_maximum_rank_patterns_must_be_nonempty_and_unique() -> None:
    config = RunConfig(ModelConfig("x"))
    invalid = replace(
        config,
        allocation=replace(
            config.allocation,
            maximum_rank_layer_patterns=("", "self_attn.v_proj", "self_attn.v_proj"),
        ),
    )

    assert {issue.code for issue in validate(invalid)} == {"CFG039", "CFG040"}


def test_overcomplete_rank_ceiling_must_be_finite_and_at_least_one() -> None:
    base = RunConfig(ModelConfig("x"))

    for value in (0.99, math.inf, math.nan):
        invalid = replace(
            base,
            allocation=replace(
                base.allocation,
                bounds=replace(
                    base.allocation.bounds,
                    overcomplete_rank_ceiling_fraction=value,
                ),
            ),
        )
        assert {issue.code for issue in validate(invalid)} == {"CFG132"}


def test_layer_budget_multipliers_must_be_valid_and_unique() -> None:
    config = RunConfig(ModelConfig("x"))
    invalid = replace(
        config,
        allocation=replace(
            config.allocation,
            layer_budget_multipliers=(
                LayerRankBudgetConfig("", 1.0),
                LayerRankBudgetConfig("", math.inf),
            ),
        ),
    )

    assert {issue.code for issue in validate(invalid)} == {"CFG041", "CFG042", "CFG043"}


def test_kl_calibrated_allocation_requires_both_profile_path_and_exact_key() -> None:
    base = RunConfig(ModelConfig("x"))
    missing = replace(
        base,
        allocation=replace(base.allocation, strategy=AllocationStrategy.KL_CALIBRATED),
    )
    complete = replace(
        base,
        allocation=replace(
            base.allocation,
            strategy=AllocationStrategy.KL_CALIBRATED,
            kl_profile_artifact="evidence/profile",
            kl_profile_key="sha256:profile",
        ),
    )
    unexpected = replace(
        base,
        allocation=replace(
            base.allocation,
            kl_profile_artifact="evidence/profile",
            kl_profile_key="sha256:profile",
        ),
    )

    missing_codes = {issue.code for issue in validate(missing)}
    complete_codes = {issue.code for issue in validate(complete)}
    assert {"CFG076", "CFG086"}.issubset(missing_codes)
    assert "CFG076" not in complete_codes
    assert "CFG086" not in complete_codes
    assert {issue.code for issue in validate(unexpected)} == {"CFG076", "CFG086"}


def test_type_block_kl_granularity_requires_kl_calibrated_allocation() -> None:
    base = RunConfig(ModelConfig("x"))
    invalid = replace(
        base,
        allocation=replace(
            base.allocation,
            kl_sensitivity_granularity=KlSensitivityGranularity.TYPE_BLOCK,
        ),
    )

    assert {issue.code for issue in validate(invalid)} == {"CFG087"}


def test_low_rank_patch_fit_and_held_out_windows_must_be_positive() -> None:
    base = RunConfig(ModelConfig("x"))
    invalid = replace(
        base,
        factorization=replace(
            base.factorization,
            low_rank_patch=replace(
                base.factorization.low_rank_patch,
                fit_tokens=0,
                held_out_tokens=0,
            ),
        ),
    )

    assert {issue.code for issue in validate(invalid)} == {"CFG084", "CFG085"}


def test_binary_factor_search_protocol_is_typed_and_bounded() -> None:
    base = RunConfig(ModelConfig("x"))
    enabled = replace(
        base,
        factorization=replace(
            base.factorization,
            binary_search=BinaryFactorSearchConfig(enabled=True),
        ),
    )
    invalid = replace(
        enabled,
        factorization=replace(
            enabled.factorization,
            binary_search=replace(
                enabled.factorization.binary_search,
                layer_patterns=("", ""),
                tabu_steps=0,
                one_bit_fraction=math.nan,
            ),
        ),
    )

    assert validate(enabled) == ()
    assert from_dict(RunConfig, to_dict(enabled)) == enabled
    assert {issue.code for issue in validate(invalid)} == {
        "CFG133",
        "CFG134",
        "CFG135",
    }


def test_observability_levels_are_validated_without_changing_schema() -> None:
    invalid_name = RunConfig(ModelConfig("x"), observability=ObservabilityConfig(event_level="trace"))
    assert {issue.code for issue in validate(invalid_name)} == {"OBS001"}

    console_more_verbose = RunConfig(
        ModelConfig("x"),
        observability=ObservabilityConfig(event_level="info", console_level="debug"),
    )
    assert {issue.code for issue in validate(console_more_verbose)} == {"OBS002"}

    silent_admm = RunConfig(
        ModelConfig("x"),
        observability=ObservabilityConfig(event_level="info", record_admm_steps=True),
    )
    assert {issue.code for issue in validate(silent_admm)} == {"OBS003"}

    debug_admm = RunConfig(
        ModelConfig("x"),
        observability=ObservabilityConfig(event_level="debug", record_admm_steps=True),
    )
    assert validate(debug_admm) == ()


def test_resource_interval_validation_rejects_nonfinite_and_warns_on_high_volume() -> None:
    invalid = RunConfig(
        ModelConfig("x"),
        observability=ObservabilityConfig(record_resource_interval_seconds=math.inf),
    )
    assert [(issue.code, issue.severity) for issue in validate(invalid)] == [("OBS004", "error")]

    noisy = RunConfig(
        ModelConfig("x"),
        observability=ObservabilityConfig(record_resource_interval_seconds=0.5),
    )
    assert [(issue.code, issue.severity) for issue in validate(noisy)] == [("OBS004", "warning")]

    disabled = RunConfig(
        ModelConfig("x"),
        observability=ObservabilityConfig(record_resource_interval_seconds=0),
    )
    assert validate(disabled) == ()


def test_foldable_mlp_initializer_path_and_hash_are_paired_and_pinned() -> None:
    base = RunConfig(ModelConfig("x"))
    missing_hash = replace(
        base,
        distillation=replace(
            base.distillation,
            foldable_mlp_multipliers=FoldableMlpMultiplierTuningConfig(
                initializer_artifact="initializers/candidate"
            ),
        ),
    )
    malformed_hash = replace(
        base,
        distillation=replace(
            base.distillation,
            foldable_mlp_multipliers=FoldableMlpMultiplierTuningConfig(
                initializer_artifact="initializers/candidate",
                initializer_sha256="ABC",
            ),
        ),
    )

    assert {issue.code for issue in validate(missing_hash)} == {"CFG116"}
    assert {issue.code for issue in validate(malformed_hash)} == {"CFG118"}

    invalid_limit = replace(
        base,
        distillation=replace(
            base.distillation,
            foldable_mlp_multipliers=FoldableMlpMultiplierTuningConfig(
                initializer_multiplier_limit=1.0
            ),
        ),
    )
    assert {issue.code for issue in validate(invalid_limit)} == {"CFG119"}


def test_tail_distillation_batch_cap_and_mass_weight_validate() -> None:
    base = RunConfig(ModelConfig("x"))
    invalid_cap = replace(
        base,
        distillation=replace(base.distillation, maximum_batches_per_epoch=0),
    )
    invalid_weight = replace(
        base,
        distillation=replace(base.distillation, tail_mass_weight=math.inf),
    )

    assert {issue.code for issue in validate(invalid_cap)} == {"CFG109"}
    assert {issue.code for issue in validate(invalid_weight)} == {"CFG110"}


def test_noop_tail_fields_preserve_legacy_config_hash() -> None:
    legacy = RunConfig(ModelConfig("x"))
    legacy_payload = to_dict(legacy)
    legacy_payload["distillation"].pop("maximum_batches_per_epoch")
    legacy_payload["distillation"].pop("tail_mass_weight")
    conditional_with_unused_tail_weight = replace(
        legacy,
        distillation=replace(legacy.distillation, tail_mass_weight=0.5),
    )
    tail = replace(
        legacy,
        distillation=replace(
            legacy.distillation,
            loss=DistillationLoss.TOP_K_TAIL,
            maximum_batches_per_epoch=32,
            tail_mass_weight=0.5,
        ),
    )

    assert config_hash(legacy) == semantic_hash(legacy_payload)
    assert config_hash(conditional_with_unused_tail_weight) == config_hash(legacy)
    assert config_hash(tail) != config_hash(legacy)


def test_legacy_migration_is_total_and_rejects_uninventoried_fields() -> None:
    migrated, inventory = migrate_legacy(
        {
            "model_id": "local/tiny",
            "bits": 0.9,
            "hessian_whitening": True,
            "outlier_dtype": "bf16",
            "weight_error_log_path": "ignored.csv",
        }
    )
    assert migrated.allocation.target_bpw == 0.9
    assert migrated.calibration.objective.kind is ObjectiveKind.DENSE_HESSIAN
    assert migrated.outliers.storage_dtype is DType.BFLOAT16
    assert any(item.legacy_field == "weight_error_log_path" and item.disposition == "removed" for item in inventory)
    with pytest.raises(ConfigDecodeError, match="mystery"):
        migrate_legacy({"model_id": "x", "mystery": 1})


def test_legacy_retry_count_migrates_to_total_attempt_count() -> None:
    migrated, _inventory = migrate_legacy({"model_id": "local/tiny", "rank_retry_max_attempts": 2})

    assert migrated.allocation.retry.maximum_attempts == 3


def test_frozen_legacy_inventory_has_one_disposition_for_all_95_fields() -> None:
    inventory = migration_inventory()
    assert len(inventory) == 95
    assert len({entry.legacy_field for entry in inventory}) == 95
    assert all(entry.disposition in {"mapped", "removed"} for entry in inventory)


def test_resolution_is_immutable_and_pins_model_and_tokenizer() -> None:
    class Resolver:
        def resolve(self, source: str, revision: str | None) -> str:
            return revision or f"sha-{source}"

    original = RunConfig(ModelConfig("local/tiny"))
    resolved = resolve_config(original, Resolver())
    assert original.model.revision is None
    assert resolved.model.revision == "sha-local/tiny"
    assert resolved.model.tokenizer_source == "local/tiny"
    assert resolved.model.tokenizer_revision == "sha-local/tiny"
