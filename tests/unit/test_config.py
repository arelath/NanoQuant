import math
from dataclasses import FrozenInstanceError, replace

import pytest

from nanoquant.config.codec import ConfigDecodeError, apply_overrides, canonical_json, from_dict, to_dict
from nanoquant.config.migration import migrate_legacy, migration_inventory
from nanoquant.config.resolution import resolve_config
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ActivationStorageConfig,
    ActivationStoreKind,
    AllocationStrategy,
    DatasetSourceConfig,
    DType,
    KlSensitivityGranularity,
    LayerRankBudgetConfig,
    ModelConfig,
    ObjectiveKind,
    ObservabilityConfig,
    ProfilingConfig,
    ProfilingLevel,
    RunConfig,
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
