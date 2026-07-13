from dataclasses import FrozenInstanceError

import pytest

from nanoquant.config.codec import ConfigDecodeError, apply_overrides, canonical_json, from_dict, to_dict
from nanoquant.config.migration import migrate_legacy, migration_inventory
from nanoquant.config.resolution import resolve_config
from nanoquant.config.schema import (
    ActivationStoreKind,
    DatasetSourceConfig,
    DType,
    ModelConfig,
    ObjectiveKind,
    ProfilingConfig,
    ProfilingLevel,
    RunConfig,
)
from nanoquant.config.validation import ValidationPhase, validate


def test_round_trip_decodes_nested_enums_tuples_and_optionals() -> None:
    raw = {
        "model": {"source": "local/tiny", "load_dtype": "float16"},
        "dataset": {"sources": [{"name": "fixture", "revision": None}], "shuffle": False},
        "runtime": {"activations": {"kind": "mmap"}},
        "calibration": {"objective": {"kind": "low_rank_diagonal", "low_rank": 4}},
        "profiling": {"level": "micro", "trace_blocks": [3, 7]},
    }
    config = from_dict(RunConfig, raw)
    assert config.model.load_dtype is DType.FLOAT16
    assert config.dataset.sources == (DatasetSourceConfig(name="fixture"),)
    assert config.runtime.activations.kind is ActivationStoreKind.MMAP
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
