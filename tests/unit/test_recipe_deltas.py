from dataclasses import dataclass

import pytest
from recipes import (
    BASE_COMPRESSION_CONFIG,
    EXPERIMENT_001_CONFIG,
    EXPERIMENT_002_CONFIG,
    EXPERIMENT_003_CONFIG,
    EXPERIMENT_004_CONFIG,
    EXPERIMENT_005_CONFIG,
    EXPERIMENT_006_CONFIG,
    EXPERIMENT_007_CONFIG,
    EXPERIMENT_008_CONFIG,
    LARGE_MODEL_COMPRESSION_CONFIG,
)
from recipes._delta import config_delta, run_config_defaults
from recipes.legacy import (
    EXPERIMENT_008_CONFIG as LEGACY_EXPERIMENT_008_CONFIG,
)
from recipes.legacy import (
    EXPERIMENT_011_CONFIG,
    EXPERIMENT_013_CONFIG,
    EXPERIMENT_018_CONFIG,
    LEGACY_SHORT_DECODE_CONFIG,
)

from nanoquant.config.codec import config_hash


@dataclass(frozen=True)
class _Fixture:
    value: int
    label: str


def test_config_delta_rejects_repeated_parent_values() -> None:
    parent = _Fixture(1, "base")

    assert config_delta(parent, value=2) == _Fixture(2, "base")
    with pytest.raises(ValueError, match="repeats inherited.*value"):
        config_delta(parent, value=1)


def test_run_config_defaults_contains_only_the_required_model_source() -> None:
    defaults = run_config_defaults("model")

    assert defaults.model.source == "model"
    assert defaults.intent.name == "unnamed-run"
    assert defaults.runtime.compute_device == "cuda:0"


def test_large_model_base_recipe_pins_bounded_memory_guards() -> None:
    config = LARGE_MODEL_COMPRESSION_CONFIG

    assert config.runtime.executor.value == "cpu_offload"
    assert config.calibration.method.value == "forward_only"
    assert config.runtime.activations.gpu_cache.value == "auto"
    assert not config.evaluation.inline_quality
    assert not config.distillation.enabled


def test_derived_recipe_hashes_are_unchanged_by_delta_only_sources() -> None:
    expected = {
        "base": "sha256:70649cdf490c4669deb3fde28820b4d2e964fc9d09f1237b935488a1c2f07d4c",
        "large": "sha256:c8411ce5c0b6574c8700df47d086436121d3dae1356e3e92fba48da88cc4be7d",
        "001": "sha256:e4c71a1e4977477bdd6a835783c8de27f867ab3941169efb139bd6e14126e3ee",
        "002": "sha256:edf87371018dd3b9d2c91d85a853d9571cdc62b18b444ba68d236e4795561f6f",
        "003": "sha256:c5bb6251a490a575ced7ad12bde112561e56b65cf71de86ea7ede2248ee42c6a",
        "004": "sha256:92e5438aceabe34a7ca6062a6924a92da9f2599ed7d32c826cb70eac65bf5302",
        "005": "sha256:86afa37533f4bc44c2f9339b5ba297458c2a7e14424d0c3450bed9c1583b2328",
        "006": "sha256:2886bd03dc3f65ad22f8ec319c659ce15357fa23a726f6dc739467294dbced62",
        "007": "sha256:eab1b0d3eaefaecedb4b8d66695b13f94ee592f07b6b65f2208355849da0b31d",
        "008": "sha256:ab866a49b5a3d061875a5d2111fdee9877994e47411762896d7b93b6c3ddf93f",
        "legacy-008": "sha256:1dd444374f8c62649e025529a7f62cae570ab019ed1fdc3b2c751fca3046f335",
        "011": "sha256:aa64a6604ff724cb526a89114b377f96903739385fba83bb15eacb47b8572013",
        "013": "sha256:3335aadc2979eb74b7d0f637811920eeb741fe1c779214e65de718a572abe2b2",
        "018": "sha256:29116da2e7ff34853dafdb9379dad66bc8031e06cc5ba580e3c3857e62ac6498",
        "short": "sha256:ee105132c19fde12a85269b0ab40e998e253fb5d05f726e313424be0ffeff326",
    }
    configs = {
        "base": BASE_COMPRESSION_CONFIG,
        "large": LARGE_MODEL_COMPRESSION_CONFIG,
        "001": EXPERIMENT_001_CONFIG,
        "002": EXPERIMENT_002_CONFIG,
        "003": EXPERIMENT_003_CONFIG,
        "004": EXPERIMENT_004_CONFIG,
        "005": EXPERIMENT_005_CONFIG,
        "006": EXPERIMENT_006_CONFIG,
        "007": EXPERIMENT_007_CONFIG,
        "008": EXPERIMENT_008_CONFIG,
        "legacy-008": LEGACY_EXPERIMENT_008_CONFIG,
        "011": EXPERIMENT_011_CONFIG,
        "013": EXPERIMENT_013_CONFIG,
        "018": EXPERIMENT_018_CONFIG,
        "short": LEGACY_SHORT_DECODE_CONFIG,
    }

    assert {name: config_hash(config) for name, config in configs.items()} == expected
