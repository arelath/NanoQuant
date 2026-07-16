from dataclasses import dataclass

import pytest
from recipes import (
    EXPERIMENT_001_CONFIG,
    EXPERIMENT_003_CONFIG,
    EXPERIMENT_004_CONFIG,
    EXPERIMENT_005_CONFIG,
)
from recipes._delta import config_delta
from recipes.legacy import EXPERIMENT_008_CONFIG, EXPERIMENT_013_CONFIG, EXPERIMENT_018_CONFIG

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


def test_derived_recipe_hashes_are_unchanged_by_delta_only_sources() -> None:
    expected = {
        "001": "sha256:023a487366084505437fac8cc33482fdea027ee9aa1a39bf54297f4754d728c1",
        "003": "sha256:8f685c7d2f4f80ea30d056808ab7b8fc15972929d75965dc0176a3e1339e23ad",
        "004": "sha256:4ef82d8a21f5e31fa0bff7f97885c74c9a01fca97ee71fd3a484a2ba85f52809",
        "005": "sha256:70494f3097448faca7d0fd5a2922ab4b7acba56a87dc2b7a591fcab80080358f",
        "008": "sha256:b9e587a326069f8c74496652c2095698a613d55ade08ebd547ea68a1216e8bee",
        "013": "sha256:663df968f4d9f176bda4c6ad194d5a630fd026eda11fb758c0c3968aa2168679",
        "018": "sha256:c20d23b8fb2bcbe168408de0fe1f180b50016c161b3d287ca3eb08447f8c192d",
    }
    configs = {
        "001": EXPERIMENT_001_CONFIG,
        "003": EXPERIMENT_003_CONFIG,
        "004": EXPERIMENT_004_CONFIG,
        "005": EXPERIMENT_005_CONFIG,
        "008": EXPERIMENT_008_CONFIG,
        "013": EXPERIMENT_013_CONFIG,
        "018": EXPERIMENT_018_CONFIG,
    }

    assert {name: config_hash(config) for name, config in configs.items()} == expected
