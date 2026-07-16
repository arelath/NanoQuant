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
)
from recipes._delta import config_delta, run_config_defaults
from recipes.legacy import (
    EXPERIMENT_008_CONFIG,
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


def test_derived_recipe_hashes_are_unchanged_by_delta_only_sources() -> None:
    expected = {
        "base": "sha256:2106e6de0e0299dd296be23e0771bbb97cc2d1bb057a39a6c1d73c9689d817c7",
        "001": "sha256:d826aa290face0ef6704bf70e149074ef0c7cfe3242ec663f919a1bf45b51ad5",
        "002": "sha256:cc309b583e4f8ca765dd26e491c4b1cbf4b964f4f4ed2df8edab340aa918da8c",
        "003": "sha256:a099ece06bae15062617c1e8c546f79cbffde4758d88d68c2e4505a85cae0a02",
        "004": "sha256:451b37cf97016dcc1da41102f435b0cc98930804700f8079e6341499a7dce72d",
        "005": "sha256:78ae6f01d8af99e5516847a2efef759f1ffe5e561d8f5d5abf63453b7cd09b22",
        "006": "sha256:d4643196beadf9ac009df9d8406cca13780ee83b08933a30682ca355a1dd2a87",
        "007": "sha256:8ed4c7d0aa65c739d16c8b6eadc98d5f46ca4b7ceb8e5d47f495acf869a00e5a",
        "008": "sha256:b0f7587302ea70524f615a65c2c26bc853142f1c345287cd8fb8122e4eaf7439",
        "011": "sha256:016a5f54b4260bfbd95cdab1b56ce56793b0cbd6212a811d2711c19a84aa2967",
        "013": "sha256:6bdf26ab16b4c73cfbdf2203977185094e14e36c47ec0d75e37e0daf3db3abf2",
        "018": "sha256:fe3603f4b5bde33721305d5c67bd037fc1b440a76bd58142f06ed806e8ffd78d",
        "short": "sha256:bccffdb3f0324a35e79725046422e0fa9b62d60dce9c35d4f4065ef728714449",
    }
    configs = {
        "base": BASE_COMPRESSION_CONFIG,
        "001": EXPERIMENT_001_CONFIG,
        "002": EXPERIMENT_002_CONFIG,
        "003": EXPERIMENT_003_CONFIG,
        "004": EXPERIMENT_004_CONFIG,
        "005": EXPERIMENT_005_CONFIG,
        "006": EXPERIMENT_006_CONFIG,
        "007": EXPERIMENT_007_CONFIG,
        "008": EXPERIMENT_008_CONFIG,
        "011": EXPERIMENT_011_CONFIG,
        "013": EXPERIMENT_013_CONFIG,
        "018": EXPERIMENT_018_CONFIG,
        "short": LEGACY_SHORT_DECODE_CONFIG,
    }

    assert {name: config_hash(config) for name, config in configs.items()} == expected
