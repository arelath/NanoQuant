from dataclasses import dataclass

import pytest
from recipes import (
    BASE_COMPRESSION_TEMPLATE,
    LARGE_MODEL_COMPRESSION_TEMPLATE,
)
from recipes._delta import config_delta, run_config_defaults

from nanoquant.config.codec import config_hash
from tests.support.experiments import load_experiment


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


def test_large_model_template_pins_bounded_memory_guards() -> None:
    config = LARGE_MODEL_COMPRESSION_TEMPLATE

    assert config.runtime.executor.value == "cpu_offload"
    assert config.calibration.method.value == "forward_only"
    assert config.runtime.activations.gpu_cache.value == "auto"
    assert config.runtime.activations.gpu_reserve_gib == 4.0
    assert not config.evaluation.inline_quality
    assert not config.distillation.enabled


def test_templates_are_unnumbered_and_concrete_configs_are_numbered() -> None:
    assert BASE_COMPRESSION_TEMPLATE.intent.experiment_number is None
    assert BASE_COMPRESSION_TEMPLATE.intent.name == "unnamed-run"

    definitions = tuple(load_experiment(number) for number in range(1, 13))
    assert tuple(definition.identity.number for definition in definitions) == tuple(range(1, 13))
    assert all(
        definition.config.intent.name == definition.identity.canonical_name
        for definition in definitions
    )
    assert all(
        definition.config.allocation.maximum_rank_layer_patterns
        == ("self_attn.v_proj", "self_attn.k_proj")
        for definition in definitions
    )
    assert all(
        tuple(
            (item.pattern, item.multiplier)
            for item in definition.config.allocation.layer_budget_multipliers
        )
        == (("self_attn.q_proj", 1.25),)
        for definition in definitions
    )
    assert len({config_hash(definition.config) for definition in definitions}) == 12
