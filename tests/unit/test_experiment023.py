from pathlib import Path

import pytest

from nanoquant.config.schema import (
    AllocationStrategy,
    KlAllocationObjective,
    KlSensitivityGranularity,
    RankResponseSource,
)
from nanoquant.self_measured_d2_workflow import (
    SelfMeasuredD2ProfileOptions,
    _uniform_control_config,
)
from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment023_applies_interaction_corrected_tuned_d2_on_gemma_1b() -> None:
    config = load_experiment(23).config

    assert config.model.source == "google/gemma-3-1b-it"
    assert config.allocation.strategy is AllocationStrategy.KL_CALIBRATED
    assert config.allocation.kl_sensitivity_granularity is KlSensitivityGranularity.EXACT
    reconstruction = config.allocation.reconstruction
    assert reconstruction.kl_objective is KlAllocationObjective.INTERACTION_NORMALIZED_UNIT_KL
    assert reconstruction.response_source is RankResponseSource.MEASURED
    assert reconstruction.objective_mode == "calibration_weighted"
    assert reconstruction.sensitivity_strength == 1
    assert reconstruction.rank_trust_reference_run is None
    assert reconstruction.rank_trust_fraction == 1
    assert config.distillation.enabled is True


def test_experiment023_differs_from_experiment022_only_by_objective_and_intent() -> None:
    config022 = load_experiment(22).config
    config023 = load_experiment(23).config

    differences = config_diff_paths(config022, config023)
    # Ignore per-experiment identity and output-location paths, which necessarily
    # differ; the only semantic recipe difference must be the KL objective.
    semantic = {
        path
        for path in differences
        if not path.startswith("intent") and path != "output.run_root"
    }

    assert semantic == {"allocation.reconstruction.kl_objective"}


def test_profile_options_default_reproduces_static_12x512_profile() -> None:
    options = SelfMeasuredD2ProfileOptions()

    assert options.wikitext_samples == 12
    assert options.sequence_length == 512
    assert options.tuned_operating_point is False


def test_profile_options_reject_degenerate_dimensions() -> None:
    with pytest.raises(ValueError, match="dataset dimensions"):
        SelfMeasuredD2ProfileOptions(wikitext_samples=0)
    with pytest.raises(ValueError, match="dataset dimensions"):
        SelfMeasuredD2ProfileOptions(sequence_length=1)


def test_tuned_operating_point_keeps_control_distillation_enabled(tmp_path: Path) -> None:
    config = load_experiment(23).config

    tuned = _uniform_control_config(config, tmp_path, tuned_operating_point=True)
    static = _uniform_control_config(config, tmp_path, tuned_operating_point=False)

    # Both controls use uniform allocation; only the tuned control keeps global
    # distillation so the profile can be measured at the tuned operating point.
    assert tuned.allocation.strategy is AllocationStrategy.UNIFORM
    assert static.allocation.strategy is AllocationStrategy.UNIFORM
    assert tuned.distillation.enabled is True
    assert static.distillation.enabled is False
