from dataclasses import replace
from pathlib import Path

import pytest

from nanoquant.config.schema import (
    AllocationStrategy,
    KlAllocationObjective,
    KlSensitivityGranularity,
    RankResponseSource,
)
from nanoquant.experiment021_workflow import run_experiment021
from tests.support.experiments import load_experiment


def test_experiment021_is_d2_only_and_quality_matched_to_experiment016() -> None:
    experiment016 = load_experiment(16)
    experiment021 = load_experiment(21)
    config016 = experiment016.config
    config021 = experiment021.config

    assert config021.allocation.strategy is AllocationStrategy.KL_CALIBRATED
    assert config021.allocation.kl_profile_artifact == "runtime-kl-profile-required"
    assert config021.allocation.kl_profile_key == "runtime-kl-profile-key-required"
    assert (
        config021.allocation.kl_sensitivity_granularity
        is KlSensitivityGranularity.EXACT
    )
    reconstruction = config021.allocation.reconstruction
    assert reconstruction.rank_trust_reference_run is None
    assert reconstruction.rank_trust_fraction == 1
    assert reconstruction.response_source is RankResponseSource.MEASURED
    assert reconstruction.response_curves == ()
    assert reconstruction.objective_mode == "calibration_weighted"
    assert reconstruction.kl_objective is KlAllocationObjective.MEASURED_UNIT_KL
    assert reconstruction.sensitivity_strength == 1
    assert reconstruction.protect_sensitive_units is False
    assert reconstruction.importance.layer_multipliers == ()
    assert reconstruction.importance.protected_layer_patterns == ()

    # The compression recipe differs from Experiment 016 only in the D2
    # allocation inputs. In particular, factorization and global tuning match.
    adjusted021 = replace(
        config021,
        intent=config016.intent,
        output=config016.output,
        allocation=config016.allocation,
    )
    assert adjusted021 == config016
    assert config021.distillation.enabled is True

    workflow016 = experiment016.workflow
    workflow021 = experiment021.workflow
    assert (
        workflow021.wikitext_samples,
        workflow021.wikitext_sequence_length,
        workflow021.wikitext_batch_size,
        workflow021.task_names,
        workflow021.task_limit,
        workflow021.task_batch_size,
        workflow021.quality_backend,
    ) == (
        workflow016.wikitext_samples,
        workflow016.wikitext_sequence_length,
        workflow016.wikitext_batch_size,
        workflow016.task_names,
        workflow016.task_limit,
        workflow016.task_batch_size,
        workflow016.quality_backend,
    )
    assert config021.intent.baseline_run == "016-compress-and-benchmark-gemma-3-270m-it"
    assert config021.allocation.target_bpw == config016.allocation.target_bpw


def test_experiment021_rejects_profiles_from_previous_campaigns(tmp_path: Path) -> None:
    experiment = load_experiment(21)
    launcher = tmp_path / "repo" / "experiments" / "021.py"

    with pytest.raises(ValueError, match="inside its own campaign root"):
        run_experiment021(
            experiment,
            launcher_path=launcher,
            arguments=[
                "--kl-profile",
                str(tmp_path / "evidence" / "020" / "old-profile"),
                "--kl-control-run",
                str(tmp_path / "evidence" / "020" / "old-run"),
            ],
        )
