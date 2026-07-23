from nanoquant.config.schema import (
    AllocationStrategy,
    KlAllocationObjective,
    KlSensitivityGranularity,
    RankResponseSource,
)
from tests.support.experiments import load_experiment


def test_experiment024_combines_best_retained_1b_quality_methods() -> None:
    experiment = load_experiment(24)
    config = experiment.config

    assert config.model.source == "google/gemma-3-1b-it"
    assert config.model.revision == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    assert config.intent.baseline_run == "022-d2-kl-compress-and-benchmark-gemma-3-1b-it"
    assert config.allocation.strategy is AllocationStrategy.KL_CALIBRATED
    assert config.allocation.kl_sensitivity_granularity is KlSensitivityGranularity.EXACT
    reconstruction = config.allocation.reconstruction
    assert reconstruction.kl_objective is KlAllocationObjective.MEASURED_UNIT_KL
    assert reconstruction.response_source is RankResponseSource.MEASURED
    assert reconstruction.objective_mode == "calibration_weighted"
    assert reconstruction.response_curves == ()
    assert reconstruction.rank_trust_reference_run is None
    assert reconstruction.rank_trust_fraction == 1

    shared = config.factorization.shared_input
    assert shared.enabled
    assert len(shared.groups) == 1
    assert shared.groups[0].member_multipliers[0].member == "self_attn.v_proj"
    assert shared.groups[0].member_multipliers[0].multiplier == 2
    assert config.factorization.bias_correction.enabled is False
    assert config.factorization.low_rank_patch.enabled is False
    assert config.distillation.enabled is True

    assert experiment.workflow.expected_blocks == 26
    assert experiment.workflow.maximum_wddm_shared_gib == 0.75
    assert experiment.workflow.task_limit == 1000
    assert experiment.workflow.local_files_only is True
