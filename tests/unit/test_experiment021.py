from dataclasses import replace

from nanoquant.config.schema import AllocationStrategy, KlSensitivityGranularity
from tests.support.experiments import load_experiment


def test_experiment021_is_d2_only_and_quality_matched_to_experiment016() -> None:
    experiment016 = load_experiment(16)
    experiment021 = load_experiment(21)
    config016 = experiment016.config
    config021 = experiment021.config

    assert config021.allocation.strategy is AllocationStrategy.KL_CALIBRATED
    assert config021.allocation.kl_profile_artifact == "evidence/020/016-kl-budget-profile-v3"
    assert config021.allocation.kl_profile_key == (
        "sha256:e62295bb78c07fa7435560f7ba2463e2bbb7164e36963ab61dac9972b2f5a324"
    )
    assert (
        config021.allocation.kl_sensitivity_granularity
        is KlSensitivityGranularity.EXACT_OR_TYPE_BLOCK
    )
    assert config021.allocation.reconstruction.rank_trust_reference_run == (
        "evidence/016/016-compress-and-benchmark-gemma-3-270m-it"
    )
    assert config021.allocation.reconstruction.rank_trust_fraction == 0.25

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
