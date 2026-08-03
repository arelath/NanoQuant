from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment055_is_an_equal_budget_overcomplete_replay_of_054() -> None:
    baseline = load_experiment(54)
    candidate = load_experiment(55)

    assert candidate.identity.baseline is not None
    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert candidate.config.allocation.target_bpw == baseline.config.allocation.target_bpw
    assert candidate.config.allocation.bounds.ceiling_fraction_of_uniform == 1.5
    assert candidate.config.allocation.bounds.overcomplete_rank_ceiling_fraction == 1.5
    assert candidate.config.allocation.kl_profile_artifact == (
        "evidence/054/054-d2-uniform-control-kl-profile"
    )
    assert candidate.config.block_tuning.factorized.learning_rates.binary == 3e-5
    assert candidate.workflow.task_limit == 1000
    assert candidate.workflow.interrupt_after_block_commits == 1
    assert config_diff_paths(baseline.config, candidate.config) == {
        "allocation.bounds.ceiling_fraction_of_uniform",
        "allocation.bounds.overcomplete_rank_ceiling_fraction",
        "allocation.kl_profile_artifact",
        "allocation.kl_profile_key",
        "intent.baseline_run",
        "intent.experiment_number",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "output.run_root",
    }
