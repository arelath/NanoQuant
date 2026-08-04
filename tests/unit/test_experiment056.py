from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment056_replays_055_with_physical_rank_redistribution() -> None:
    baseline = load_experiment(55)
    candidate = load_experiment(56)

    assert candidate.identity.baseline is not None
    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert candidate.config.allocation.target_bpw == baseline.config.allocation.target_bpw
    assert candidate.config.allocation.bounds.ceiling_fraction_of_uniform == 1.5
    assert candidate.config.allocation.bounds.overcomplete_rank_ceiling_fraction == 1.0
    assert candidate.config.outliers == baseline.config.outliers
    assert candidate.config.factorization == baseline.config.factorization
    assert candidate.config.block_tuning == baseline.config.block_tuning
    assert candidate.config.distillation == baseline.config.distillation
    assert candidate.workflow.task_limit == 1000
    assert candidate.workflow.interrupt_after_block_commits is None
    assert config_diff_paths(baseline.config, candidate.config) == {
        "allocation.bounds.overcomplete_rank_ceiling_fraction",
        "intent.baseline_run",
        "intent.experiment_number",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "output.run_root",
    }
