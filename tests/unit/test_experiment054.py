from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment054_changes_only_identity_output_and_binary_rate_from_024() -> None:
    baseline = load_experiment(24)
    candidate = load_experiment(54)

    rates = candidate.config.block_tuning.factorized.learning_rates
    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert rates.binary == 3e-5
    assert rates.scale == 1e-5
    assert rates.outlier is None
    assert rates.bias == 1e-5
    assert config_diff_paths(baseline.config, candidate.config) == {
        "block_tuning.factorized.learning_rates.binary",
        "intent.baseline_run",
        "intent.experiment_number",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "output.run_root",
    }
