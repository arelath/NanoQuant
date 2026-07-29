from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment032_changes_only_identity_output_and_fisher_shrinkage_from_022() -> None:
    baseline = load_experiment(22)
    candidate = load_experiment(32)

    assert candidate.config.calibration.shrinkage == 0.0
    assert baseline.config.calibration.shrinkage == 0.6
    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert config_diff_paths(baseline.config, candidate.config) == {
        "calibration.shrinkage",
        "intent.baseline_run",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "intent.experiment_number",
        "output.run_root",
    }
