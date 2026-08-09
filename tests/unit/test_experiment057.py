from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment057_changes_only_056_product_codebook_encoding_policy() -> None:
    baseline = load_experiment(56)
    candidate = load_experiment(57)

    assert candidate.identity.baseline is not None
    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert candidate.config.factorization.product_codebook.enabled
    assert candidate.config.factorization.product_codebook.index_bits == 16
    assert candidate.config.factorization.product_codebook.flips_per_word == 0
    assert candidate.config.factorization.product_codebook.measured_option_allocation
    assert candidate.config.factorization.product_codebook.allow_free_factor_fallback
    assert candidate.config.factorization.product_codebook.layer_patterns == ("mlp.*",)
    assert candidate.workflow.task_limit == baseline.workflow.task_limit == 1000
    assert config_diff_paths(baseline.config, candidate.config) == {
        "factorization.implementation",
        "factorization.product_codebook.enabled",
        "intent.baseline_run",
        "intent.experiment_number",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "output.run_root",
    }
