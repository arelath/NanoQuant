from pathlib import Path

from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment010_uses_baseline_rank_and_admm_policy_without_huggingface_publication() -> None:
    previous = load_experiment(9)
    definition = load_experiment(10)
    config = definition.config
    experiment = definition.workflow

    assert config_diff_paths(previous.config, config) == {
        "intent.experiment_number",
        "intent.name",
        "intent.purpose",
        "intent.hypothesis",
        "intent.tags",
        "output.run_root",
    }
    assert definition.identity.canonical_name == (
        "010-compress-and-benchmark-gemma-3-270m-it"
    )
    assert config.model.source == "unsloth/gemma-3-270m-it"
    assert config.model.revision == "23cf460f6bb16954176b3ddcc8d4f250501458a9"
    assert config.output.run_root == "evidence/010"
    assert config.factorization.admm.outer_iterations == (
        previous.config.factorization.admm.outer_iterations
    ) == 800
    assert config.factorization.admm.penalty_schedule == (
        previous.config.factorization.admm.penalty_schedule
    ) == "cubic"
    assert experiment.expected_blocks == previous.workflow.expected_blocks == 18
    assert experiment.maximum_wddm_shared_gib == previous.workflow.maximum_wddm_shared_gib == 0.75
    assert experiment.quality_backend == previous.workflow.quality_backend == "factorized"
    assert experiment.wikitext_samples == previous.workflow.wikitext_samples
    assert experiment.wikitext_batch_size == previous.workflow.wikitext_batch_size == 8
    assert experiment.task_names == previous.workflow.task_names
    assert experiment.task_limit == previous.workflow.task_limit
    assert experiment.task_batch_size == previous.workflow.task_batch_size == 4
    assert experiment.summary_output == Path(
        "outputs/010/010-compress-and-benchmark-gemma-3-270m-it-summary.json"
    )
    assert experiment.quality_markdown_output == Path(
        "Results/010/010-compress-and-benchmark-gemma-3-270m-it-quality.md"
    )
    assert experiment.export.gguf_output == Path(
        "Results/010/gemma-3-270m-it-nanoquant.gguf"
    )
    assert previous.workflow.export.huggingface is not None
    assert experiment.export.huggingface is None
