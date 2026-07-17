from pathlib import Path

from tests.support.experiments import load_experiment


def test_experiment009_runs_quality_and_public_huggingface_publication() -> None:
    definition = load_experiment(9)
    config = definition.config
    experiment = definition.workflow

    assert definition.identity.canonical_name == (
        "009-compress-benchmark-and-publish-gemma-3-270m-it"
    )
    assert config.model.source == "unsloth/gemma-3-270m-it"
    assert config.model.revision == "23cf460f6bb16954176b3ddcc8d4f250501458a9"
    assert config.output.run_root == "evidence/009"
    assert experiment.expected_blocks == 18
    assert experiment.quality_backend == "factorized"
    assert experiment.summary_output == Path(
        "outputs/009/009-compress-benchmark-and-publish-gemma-3-270m-it-summary.json"
    )
    assert experiment.quality_markdown_output == Path(
        "Results/009/009-compress-benchmark-and-publish-gemma-3-270m-it-quality.md"
    )
    assert experiment.export.gguf_output == Path(
        "outputs/009/gemma-3-270m-it-nanoquant.gguf"
    )
    assert experiment.export.huggingface is not None
    assert experiment.export.huggingface.repo_id == "gemma-3-270m-it-nanoquant-GGUF"
    assert experiment.export.huggingface.private is False
    assert experiment.export.huggingface.commit_message == "Publish NanoQuant Experiment 009"
