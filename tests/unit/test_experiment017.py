from dataclasses import replace

from tests.support.experiments import load_experiment


def test_experiment017_changes_only_model_sensitivity_and_intent_from_experiment016() -> None:
    experiment016 = load_experiment(16)
    experiment017 = load_experiment(17)
    config016 = experiment016.config
    config017 = experiment017.config
    adjusted017 = replace(
        config017,
        model=config016.model,
        intent=config016.intent,
        output=config016.output,
        allocation=replace(
            config017.allocation,
            reconstruction=replace(
                config017.allocation.reconstruction,
                sensitivity_strength=config016.allocation.reconstruction.sensitivity_strength,
            ),
        ),
    )

    assert adjusted017 == config016
    assert config017.model.source == "google/gemma-3-1b-it"
    assert config017.model.revision == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    assert config017.allocation.reconstruction.sensitivity_strength == 0.5
    assert config017.factorization.shared_input.enabled is True
    assert experiment017.workflow.expected_blocks == 26
    assert config017.intent.baseline_run == "012-compress-and-benchmark-gemma-3-1b-it"
    upload = experiment017.workflow.export.huggingface
    assert upload is not None
    assert upload.repo_id == "gemma-3-1b-it-nanoquant-GGUF"
    assert upload.private is False
    assert upload.commit_message == "Publish NanoQuant Experiment 017"
