from dataclasses import replace
from pathlib import Path

from tests.support.experiments import load_experiment


def test_experiment019_changes_only_llama_identity_and_architecture_contracts() -> None:
    experiment018 = load_experiment(18)
    experiment019 = load_experiment(19)
    config018 = experiment018.config
    config019 = experiment019.config
    workflow = experiment019.workflow

    assert config019.model.source == "meta-llama/Llama-3.2-1B-Instruct"
    assert config019.model.revision == "9213176726f574b556790deb65791e0c5aa438b6"
    assert config019.model.tokenizer_revision == config019.model.revision
    assert replace(
        config019,
        model=config018.model,
        intent=config018.intent,
        output=config018.output,
    ) == config018
    assert workflow.expected_blocks == 16
    assert workflow.maximum_wddm_shared_gib == 0.75
    assert workflow.restore_completed_blocks is False
    assert workflow.quality_backend == "dense"
    assert workflow.export.runtime_family == "llama"
    assert workflow.export.gguf_output == Path("Results/019/llama-3-2-1b-instruct-nanoquant.gguf")
    assert config019.intent.baseline_run == "none:first NanoQuant experiment for the Llama architecture"
    upload = workflow.export.huggingface
    assert upload is not None
    assert upload.repo_id == "Llama-3.2-1B-Instruct-nanoquant-GGUF"
    assert upload.private is False
    assert upload.commit_message == "Publish NanoQuant Experiment 019"
