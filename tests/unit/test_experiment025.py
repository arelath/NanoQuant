from dataclasses import replace
from pathlib import Path

from tests.support.experiments import load_experiment


def test_experiment025_repeats_experiment019_on_the_current_pipeline() -> None:
    experiment019 = load_experiment(19)
    experiment025 = load_experiment(25)
    config019 = experiment019.config
    config025 = experiment025.config
    workflow = experiment025.workflow

    assert config025.model.source == "meta-llama/Llama-3.2-1B-Instruct"
    assert config025.model.revision == "9213176726f574b556790deb65791e0c5aa438b6"
    assert config025.model.tokenizer_revision == config025.model.revision
    assert replace(
        config025,
        intent=config019.intent,
        output=config019.output,
    ) == config019
    assert config025.intent.baseline_run == "019-compress-and-benchmark-llama-3-2-1b-instruct"
    assert workflow.expected_blocks == 16
    assert workflow.maximum_wddm_shared_gib == 0.75
    assert workflow.restore_completed_blocks is False
    assert workflow.quality_backend == "dense"
    assert workflow.export.runtime_family == "llama"
    assert workflow.export.gguf_output == Path(
        "Results/025/llama-3-2-1b-instruct-nanoquant.gguf"
    )
    upload = workflow.export.huggingface
    assert upload is not None
    assert upload.repo_id == "Llama-3.2-1B-Instruct-nanoquant-GGUF"
    assert upload.private is False
    assert upload.commit_message == "Publish NanoQuant Experiment 025"


def test_runpod_defaults_to_experiment025() -> None:
    bootstrap = Path("tools/runpod_bootstrap.sh").read_text(encoding="utf-8")
    experiment025_case = bootstrap.split("  025)", maxsplit=1)[1].split(";;", maxsplit=1)[0]

    assert 'EXPERIMENT="${NANOQUANT_EXPERIMENT:-025}"' in bootstrap
    assert 'MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"' in experiment025_case
    assert 'MODEL_REVISION="9213176726f574b556790deb65791e0c5aa438b6"' in experiment025_case
    assert (
        'LAUNCHER="experiments/025-compress-and-benchmark-llama-3-2-1b-instruct.py"'
        in experiment025_case
    )
    assert "REQUIRES_HF_WRITE=1" in experiment025_case
    assert "PREFLIGHT_CCE=1" in experiment025_case
