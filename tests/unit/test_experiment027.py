from dataclasses import replace
from pathlib import Path

from tests.support.experiments import load_experiment


def test_experiment027_retargets_experiment025_settings_to_meta_llama_3_8b() -> None:
    experiment025 = load_experiment(25)
    experiment027 = load_experiment(27)
    config025 = experiment025.config
    config027 = experiment027.config
    workflow025 = experiment025.workflow
    workflow027 = experiment027.workflow

    assert config027.model.source == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert config027.model.revision == "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    assert config027.model.tokenizer_revision == config027.model.revision
    assert replace(
        config027,
        model=config025.model,
        intent=config025.intent,
        output=config025.output,
    ) == config025
    assert config027.intent.baseline_run == "025-compress-and-benchmark-llama-3-2-1b-instruct"
    assert replace(
        workflow027,
        export=workflow025.export,
        summary_output=workflow025.summary_output,
        quality_output=workflow025.quality_output,
        quality_markdown_output=workflow025.quality_markdown_output,
        expected_blocks=workflow025.expected_blocks,
    ) == workflow025
    assert workflow027.expected_blocks == 32
    assert workflow027.export.gguf_output == Path(
        "Results/027/meta-llama-3-8b-instruct-nanoquant.gguf"
    )
    upload = workflow027.export.huggingface
    assert upload is not None
    assert upload.repo_id == "Meta-Llama-3-8B-Instruct-nanoquant-GGUF"
    assert upload.private is False
    assert upload.commit_message == "Publish NanoQuant Experiment 027"


def test_runpod_supports_experiment027() -> None:
    bootstrap = Path("tools/runpod_bootstrap.sh").read_text(encoding="utf-8")
    experiment027_case = bootstrap.split("  027)", maxsplit=1)[1].split(";;", maxsplit=1)[0]

    assert 'MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"' in experiment027_case
    assert 'MODEL_REVISION="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"' in experiment027_case
    assert (
        'LAUNCHER="experiments/027-compress-and-benchmark-meta-llama-3-8b-instruct.py"'
        in experiment027_case
    )
    assert "REQUIRES_HF_WRITE=1" in experiment027_case
    assert "PREFLIGHT_CCE=1" in experiment027_case
