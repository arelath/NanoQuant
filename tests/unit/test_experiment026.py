from dataclasses import replace
from pathlib import Path

from tests.support.experiments import load_experiment


def test_experiment026_retargets_experiment025_settings_to_llama_3_2_3b() -> None:
    experiment025 = load_experiment(25)
    experiment026 = load_experiment(26)
    config025 = experiment025.config
    config026 = experiment026.config
    workflow025 = experiment025.workflow
    workflow026 = experiment026.workflow

    assert config026.model.source == "meta-llama/Llama-3.2-3B-Instruct"
    assert config026.model.revision == "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    assert config026.model.tokenizer_revision == config026.model.revision
    assert replace(
        config026,
        model=config025.model,
        intent=config025.intent,
        output=config025.output,
    ) == config025
    assert config026.intent.baseline_run == "025-compress-and-benchmark-llama-3-2-1b-instruct"
    assert replace(
        workflow026,
        export=workflow025.export,
        summary_output=workflow025.summary_output,
        quality_output=workflow025.quality_output,
        quality_markdown_output=workflow025.quality_markdown_output,
        expected_blocks=workflow025.expected_blocks,
    ) == workflow025
    assert workflow026.expected_blocks == 28
    assert workflow026.export.gguf_output == Path(
        "Results/026/llama-3-2-3b-instruct-nanoquant.gguf"
    )
    upload = workflow026.export.huggingface
    assert upload is not None
    assert upload.repo_id == "Llama-3.2-3B-Instruct-nanoquant-GGUF"
    assert upload.private is False
    assert upload.commit_message == "Publish NanoQuant Experiment 026"


def test_runpod_defaults_to_experiment026() -> None:
    bootstrap = Path("tools/runpod_bootstrap.sh").read_text(encoding="utf-8")
    experiment026_case = bootstrap.split("  026)", maxsplit=1)[1].split(";;", maxsplit=1)[0]

    assert 'EXPERIMENT="${NANOQUANT_EXPERIMENT:-026}"' in bootstrap
    assert 'MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"' in experiment026_case
    assert 'MODEL_REVISION="0cb88a4f764b7a12671c53f0838cd831a0843b95"' in experiment026_case
    assert (
        'LAUNCHER="experiments/026-compress-and-benchmark-llama-3-2-3b-instruct.py"'
        in experiment026_case
    )
    assert "REQUIRES_HF_WRITE=1" in experiment026_case
    assert "PREFLIGHT_CCE=1" in experiment026_case
