from dataclasses import replace
from pathlib import Path

from tests.support.experiments import load_experiment


def test_experiment028_retargets_experiment027_settings_to_qwen3_0_6b() -> None:
    experiment027 = load_experiment(27)
    experiment028 = load_experiment(28)
    config027 = experiment027.config
    config028 = experiment028.config
    workflow027 = experiment027.workflow
    workflow028 = experiment028.workflow

    assert config028.model.source == "Qwen/Qwen3-0.6B"
    assert config028.model.revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert config028.model.tokenizer_revision == config028.model.revision
    assert replace(
        config028,
        model=config027.model,
        intent=config027.intent,
        output=config027.output,
    ) == config027
    assert config028.intent.baseline_run == "027-compress-and-benchmark-meta-llama-3-8b-instruct"
    assert (
        replace(
            workflow028,
            export=workflow027.export,
            summary_output=workflow027.summary_output,
            quality_output=workflow027.quality_output,
            quality_markdown_output=workflow027.quality_markdown_output,
        )
        == workflow027
    )
    assert workflow028.quality_backend is None
    assert workflow028.llamacpp_quality is True
    assert workflow028.export.runtime_family == "qwen"
    assert workflow028.export.gguf_output == Path("Results/028/qwen3-0-6b-nanoquant.gguf")
    upload = workflow028.export.huggingface
    assert upload is not None
    assert upload.repo_id == "Qwen3-0.6B-nanoquant-GGUF"
    assert upload.private is False
    assert upload.commit_message == "Publish NanoQuant Experiment 028"


def test_runpod_supports_experiment028() -> None:
    bootstrap = Path("tools/runpod_bootstrap.sh").read_text(encoding="utf-8")
    experiment028_case = bootstrap.split("  028)", maxsplit=1)[1].split(";;", maxsplit=1)[0]

    assert 'MODEL_ID="Qwen/Qwen3-0.6B"' in experiment028_case
    assert 'MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"' in experiment028_case
    assert (
        'LAUNCHER="experiments/028-compress-and-benchmark-qwen3-0-6b.py"'
        in experiment028_case
    )
    assert "REQUIRES_HF_WRITE=1" in experiment028_case
    assert "PREFLIGHT_CCE=1" in experiment028_case
