from dataclasses import replace
from pathlib import Path

from tests.support.experiments import load_experiment


def test_experiment029_retargets_all_experiment028_settings_to_qwen3_8b() -> None:
    experiment028 = load_experiment(28)
    experiment029 = load_experiment(29)
    config028 = experiment028.config
    config029 = experiment029.config
    workflow028 = experiment028.workflow
    workflow029 = experiment029.workflow

    assert config029.model.source == "Qwen/Qwen3-8B"
    assert config029.model.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert config029.model.tokenizer_revision == config029.model.revision
    assert replace(
        config029,
        model=config028.model,
        intent=config028.intent,
        output=config028.output,
    ) == config028
    assert config029.intent.baseline_run == "028-compress-and-benchmark-qwen3-0-6b"
    assert (
        replace(
            workflow029,
            export=workflow028.export,
            summary_output=workflow028.summary_output,
            quality_output=workflow028.quality_output,
            quality_markdown_output=workflow028.quality_markdown_output,
            expected_blocks=workflow028.expected_blocks,
        )
        == workflow028
    )
    assert workflow029.expected_blocks == 36
    assert workflow029.quality_backend is None
    assert workflow029.llamacpp_quality is True
    assert workflow029.export.runtime_family == "qwen"
    assert workflow029.export.gguf_output == Path("Results/029/qwen3-8b-nanoquant.gguf")
    upload = workflow029.export.huggingface
    assert upload is not None
    assert upload.repo_id == "Qwen3-8B-nanoquant-GGUF"
    assert upload.private is False
    assert upload.commit_message == "Publish NanoQuant Experiment 029"


def test_runpod_defaults_to_and_supports_experiment029() -> None:
    bootstrap = Path("tools/runpod_bootstrap.sh").read_text(encoding="utf-8")
    experiment029_case = bootstrap.split("  029)", maxsplit=1)[1].split(";;", maxsplit=1)[0]

    assert 'EXPERIMENT="${NANOQUANT_EXPERIMENT:-029}"' in bootstrap
    assert 'MODEL_ID="Qwen/Qwen3-8B"' in experiment029_case
    assert 'MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218"' in experiment029_case
    assert 'LAUNCHER="experiments/029-compress-and-benchmark-qwen3-8b.py"' in experiment029_case
    assert "REQUIRES_HF_WRITE=1" in experiment029_case
    assert "PREFLIGHT_CCE=1" in experiment029_case
    assert (
        'LLAMA_CPP_REPOSITORY="${NANOQUANT_LLAMA_CPP_REPOSITORY:-'
        'https://github.com/arelath/llama.cpp.git}"'
    ) in bootstrap
    assert (
        'LLAMA_CPP_REVISION="${NANOQUANT_LLAMA_CPP_REVISION:-nanoquants}"'
        in bootstrap
    )
    assert "-DGGML_CUDA=ON" in bootstrap
    assert "tools/build_llamacpp_quality.py" in bootstrap
