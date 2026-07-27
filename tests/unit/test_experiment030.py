from pathlib import Path

from nanoquant.config.schema import ReasoningMode
from tests.support.experiments import load_experiment


def test_experiment030_is_additive_dual_mode_recovery_canary() -> None:
    experiment = load_experiment(30)
    config = experiment.config
    workflow = experiment.workflow

    assert config.intent.baseline_run == "028-compress-and-benchmark-qwen3-0-6b"
    assert config.model.source == "Qwen/Qwen3-0.6B"
    assert config.calibration.sample_count == 528
    assert tuple(item.mode for item in config.dataset.behavior_slices) == (
        ReasoningMode.RAW,
        ReasoningMode.NON_THINKING,
        ReasoningMode.THINKING,
    )
    assert tuple(item.target_valid_token_fraction for item in config.dataset.behavior_slices) == (
        0.25,
        0.25,
        0.50,
    )
    assert config.dataset.behavior_slices[0].minimum_valid_tokens == 256 // 2 * 2048
    assert config.dataset.behavior_slices[1].minimum_valid_tokens == 256 // 2 * 2048
    assert config.dataset.behavior_slices[2].minimum_valid_tokens == 256 // 2 * 2048
    assert all(
        item.source.name == "HuggingFaceH4/ultrachat_200k"
        and item.teacher_trace_generation is not None
        for item in config.dataset.behavior_slices[1:]
    )
    assert all(
        item.teacher_trace_generation is not None
        and item.teacher_trace_generation.implementation == "hf-greedy-qwen3-v1"
        for item in config.dataset.behavior_slices[1:]
    )
    assert config.evaluation.reasoning_modes == (
        ReasoningMode.THINKING,
        ReasoningMode.NON_THINKING,
    )
    assert config.evaluation.reasoning_sequence_length == 512
    assert workflow.reasoning_sequence_length_override == 1024
    assert config.distillation.enabled
    assert workflow.llamacpp_quality
    assert workflow.export.huggingface is None
    assert workflow.export.gguf_output == Path(
        "Results/030/qwen3-0-6b-teacher-dual-mode-exp030-nanoquant.gguf"
    )


def test_runpod_supports_experiment030_without_prepromotion_upload() -> None:
    bootstrap = Path("tools/runpod_bootstrap.sh").read_text(encoding="utf-8")
    case = bootstrap.split("  030)", maxsplit=1)[1].split(";;", maxsplit=1)[0]
    assert 'MODEL_ID="Qwen/Qwen3-0.6B"' in case
    assert 'LAUNCHER="experiments/030-recover-qwen3-0-6b-thinking-quality.py"' in case
    assert "REQUIRES_HF_WRITE=1" not in case
