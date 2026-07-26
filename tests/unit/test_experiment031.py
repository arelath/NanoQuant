from pathlib import Path

from tests.support.experiments import load_experiment


def test_experiment031_scales_dual_mode_recipe_with_serial_deployment_quality() -> None:
    experiment030 = load_experiment(30)
    experiment031 = load_experiment(31)
    config030 = experiment030.config
    config031 = experiment031.config
    workflow031 = experiment031.workflow

    assert config031.intent.baseline_run == "030-recover-qwen3-0-6b-thinking-quality"
    assert config031.model.source == "Qwen/Qwen3-8B"
    assert config031.model.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert config031.dataset == config030.dataset
    assert config031.calibration == config030.calibration
    assert config031.evaluation.reasoning_modes == config030.evaluation.reasoning_modes
    assert config031.evaluation.default_tier.value == "full"
    assert workflow031.llamacpp_quality_parallel == 1
    assert workflow031.reasoning_sequence_length_override == 1024
    assert workflow031.export.huggingface is None
    assert workflow031.export.gguf_output == Path(
        "Results/031/qwen3-8b-teacher-dual-mode-exp031-nanoquant.gguf"
    )
