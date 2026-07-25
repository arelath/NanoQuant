"""Experiment 030: recover Qwen3 0.6B thinking and non-thinking quality."""

from recipes import (
    QWEN_3_0_6B_DUAL_MODE_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_main,
)

BASELINE = ExperimentRef(28, "compress-and-benchmark-qwen3-0-6b")

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=30,
        name="recover-qwen3-0-6b-thinking-quality",
        purpose=(
            "Recover the pinned Qwen3 0.6B model's switchable thinking behavior with an "
            "additive raw, non-thinking, and complete-reasoning calibration mixture."
        ),
        hypothesis=(
            "A trace-complete 25/25/50 valid-token mixture plus assistant-targeted global "
            "distillation preserves non-thinking quality while closing the thinking-mode "
            "response-token NLL gap observed in Experiment 028."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "qwen3-0-6b",
            "thinking-quality-recovery",
            "dual-mode",
            "openr1-math-220k",
            "record-aware-packing",
            "masked-distillation",
            "gguf",
            "canary",
        ),
    ),
    QWEN_3_0_6B_DUAL_MODE_COMPRESSION_TEMPLATE,
    expected_blocks=28,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend=None,
    llamacpp_quality=True,
    export=CompressionExportPolicy(
        release_name="qwen3-0-6b-dual-mode-exp030",
        runtime_family="qwen",
    ),
)


experiment_main(__name__, __file__, EXPERIMENT)
