"""Experiment 030: recover Qwen3 0.6B thinking and non-thinking quality."""

from recipes import (
    QWEN_3_0_6B_TEACHER_TRACE_COMPRESSION_TEMPLATE,
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
            "additive raw mixture plus complete teacher-generated thinking and non-thinking "
            "responses over pinned UltraChat prompts."
        ),
        hypothesis=(
            "A teacher-coherent 25/25/50 valid-token mixture plus assistant-targeted global "
            "distillation preserves both source chat modes while closing the thinking-mode "
            "response-token NLL gap observed in Experiment 028 without math specialization."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "qwen3-0-6b",
            "thinking-quality-recovery",
            "dual-mode",
            "ultrachat-200k",
            "teacher-generated-thinking",
            "teacher-generated-non-thinking",
            "record-aware-packing",
            "masked-distillation",
            "gguf",
            "canary",
        ),
    ),
    QWEN_3_0_6B_TEACHER_TRACE_COMPRESSION_TEMPLATE,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend=None,
    llamacpp_quality=True,
    reasoning_sequence_length_override=1024,
    export=CompressionExportPolicy(
        release_name="qwen3-0-6b-teacher-dual-mode-exp030",
        runtime_family="qwen",
    ),
)


experiment_main(__name__, __file__, EXPERIMENT)
