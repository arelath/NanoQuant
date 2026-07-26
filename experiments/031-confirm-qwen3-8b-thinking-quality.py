"""Experiment 031: confirm the accepted dual-mode recipe on Qwen3 8B."""

from recipes import (
    QWEN_3_8B_DUAL_MODE_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_main,
)

BASELINE = ExperimentRef(30, "recover-qwen3-0-6b-thinking-quality")

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=31,
        name="confirm-qwen3-8b-thinking-quality",
        purpose=(
            "Confirm Experiment 030's mode-aware data and distillation recipe on the pinned "
            "Qwen3 8B model with deployment-authoritative serial llama.cpp scoring."
        ),
        hypothesis=(
            "The additive dual-mode recipe scales to Qwen3 8B at the retained bit budget and "
            "passes independent thinking, non-thinking, generic-quality, artifact, memory, and "
            "resume gates without changing the source chat-template default."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "qwen3-8b",
            "thinking-quality-recovery",
            "dual-mode",
            "openr1-math-220k",
            "record-aware-packing",
            "masked-distillation",
            "gguf",
            "confirmation",
            "llamacpp-serial-quality",
        ),
    ),
    QWEN_3_8B_DUAL_MODE_COMPRESSION_TEMPLATE,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend=None,
    llamacpp_quality=True,
    llamacpp_quality_parallel=1,
    reasoning_sequence_length_override=1024,
    export=CompressionExportPolicy(
        release_name="qwen3-8b-dual-mode-exp031",
        runtime_family="qwen",
    ),
)


experiment_main(__name__, __file__, EXPERIMENT)
