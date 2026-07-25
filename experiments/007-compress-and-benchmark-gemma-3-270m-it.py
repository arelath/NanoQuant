"""Experiment 007: compress and quality-benchmark pinned Gemma 3 270M."""

from recipes import (
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    define_compression_quality_experiment,
    experiment_main,
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=7,
        name="compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Establish a complete Gemma 3 270M compression and quality benchmark using the promoted "
            "attention-projection allocation policy."
        ),
        hypothesis=(
            "The full-rank v_proj/k_proj and enlarged q_proj recipe remains effective at 270M scale "
            "after complete tuning and distillation."
        ),
        baseline=BaselineRef.external("bf16-unsloth-gemma-3-270m-it"),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "attention-rank",
            "wikitext2",
            "ultrachat",
        ),
    ),
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    expected_blocks=18,
    maximum_wddm_shared_gib=0.75,
)


experiment_main(__name__, __file__, EXPERIMENT)
