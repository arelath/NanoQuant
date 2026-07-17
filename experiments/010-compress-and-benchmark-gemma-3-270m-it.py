"""Experiment 010: repeat Experiment 009 without Hugging Face publication."""

from recipes import (
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=10,
        name="compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Repeat Experiment 009's pinned Gemma 3 270M compression and complete "
            "BF16-versus-NanoQuant quality benchmark without publishing artifacts to Hugging Face."
        ),
        hypothesis=(
            "The promoted attention-rank recipe reproduces Experiment 009's validated 270M "
            "NanoQuant GGUF and measured WikiText-2 and common-task quality without requiring "
            "external publication."
        ),
        baseline=BaselineRef.external("bf16-unsloth-gemma-3-270m-it"),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "gguf",
            "wikitext2",
            "ultrachat",
        ),
    ),
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    expected_blocks=18,
    maximum_wddm_shared_gib=0.75,
)


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
