"""Experiment 010: repeat the pinned Gemma 3 270M compression-quality baseline."""

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
            "Repeat Experiment 009's pinned Gemma 3 270M compression and quality workflow "
            "without publishing the result."
        ),
        hypothesis=(
            "The baseline recipe reproduces Experiment 009's compression behavior and enables "
            "a local quality comparison."
        ),
        baseline=BaselineRef.external("bf16-unsloth-gemma-3-270m-it"),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "gguf",
            "baseline-repeat",
            "cubic-schedule",
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
