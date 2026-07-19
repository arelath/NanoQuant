"""Experiment 015: architecture-protected reconstruction ranks for Gemma 3 270M."""

from recipes import (
    GEMMA_3_270M_ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

PARENT = ExperimentRef(14, "compress-and-benchmark-gemma-3-270m-it")

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=15,
        name="compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Allocate the reconstruction-informed fixed rank budget while explicitly protecting "
            "Q/K/V/O/down projections and the first and last transformer blocks."
        ),
        hypothesis=(
            "Architectural importance priors move rank toward Q/K/V/O/down and edge blocks, "
            "lowering their reconstruction error and improving quality relative to Experiment 014."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "shared-input-qkv",
            "reconstruction-aware-ranks",
            "architecture-protected-ranks",
            "edge-block-protection",
            "wikitext2",
            "ultrachat",
        ),
    ),
    GEMMA_3_270M_ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
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
