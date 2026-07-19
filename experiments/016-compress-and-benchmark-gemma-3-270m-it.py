"""Experiment 016: refined down/edge reconstruction priorities for Gemma 3 270M."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

PARENT = ExperimentRef(15, "compress-and-benchmark-gemma-3-270m-it")

CONFIG = replace(
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=16,
        name="compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Refine the architecture-protected fixed-budget allocation by increasing only "
            "down-projection importance and slightly increasing first/last-block importance."
        ),
        hypothesis=(
            "A 1.50 down-projection weight, 1.30 edge-block weight, and 0.75 sensitivity "
            "strength allocate more rank to the remaining high-error protected units and improve "
            "quality relative to Experiment 015."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "shared-input-qkv",
            "reconstruction-aware-ranks",
            "architecture-protected-ranks",
            "down-projection-priority",
            "edge-block-protection",
            "wikitext2",
            "ultrachat",
        ),
    ),
    CONFIG,
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
