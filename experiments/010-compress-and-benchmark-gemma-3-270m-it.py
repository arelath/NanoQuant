"""Experiment 010: test 1,600-iteration cubic ADMM on pinned Gemma 3 270M."""

from recipes import (
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    define_compression_quality_experiment,
)
from recipes._delta import config_delta

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

_TEMPLATE = config_delta(
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    factorization=config_delta(
        GEMMA_3_270M_COMPRESSION_TEMPLATE.factorization,
        admm=config_delta(
            GEMMA_3_270M_COMPRESSION_TEMPLATE.factorization.admm,
            outer_iterations=1600,
        ),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=10,
        name="compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Measure whether doubling cubic-schedule ADMM from 800 to 1,600 iterations improves "
            "Experiment 009's pinned Gemma 3 270M compression and quality results."
        ),
        hypothesis=(
            "Giving cubic ADMM 1,600 outer iterations lowers layer and block fitting error without "
            "changing the rank, outlier, tuning, export, or evaluation policies."
        ),
        baseline=BaselineRef.external("bf16-unsloth-gemma-3-270m-it"),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "gguf",
            "admm-1600",
            "cubic-schedule",
            "wikitext2",
            "ultrachat",
        ),
    ),
    _TEMPLATE,
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
