"""Experiment 014: reconstruction-informed stacked-QKV Gemma 3 270M compression."""

from recipes import (
    GEMMA_3_270M_RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

PARENT = ExperimentRef(13, "compress-and-benchmark-gemma-3-270m-it")

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=14,
        name="compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Probe every stacked-QKV/ordinary physical unit before fitting, allocate the fixed "
            "Gemma 3 270M rank budget from measured reconstruction error, and run the complete "
            "quality benchmark."
        ),
        hypothesis=(
            "Reconstruction-informed fixed-budget ranks reduce error in the protected sensitive "
            "cohort and improve quality relative to Experiment 013."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "shared-input-qkv",
            "reconstruction-aware-ranks",
            "full-model-rank-probe",
            "fixed-rank-plan",
            "wikitext2",
            "ultrachat",
        ),
    ),
    GEMMA_3_270M_RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE,
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
