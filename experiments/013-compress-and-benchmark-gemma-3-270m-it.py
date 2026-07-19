"""Experiment 013: shared-input QKV compression and quality on Gemma 3 270M."""

from recipes import (
    GEMMA_3_270M_STACKED_QKV_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

PARENT = ExperimentRef(10, "compress-and-benchmark-gemma-3-270m-it")

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=13,
        name="compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Compress pinned Gemma 3 270M with one shared-input QKV factorization per block "
            "and measure the complete BF16-versus-NanoQuant quality benchmark."
        ),
        hypothesis=(
            "Sharing the QKV input basis lowers attention reconstruction error at the same "
            "planned bit budget and improves quality relative to Experiment 010."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "shared-input-qkv",
            "stacked-factorization",
            "fixed-rank-plan",
            "wikitext2",
            "ultrachat",
        ),
    ),
    GEMMA_3_270M_STACKED_QKV_COMPRESSION_TEMPLATE,
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
