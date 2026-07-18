"""Experiment 011: test twice as many INT8 outliers on pinned Gemma 3 1B."""

from recipes import (
    BASE_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
)
from recipes._delta import config_delta

from nanoquant.compression_quality_workflow import run_compression_quality_experiment
from nanoquant.config.schema import DType

PARENT = ExperimentRef(6, "compress-and-benchmark-gemma-3-1b-it")

EXPERIMENT_TEMPLATE = config_delta(
    BASE_COMPRESSION_TEMPLATE,
    outliers=config_delta(
        BASE_COMPRESSION_TEMPLATE.outliers,
        fraction=0.002,
        storage_dtype=DType.INT8,
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=11,
        name="compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Measure the quality and storage effect of retaining twice as many residual outliers "
            "as Experiment 006 while quantizing their values to INT8."
        ),
        hypothesis=(
            "Doubling residual outlier coverage from 0.1% to 0.2% improves reconstruction and "
            "quality enough to offset the precision lost by storing outlier values as INT8."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "double-outliers",
            "int8-outliers",
            "wikitext2",
            "ultrachat",
        ),
    ),
    EXPERIMENT_TEMPLATE,
    expected_blocks=26,
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
