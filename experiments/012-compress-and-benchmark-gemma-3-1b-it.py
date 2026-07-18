"""Experiment 012: test ten times Experiment 011's INT8 outlier coverage."""

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

PARENT = ExperimentRef(11, "compress-and-benchmark-gemma-3-1b-it")

EXPERIMENT_TEMPLATE = config_delta(
    BASE_COMPRESSION_TEMPLATE,
    outliers=config_delta(
        BASE_COMPRESSION_TEMPLATE.outliers,
        fraction=0.02,
        storage_dtype=DType.INT8,
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=12,
        name="compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Measure the quality and storage effect of retaining ten times as many INT8 residual "
            "outliers as Experiment 011."
        ),
        hypothesis=(
            "Increasing INT8 residual outlier coverage from 0.2% to 2.0% produces a substantial "
            "quality improvement over Experiment 011."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "tenfold-outliers",
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
