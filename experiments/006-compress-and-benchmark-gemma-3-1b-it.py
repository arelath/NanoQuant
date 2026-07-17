"""Experiment 006: compress and quality-benchmark pinned Gemma 3 1B."""

from recipes import EXPERIMENT_006

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

EXPERIMENT = EXPERIMENT_006


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
