"""Experiment 003: compress and quality-benchmark pinned Gemma 3 4B."""

from recipes import EXPERIMENT_003

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

EXPERIMENT = EXPERIMENT_003


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
