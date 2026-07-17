"""Experiment 007: compress and quality-benchmark pinned Gemma 3 270M."""

from recipes import EXPERIMENT_007

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

EXPERIMENT = EXPERIMENT_007


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
