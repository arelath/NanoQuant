"""Experiment 001: compress, export, and benchmark pinned Gemma 3 1B."""

from recipes import EXPERIMENT_001

from nanoquant.compression_benchmark_workflow import run_compression_benchmark_experiment

EXPERIMENT = EXPERIMENT_001


if __name__ == "__main__":
    raise SystemExit(
        run_compression_benchmark_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
