"""Experiment 001: compress, export, and benchmark pinned Gemma 3 1B."""

from nanoquant.compression_benchmark_workflow import run_compression_benchmark_experiment
from nanoquant.recipes import EXPERIMENT_001, EXPERIMENT_001_CONFIG

CONFIG = EXPERIMENT_001_CONFIG
EXPERIMENT = EXPERIMENT_001


if __name__ == "__main__":
    raise SystemExit(
        run_compression_benchmark_experiment(CONFIG, EXPERIMENT, launcher_path=__file__)
    )
