"""Experiment 003: compress and quality-benchmark pinned Gemma 3 4B."""

from nanoquant.compression_quality_workflow import run_compression_quality_experiment
from nanoquant.recipes import EXPERIMENT_003, EXPERIMENT_003_CONFIG

CONFIG = EXPERIMENT_003_CONFIG
EXPERIMENT = EXPERIMENT_003


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(CONFIG, EXPERIMENT, launcher_path=__file__)
    )
