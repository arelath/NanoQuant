"""Experiment 007: compress and quality-benchmark pinned Gemma 3 270M."""

from recipes import EXPERIMENT_007, EXPERIMENT_007_CONFIG

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

CONFIG = EXPERIMENT_007_CONFIG
EXPERIMENT = EXPERIMENT_007


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(CONFIG, EXPERIMENT, launcher_path=__file__)
    )
