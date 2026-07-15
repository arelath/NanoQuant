"""Experiment 013: improved Gemma 3 1B free residual-outlier recipe."""

from nanoquant.recipes import EXPERIMENT_013_CONFIG
from nanoquant.resident_workflow import run_resident_experiment

CONFIG = EXPERIMENT_013_CONFIG


if __name__ == "__main__":
    raise SystemExit(run_resident_experiment(CONFIG, launcher_path=__file__))
