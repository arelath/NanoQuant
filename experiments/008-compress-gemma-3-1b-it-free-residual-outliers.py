"""Experiment 008: Gemma 3 1B free residual-outlier ablation."""

from nanoquant.recipes import EXPERIMENT_008_CONFIG
from nanoquant.resident_workflow import run_resident_experiment

CONFIG = EXPERIMENT_008_CONFIG


if __name__ == "__main__":
    raise SystemExit(run_resident_experiment(CONFIG, launcher_path=__file__))
