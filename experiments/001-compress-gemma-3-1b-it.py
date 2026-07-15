"""Experiment 001: historical Gemma 3 1B compression baseline."""

from nanoquant.recipes import EXPERIMENT_001_CONFIG
from nanoquant.resident_workflow import run_resident_experiment

CONFIG = EXPERIMENT_001_CONFIG


if __name__ == "__main__":
    raise SystemExit(run_resident_experiment(CONFIG, launcher_path=__file__))
