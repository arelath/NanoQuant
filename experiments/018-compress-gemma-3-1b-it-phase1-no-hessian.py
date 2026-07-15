"""Experiment 018: pinned Gemma 3 1B diagonal-objective parity recipe."""

from nanoquant.recipes import EXPERIMENT_018_CONFIG
from nanoquant.resident_workflow import run_resident_experiment

CONFIG = EXPERIMENT_018_CONFIG


if __name__ == "__main__":
    raise SystemExit(run_resident_experiment(CONFIG, launcher_path=__file__))
