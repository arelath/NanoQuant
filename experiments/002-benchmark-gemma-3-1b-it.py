"""Experiment 002: paired source, logical, and packed short-decode benchmark."""

from nanoquant.recipes import EXPERIMENT_002_BENCHMARK, EXPERIMENT_002_CONFIG
from nanoquant.short_decode_workflow import run_short_decode_experiment

CONFIG = EXPERIMENT_002_CONFIG
BENCHMARK = EXPERIMENT_002_BENCHMARK


if __name__ == "__main__":
    raise SystemExit(run_short_decode_experiment(CONFIG, BENCHMARK, launcher_path=__file__))
