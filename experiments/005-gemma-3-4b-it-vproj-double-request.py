"""Experiment 005: request twice the Experiment 003 v_proj bits and benchmark quality."""

from recipes import EXPERIMENT_005, EXPERIMENT_005_CONFIG

from nanoquant.rank_expansion_experiment import run_rank_expansion_experiment

CONFIG = EXPERIMENT_005_CONFIG
EXPERIMENT = EXPERIMENT_005


if __name__ == "__main__":
    raise SystemExit(run_rank_expansion_experiment(CONFIG, EXPERIMENT, launcher_path=__file__))
