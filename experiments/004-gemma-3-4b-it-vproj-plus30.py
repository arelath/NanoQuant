"""Experiment 004: selectively expand Experiment 003 v_proj rank and benchmark quality."""

from recipes import EXPERIMENT_004, EXPERIMENT_004_CONFIG

from nanoquant.rank_expansion_experiment import run_rank_expansion_experiment

CONFIG = EXPERIMENT_004_CONFIG
EXPERIMENT = EXPERIMENT_004


if __name__ == "__main__":
    raise SystemExit(run_rank_expansion_experiment(CONFIG, EXPERIMENT, launcher_path=__file__))
