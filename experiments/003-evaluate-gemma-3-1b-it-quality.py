"""Experiment 003: base-versus-frozen Gemma quality smoke evaluation."""

from nanoquant.quality_evaluation_workflow import run_quality_evaluation_experiment
from nanoquant.recipes import EXPERIMENT_003_CONFIG, EXPERIMENT_003_EVALUATION

CONFIG = EXPERIMENT_003_CONFIG
EVALUATION = EXPERIMENT_003_EVALUATION


if __name__ == "__main__":
    raise SystemExit(
        run_quality_evaluation_experiment(CONFIG, EVALUATION, launcher_path=__file__)
    )
