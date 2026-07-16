"""Experiment 002: benchmark BF16 against the accepted NanoQuant Gemma candidate."""

from recipes import EXPERIMENT_002_CONFIG, EXPERIMENT_002_EVALUATION

from nanoquant.quality_evaluation_workflow import run_quality_evaluation_experiment

CONFIG = EXPERIMENT_002_CONFIG
EVALUATION = EXPERIMENT_002_EVALUATION


if __name__ == "__main__":
    raise SystemExit(
        run_quality_evaluation_experiment(CONFIG, EVALUATION, launcher_path=__file__)
    )
