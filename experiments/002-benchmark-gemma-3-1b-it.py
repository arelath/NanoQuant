"""Experiment 002: benchmark BF16 against the accepted NanoQuant Gemma candidate."""

from recipes import EXPERIMENT_002

from nanoquant.quality_evaluation_workflow import run_quality_evaluation_experiment

EXPERIMENT = EXPERIMENT_002


if __name__ == "__main__":
    raise SystemExit(
        run_quality_evaluation_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
