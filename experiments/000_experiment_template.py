"""Template: copy to the next numbered filename and fill in experiment intent."""

from nanoquant.application import run_experiment
from nanoquant.config.schema import IntentConfig, ModelConfig, RunConfig

CONFIG = RunConfig(
    model=ModelConfig(source="replace-with-pinned-model-source"),
    intent=IntentConfig(
        experiment_number=None,
        name="experiment-template",
        purpose="State the decision this experiment supports.",
        hypothesis="State the expected measurable outcome.",
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_experiment(CONFIG, launcher_path=__file__))
