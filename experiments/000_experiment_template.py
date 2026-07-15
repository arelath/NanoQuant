"""Template for a new experiment; do not copy orchestration from an old experiment."""

from nanoquant.application import run_quantization_experiment
from nanoquant.config.schema import DatasetConfig, IntentConfig, ModelConfig, RunConfig

CONFIG = RunConfig(
    model=ModelConfig(
        source="replace-with-pinned-model-source",
        revision="replace-with-pinned-model-revision",
        tokenizer_revision="replace-with-pinned-tokenizer-revision",
    ),
    intent=IntentConfig(
        experiment_number=None,
        name="experiment-template",
        purpose="State the decision this experiment supports.",
        hypothesis="State the expected measurable outcome.",
    ),
    # Prepare a new immutable calibration dataset through the shared dataset
    # service, then reference it here. Do not reuse an old experiment runfile.
    dataset=DatasetConfig(
        prepared_root="replace-with-calibration-artifact-root",
        prepared_artifact="replace-with-calibration-manifest-artifact-id",
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_quantization_experiment(CONFIG, launcher_path=__file__))
