"""Experiment 002: benchmark BF16 against the accepted NanoQuant Gemma candidate."""

from pathlib import Path

from recipes import BaselineRef, ExperimentIdentity, define_quality_evaluation_experiment
from recipes._delta import config_delta, run_config_defaults

from nanoquant.quality_evaluation import QualityEvaluationRequest
from nanoquant.quality_evaluation_workflow import run_quality_evaluation_experiment

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"

_SCHEMA_DEFAULTS = run_config_defaults("google/gemma-3-1b-it")

_TEMPLATE = config_delta(
    _SCHEMA_DEFAULTS,
    model=config_delta(
        _SCHEMA_DEFAULTS.model,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    ),
    evaluation=config_delta(
        _SCHEMA_DEFAULTS.evaluation,
        suites=(
            "wikitext2-limited",
            "piqa",
            "arc_easy",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "boolq",
        ),
        sample_limit=200,
    ),
)

EXPERIMENT = define_quality_evaluation_experiment(
    ExperimentIdentity(
        number=2,
        name="benchmark-gemma-3-1b-it",
        purpose="Benchmark the accepted NanoQuant Gemma candidate against its pinned BF16 source model.",
        hypothesis="NanoQuant retains quality across WikiText-2 and the common multiple-choice suite.",
        baseline=BaselineRef.external("bf16-google-gemma-3-1b-it"),
        tags=("gemma-3-1b-it", "benchmark", "bf16-comparison", "wikitext2", "multiple-choice"),
    ),
    _TEMPLATE,
    QualityEvaluationRequest(
        snapshot=Path("google/gemma-3-1b-it"),
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        run_output=Path("evidence/m4/gemma-pageable-v28-four-block-canary"),
        device="cuda:0",
        backend="factorized",
        use_global_tuning=True,
        wikitext_samples=64,
        wikitext_sequence_length=128,
        wikitext_batch_size=1,
        task_names=(
            "piqa",
            "arc_easy",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "boolq",
        ),
        task_limit=200,
        task_batch_size=1,
        local_files_only=True,
    ),
    resolve_model_from_config=True,
)


if __name__ == "__main__":
    raise SystemExit(
        run_quality_evaluation_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
