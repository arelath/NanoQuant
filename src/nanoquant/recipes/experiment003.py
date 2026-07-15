"""Canonical recipe for legacy Experiment 003's quality smoke comparison."""

from pathlib import Path

from nanoquant.config.schema import (
    EvaluationConfig,
    IntentConfig,
    ModelConfig,
    RunConfig,
    RuntimeConfig,
)
from nanoquant.quality_evaluation import QualityEvaluationRequest
from nanoquant.quality_evaluation_workflow import QualityEvaluationExperiment

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"

EXPERIMENT_003_CONFIG = RunConfig(
    model=ModelConfig(
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        sequence_length=128,
    ),
    intent=IntentConfig(
        experiment_number=3,
        name="003-evaluate-gemma-3-1b-it-quality",
        purpose="Compare the base and native frozen Gemma models on the historical quality smoke protocol.",
        hypothesis="The native release candidate preserves the legacy smoke evaluator contract and reports its gap.",
        baseline_run="legacy-experiment-003",
        tags=("gemma-3-1b-it", "quality", "wikitext2", "multiple-choice"),
    ),
    runtime=RuntimeConfig(compute_device="cuda:0"),
    evaluation=EvaluationConfig(
        suites=("wikitext2-limited", "piqa", "arc_easy", "boolq"),
        sample_limit=25,
        few_shot=0,
    ),
)

EXPERIMENT_003_EVALUATION = QualityEvaluationExperiment(
    QualityEvaluationRequest(
        snapshot=Path("google/gemma-3-1b-it"),
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        run_output=Path("evidence/m4/gemma-pageable-v28-four-block-canary"),
        device="cuda:0",
        backend="factorized",
        use_global_tuning=True,
        wikitext_samples=16,
        wikitext_sequence_length=128,
        wikitext_batch_size=1,
        task_names=("piqa", "arc_easy", "boolq"),
        task_limit=25,
        task_batch_size=1,
        local_files_only=True,
    ),
    Path("evidence/m9/003-gemma-3-1b-it-quality.json"),
    resolve_model_from_config=True,
)

__all__ = ["EXPERIMENT_003_CONFIG", "EXPERIMENT_003_EVALUATION", "MODEL_REVISION"]
