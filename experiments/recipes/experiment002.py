"""Benchmark-only BF16-versus-NanoQuant Gemma 3 1B experiment."""

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

EXPERIMENT_002_CONFIG = RunConfig(
    model=ModelConfig(
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        sequence_length=2048,
    ),
    intent=IntentConfig(
        experiment_number=2,
        name="002-benchmark-gemma-3-1b-it",
        purpose="Benchmark the accepted NanoQuant Gemma candidate against its pinned BF16 source model.",
        hypothesis="NanoQuant retains quality across WikiText-2 and the common legacy multiple-choice suite.",
        baseline_run="bf16-google-gemma-3-1b-it",
        tags=("gemma-3-1b-it", "benchmark", "bf16-comparison", "wikitext2", "multiple-choice"),
    ),
    runtime=RuntimeConfig(compute_device="cuda:0"),
    evaluation=EvaluationConfig(
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
        few_shot=0,
    ),
)

EXPERIMENT_002_EVALUATION = QualityEvaluationExperiment(
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
    Path("evidence/m9/002-gemma-3-1b-it-quality-benchmark.json"),
    resolve_model_from_config=True,
    markdown_path=Path("evidence/m9/002-gemma-3-1b-it-quality-benchmark.md"),
)

__all__ = ["EXPERIMENT_002_CONFIG", "EXPERIMENT_002_EVALUATION", "MODEL_REVISION"]
