"""Experiment 007: Gemma 3 270M compression quality benchmark."""

from pathlib import Path

from nanoquant.compression_quality_workflow import CompressionQualityExperiment

from ._delta import config_delta
from .base_compression import compression_export_recipe
from .experiment006 import EXPERIMENT_006_CONFIG

MODEL_REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"

EXPERIMENT_007_CONFIG = config_delta(
    EXPERIMENT_006_CONFIG,
    model=config_delta(
        EXPERIMENT_006_CONFIG.model,
        source="unsloth/gemma-3-270m-it",
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    ),
    intent=config_delta(
        EXPERIMENT_006_CONFIG.intent,
        experiment_number=7,
        name="007-compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Establish a complete Gemma 3 270M compression and quality benchmark using the promoted "
            "attention-projection allocation policy."
        ),
        hypothesis=(
            "The full-rank v_proj/k_proj and enlarged q_proj recipe remains effective at 270M scale "
            "after complete tuning and distillation."
        ),
        baseline_run="bf16-unsloth-gemma-3-270m-it",
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "attention-rank",
            "wikitext2",
            "ultrachat",
        ),
    ),
    output=config_delta(
        EXPERIMENT_006_CONFIG.output,
        run_root="evidence/m14",
    ),
)

EXPERIMENT_007 = CompressionQualityExperiment(
    export=compression_export_recipe(7, "gemma-3-270m-it"),
    summary_output=Path("evidence/m14/007-gemma-3-270m-it-summary.json"),
    quality_output=Path("evidence/m14/007-gemma-3-270m-it-quality.json"),
    quality_markdown_output=Path("evidence/m14/007-gemma-3-270m-it-quality.md"),
    expected_blocks=18,
    maximum_wddm_shared_gib=0.75,
)

__all__ = ["EXPERIMENT_007", "EXPERIMENT_007_CONFIG", "MODEL_REVISION"]
