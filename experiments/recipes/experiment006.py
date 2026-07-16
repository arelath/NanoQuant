"""Experiment 006: Gemma 3 1B attention-rank compression quality baseline."""

from pathlib import Path

from nanoquant.compression_quality_workflow import CompressionQualityExperiment

from ._delta import config_delta
from .base_compression import BASE_COMPRESSION_CONFIG, compression_export_recipe

EXPERIMENT_006_CONFIG = config_delta(
    BASE_COMPRESSION_CONFIG,
    intent=config_delta(
        BASE_COMPRESSION_CONFIG.intent,
        experiment_number=6,
        name="006-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Establish a complete Gemma 3 1B compression and quality baseline for the full-rank "
            "v_proj/k_proj and enlarged q_proj allocation policy."
        ),
        hypothesis=(
            "Additional attention-projection capacity lowers reconstruction error and improves matched "
            "WikiText-2 and task quality after the complete tuning and distillation pipeline."
        ),
        baseline_run="bf16-google-gemma-3-1b-it",
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "attention-rank",
            "wikitext2",
            "ultrachat",
        ),
    ),
    output=config_delta(
        BASE_COMPRESSION_CONFIG.output,
        run_root="evidence/m13",
    ),
)

EXPERIMENT_006 = CompressionQualityExperiment(
    export=compression_export_recipe(6, "gemma-3-1b-it"),
    summary_output=Path("evidence/m13/006-gemma-3-1b-it-summary.json"),
    quality_output=Path("evidence/m13/006-gemma-3-1b-it-quality.json"),
    quality_markdown_output=Path("evidence/m13/006-gemma-3-1b-it-quality.md"),
    expected_blocks=26,
    maximum_wddm_shared_gib=0.75,
)

__all__ = ["EXPERIMENT_006", "EXPERIMENT_006_CONFIG"]
