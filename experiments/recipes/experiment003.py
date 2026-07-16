"""Gemma 3 4B compression and quality-proof experiment."""

from pathlib import Path

from nanoquant.compression_quality_workflow import CompressionQualityExperiment

from ._delta import config_delta
from .base_compression import BASE_COMPRESSION_CONFIG, compression_export_recipe

MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"

_base_tuning = BASE_COMPRESSION_CONFIG.block_tuning

EXPERIMENT_003_CONFIG = config_delta(
    BASE_COMPRESSION_CONFIG,
    model=config_delta(
        BASE_COMPRESSION_CONFIG.model,
        source="google/gemma-3-4b-it",
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    ),
    intent=config_delta(
        BASE_COMPRESSION_CONFIG.intent,
        experiment_number=3,
        name="003-compress-and-benchmark-gemma-3-4b-it-v5",
        purpose=(
            "Prove that the multimodal Gemma 3 4B checkpoint's text model still compresses "
            "within dedicated VRAM and measure its BF16-versus-NanoQuant quality."
        ),
        hypothesis=(
            "CPU extraction of the language model plus pageable activation streaming keeps WDDM "
            "shared memory below the hard limit while producing a complete 34-block candidate."
        ),
        baseline_run="bf16-google-gemma-3-4b-it",
        tags=("gemma-3-4b-it", "compression", "quality", "shared-vram-guard", "profiling"),
    ),
    allocation=config_delta(
        BASE_COMPRESSION_CONFIG.allocation,
        maximum_rank_layer_patterns=(),
        layer_budget_multipliers=(),
        retry=config_delta(
            BASE_COMPRESSION_CONFIG.allocation.retry,
            thresholds=config_delta(
                BASE_COMPRESSION_CONFIG.allocation.retry.thresholds,
                weighted_normalized_error=0.35,
                raw_normalized_error=0.40,
            ),
        ),
    ),
    block_tuning=config_delta(
        _base_tuning,
        non_factorized=config_delta(
            _base_tuning.non_factorized,
            loop=config_delta(_base_tuning.non_factorized.loop, batch_size=4),
        ),
        factorized=config_delta(
            _base_tuning.factorized,
            loop=config_delta(_base_tuning.factorized.loop, batch_size=1),
        ),
        post_block_refit=config_delta(
            _base_tuning.post_block_refit,
            batch_size=1,
        ),
        microbatch_size=1,
    ),
    runtime=config_delta(
        BASE_COMPRESSION_CONFIG.runtime,
        block_forward_batch_size=4,
    ),
    evaluation=config_delta(
        BASE_COMPRESSION_CONFIG.evaluation,
        inline_quality=False,
    ),
    observability=config_delta(
        BASE_COMPRESSION_CONFIG.observability,
        record_resource_interval_seconds=1.0,
    ),
    profiling=config_delta(
        BASE_COMPRESSION_CONFIG.profiling,
        cuda_timing=True,
        memory_counters=True,
        emit_span_events=True,
    ),
    output=config_delta(
        BASE_COMPRESSION_CONFIG.output,
        run_root="evidence/m10",
    ),
)

EXPERIMENT_003 = CompressionQualityExperiment(
    export=compression_export_recipe(3, "gemma-3-4b-it"),
    summary_output=Path("evidence/m10/003-gemma-3-4b-it-summary.json"),
    quality_output=Path("evidence/m10/003-gemma-3-4b-it-quality.json"),
    quality_markdown_output=Path("evidence/m10/003-gemma-3-4b-it-quality.md"),
    expected_blocks=34,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
)

__all__ = ["EXPERIMENT_003", "EXPERIMENT_003_CONFIG", "MODEL_REVISION"]
