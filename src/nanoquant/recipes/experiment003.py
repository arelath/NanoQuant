"""Gemma 3 4B compression and quality-proof experiment."""

from dataclasses import replace
from pathlib import Path

from nanoquant.compression_quality_workflow import CompressionQualityExperiment
from nanoquant.config.schema import (
    BlockTuningConfig,
    IntentConfig,
    ModelConfig,
    ObservabilityConfig,
    OutputConfig,
    ProfilingConfig,
    ProfilingLevel,
    RankRetryConfig,
    RetryThresholdConfig,
)

from .experiment018 import EXPERIMENT_018_CONFIG

MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"

_base_tuning = EXPERIMENT_018_CONFIG.block_tuning
_base_runtime = EXPERIMENT_018_CONFIG.runtime

EXPERIMENT_003_CONFIG = replace(
    EXPERIMENT_018_CONFIG,
    model=ModelConfig(
        source="google/gemma-3-4b-it",
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        sequence_length=2048,
        load_dtype=EXPERIMENT_018_CONFIG.model.load_dtype,
    ),
    intent=IntentConfig(
        experiment_number=3,
        name="003-compress-and-benchmark-gemma-3-4b-it-v4",
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
    allocation=replace(
        EXPERIMENT_018_CONFIG.allocation,
        retry=RankRetryConfig(
            thresholds=RetryThresholdConfig(
                weighted_normalized_error=0.35,
                raw_normalized_error=0.40,
            ),
            rank_increase_fraction=0.25,
            maximum_attempts=3,
            extra_bit_budget_fraction=0.02,
            allow_above_allocator_cap=True,
        ),
    ),
    block_tuning=BlockTuningConfig(
        layer_order=_base_tuning.layer_order,
        non_factorized=replace(
            _base_tuning.non_factorized,
            loop=replace(_base_tuning.non_factorized.loop, batch_size=4),
        ),
        factorized=replace(
            _base_tuning.factorized,
            loop=replace(_base_tuning.factorized.loop, batch_size=1),
        ),
        post_block_refit=replace(
            _base_tuning.post_block_refit,
            batch_size=1,
        ),
        microbatch_size=1,
        reset_seed_each_stage=_base_tuning.reset_seed_each_stage,
        restore_best_state=_base_tuning.restore_best_state,
        epoch_loss_mode=_base_tuning.epoch_loss_mode,
    ),
    runtime=replace(
        _base_runtime,
        block_forward_batch_size=4,
    ),
    observability=ObservabilityConfig(
        event_level="info",
        console_level="info",
        record_resource_interval_seconds=1.0,
        record_weight_reconstruction_table=True,
        record_block_loss_snapshots=True,
    ),
    profiling=ProfilingConfig(
        level=ProfilingLevel.MACRO,
        cuda_timing=True,
        cuda_sample_every=16,
        memory_counters=True,
        raw_samples_per_phase=64,
        emit_span_events=True,
    ),
    output=OutputConfig(
        run_root="evidence/m10",
        artifact_root="artifacts",
        retain_temporary_artifacts=False,
    ),
)

EXPERIMENT_003 = CompressionQualityExperiment(
    summary_output=Path("evidence/m10/003-gemma-3-4b-it-summary.json"),
    quality_output=Path("evidence/m10/003-gemma-3-4b-it-quality.json"),
    quality_markdown_output=Path("evidence/m10/003-gemma-3-4b-it-quality.md"),
    expected_blocks=34,
    maximum_wddm_shared_gib=0.75,
)

__all__ = ["EXPERIMENT_003", "EXPERIMENT_003_CONFIG", "MODEL_REVISION"]
