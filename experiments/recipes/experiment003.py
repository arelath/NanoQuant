"""Gemma 3 4B compression and quality-proof experiment."""

from ._delta import config_delta
from ._experiment import (
    BaselineRef,
    ExperimentIdentity,
    define_compression_quality_experiment,
)
from .base_compression import GEMMA_3_1B_PARITY_TEMPLATE

MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"

_base_tuning = GEMMA_3_1B_PARITY_TEMPLATE.block_tuning

_TEMPLATE = config_delta(
    GEMMA_3_1B_PARITY_TEMPLATE,
    model=config_delta(
        GEMMA_3_1B_PARITY_TEMPLATE.model,
        source="google/gemma-3-4b-it",
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    ),
    allocation=config_delta(
        GEMMA_3_1B_PARITY_TEMPLATE.allocation,
        retry=config_delta(
            GEMMA_3_1B_PARITY_TEMPLATE.allocation.retry,
            thresholds=config_delta(
                GEMMA_3_1B_PARITY_TEMPLATE.allocation.retry.thresholds,
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
        GEMMA_3_1B_PARITY_TEMPLATE.runtime,
        block_forward_batch_size=4,
    ),
    evaluation=config_delta(
        GEMMA_3_1B_PARITY_TEMPLATE.evaluation,
        inline_quality=False,
    ),
    observability=config_delta(
        GEMMA_3_1B_PARITY_TEMPLATE.observability,
        record_resource_interval_seconds=1.0,
    ),
    profiling=config_delta(
        GEMMA_3_1B_PARITY_TEMPLATE.profiling,
        cuda_timing=True,
        memory_counters=True,
        emit_span_events=True,
    ),
)

EXPERIMENT_003 = define_compression_quality_experiment(
    ExperimentIdentity(
        number=3,
        name="compress-and-benchmark-gemma-3-4b-it",
        purpose=(
            "Prove that the multimodal Gemma 3 4B checkpoint's text model still compresses "
            "within dedicated VRAM and measure its BF16-versus-NanoQuant quality."
        ),
        hypothesis=(
            "CPU extraction of the language model plus pageable activation streaming keeps WDDM "
            "shared memory below the hard limit while producing a complete 34-block candidate."
        ),
        baseline=BaselineRef.external("bf16-google-gemma-3-4b-it"),
        tags=("gemma-3-4b-it", "compression", "quality", "shared-vram-guard", "profiling"),
    ),
    _TEMPLATE,
    expected_blocks=34,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
)

__all__ = ["EXPERIMENT_003", "MODEL_REVISION"]
