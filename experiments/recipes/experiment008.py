"""Experiment 008: Gemma 3 12B large-model compression quality benchmark."""

from pathlib import Path

from nanoquant.compression_quality_workflow import CompressionQualityExperiment

from ._delta import config_delta
from .base_compression import LARGE_MODEL_COMPRESSION_CONFIG, compression_export_recipe

# The requested repository publishes GGUF containers only. NanoQuant currently
# inventories safetensors sources, so compression uses the matching Unsloth BF16
# Transformers repository and records the requested BF16 GGUF as provenance.
REQUESTED_GGUF_REPOSITORY = "unsloth/gemma-3-12b-it-GGUF"
REQUESTED_GGUF_REVISION = "d15e4c7dc21dc55d56bf8549db57a71ad8a2a35d"
REQUESTED_GGUF_FILENAME = "gemma-3-12b-it-BF16.gguf"
MODEL_SOURCE = "unsloth/gemma-3-12b-it"
MODEL_REVISION = "9478e665381f42974aa06177b019352fb6291876"

_base_tuning = LARGE_MODEL_COMPRESSION_CONFIG.block_tuning

EXPERIMENT_008_CONFIG = config_delta(
    LARGE_MODEL_COMPRESSION_CONFIG,
    model=config_delta(
        LARGE_MODEL_COMPRESSION_CONFIG.model,
        source=MODEL_SOURCE,
        revision=MODEL_REVISION,
        tokenizer_source=MODEL_SOURCE,
        tokenizer_revision=MODEL_REVISION,
    ),
    intent=config_delta(
        LARGE_MODEL_COMPRESSION_CONFIG.intent,
        experiment_number=8,
        name="008-compress-and-benchmark-gemma-3-12b-it",
        purpose=(
            "Compress and quality-benchmark the BF16 weights corresponding to "
            "unsloth/gemma-3-12b-it-GGUF without entering WDDM shared memory."
        ),
        hypothesis=(
            "CPU-offloaded block compression and packed evaluation can process the 48-block 12B model "
            "within 12 GiB of dedicated VRAM."
        ),
        baseline_run=(
            f"{REQUESTED_GGUF_REPOSITORY}@{REQUESTED_GGUF_REVISION}:"
            f"{REQUESTED_GGUF_FILENAME}"
        ),
        tags=(
            "gemma-3-12b-it",
            "unsloth",
            "compression",
            "quality",
            "large-model",
            "cpu-offload",
            "packed-quality",
            "wikitext2",
            "ultrachat",
        ),
    ),
    block_tuning=config_delta(
        _base_tuning,
        non_factorized=config_delta(
            _base_tuning.non_factorized,
            loop=config_delta(_base_tuning.non_factorized.loop, batch_size=1),
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
        LARGE_MODEL_COMPRESSION_CONFIG.runtime,
        block_forward_batch_size=1,
    ),
    output=config_delta(
        LARGE_MODEL_COMPRESSION_CONFIG.output,
        run_root="evidence/m15",
    ),
)

EXPERIMENT_008 = CompressionQualityExperiment(
    export=compression_export_recipe(8, "gemma-3-12b-it"),
    summary_output=Path("evidence/m15/008-gemma-3-12b-it-summary.json"),
    quality_output=Path("evidence/m15/008-gemma-3-12b-it-quality.json"),
    quality_markdown_output=Path("evidence/m15/008-gemma-3-12b-it-quality.md"),
    expected_blocks=48,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    large_model_guards=True,
)

__all__ = [
    "EXPERIMENT_008",
    "EXPERIMENT_008_CONFIG",
    "MODEL_REVISION",
    "MODEL_SOURCE",
    "REQUESTED_GGUF_FILENAME",
    "REQUESTED_GGUF_REPOSITORY",
    "REQUESTED_GGUF_REVISION",
]
