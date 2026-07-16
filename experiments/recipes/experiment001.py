"""Compression, GGUF export, and BF16-versus-NanoQuant benchmark recipe."""

from nanoquant.compression_benchmark_workflow import CompressionBenchmarkExperiment

from ._delta import config_delta
from .base_compression import BASE_COMPRESSION_CONFIG, compression_export_recipe

EXPERIMENT_001_CONFIG = config_delta(
    BASE_COMPRESSION_CONFIG,
    intent=config_delta(
        BASE_COMPRESSION_CONFIG.intent,
        experiment_number=1,
        name="001-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Compress pinned Gemma 3 1B with the current parity recipe, export a deployable "
            "NanoQuant GGUF, and compare it with the BF16 source model."
        ),
        hypothesis=(
            "The current resident recipe produces a validated GGUF while retaining measured "
            "quality parity on WikiText-2 and the six-task legacy evaluation suite."
        ),
        baseline_run="bf16-google-gemma-3-1b-it",
        tags=("gemma-3-1b-it", "compression", "gguf", "bf16-comparison", "quality"),
    ),
    allocation=config_delta(
        BASE_COMPRESSION_CONFIG.allocation,
        maximum_rank_layer_patterns=(),
        layer_budget_multipliers=(),
    ),
)

_EXPORT = compression_export_recipe(1, "gemma-3-1b-it")

EXPERIMENT_001 = CompressionBenchmarkExperiment(
    export=_EXPORT,
    benchmark_output=_EXPORT.gguf_output.parent / "benchmark.json",
)

__all__ = ["EXPERIMENT_001", "EXPERIMENT_001_CONFIG"]
