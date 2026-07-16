"""Compression, GGUF export, and BF16-versus-NanoQuant benchmark recipe."""

from dataclasses import replace
from pathlib import Path

from nanoquant.compression_benchmark_workflow import CompressionBenchmarkExperiment
from nanoquant.config.schema import IntentConfig

from .experiment018 import EXPERIMENT_018_CONFIG

EXPERIMENT_001_CONFIG = replace(
    EXPERIMENT_018_CONFIG,
    intent=IntentConfig(
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
)

_OUTPUT_ROOT = Path("outputs/001-gemma-3-1b-it")

EXPERIMENT_001 = CompressionBenchmarkExperiment(
    logical_output=_OUTPUT_ROOT / "logical",
    packed_output=_OUTPUT_ROOT / "packed",
    checkpoint_output=_OUTPUT_ROOT / "llamacpp-checkpoint",
    gguf_output=_OUTPUT_ROOT / "gemma-3-1b-it-nanoquant.gguf",
    benchmark_output=_OUTPUT_ROOT / "benchmark.json",
    llama_cpp_root=Path(r"D:\dev\research\llama.cpp"),
)

__all__ = ["EXPERIMENT_001", "EXPERIMENT_001_CONFIG"]
