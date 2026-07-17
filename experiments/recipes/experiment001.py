"""Compression, GGUF export, and BF16-versus-NanoQuant benchmark recipe."""

from ._experiment import (
    BaselineRef,
    ExperimentIdentity,
    define_compression_benchmark_experiment,
)
from .base_compression import GEMMA_3_1B_PARITY_TEMPLATE

EXPERIMENT_001 = define_compression_benchmark_experiment(
    ExperimentIdentity(
        number=1,
        name="compress-gemma-3-1b-it",
        purpose=(
            "Compress pinned Gemma 3 1B with the current parity recipe, export a deployable "
            "NanoQuant GGUF, and compare it with the BF16 source model."
        ),
        hypothesis=(
            "The current resident recipe produces a validated GGUF while retaining measured "
            "quality parity on WikiText-2 and the six-task evaluation suite."
        ),
        baseline=BaselineRef.external("bf16-google-gemma-3-1b-it"),
        tags=("gemma-3-1b-it", "compression", "gguf", "bf16-comparison", "quality"),
    ),
    GEMMA_3_1B_PARITY_TEMPLATE,
)

__all__ = ["EXPERIMENT_001"]
