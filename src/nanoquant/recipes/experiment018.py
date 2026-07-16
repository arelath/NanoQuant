"""Canonical recipe for legacy Experiment 018 parity."""

from dataclasses import replace

from nanoquant.config.schema import IntentConfig

from .base_compression import BASE_COMPRESSION_CONFIG, MODEL_REVISION

EXPERIMENT_018_CONFIG = replace(
    BASE_COMPRESSION_CONFIG,
    intent=IntentConfig(
        experiment_number=18,
        name="018-compress-gemma-3-1b-it-phase1-no-hessian",
        purpose="Reproduce the closest retained legacy 1B diagonal/no-Hessian compression baseline.",
        hypothesis="The rewrite matches legacy rank, tuning, KD, BPW, and quality behavior with bounded memory.",
        baseline_run="legacy-experiment-018",
        tags=("gemma-3-1b-it", "parity", "diagonal", "model-kd"),
    ),
    allocation=replace(
        BASE_COMPRESSION_CONFIG.allocation,
        maximum_rank_layer_patterns=(),
    ),
)

__all__ = ["EXPERIMENT_018_CONFIG", "MODEL_REVISION"]
