"""Canonical recipe for legacy Experiment 018 parity."""

from .._delta import config_delta
from ..base_compression import BASE_COMPRESSION_CONFIG, MODEL_REVISION

EXPERIMENT_018_CONFIG = config_delta(
    BASE_COMPRESSION_CONFIG,
    intent=config_delta(
        BASE_COMPRESSION_CONFIG.intent,
        experiment_number=18,
        name="018-compress-gemma-3-1b-it-phase1-no-hessian",
        purpose="Reproduce the closest retained legacy 1B diagonal/no-Hessian compression baseline.",
        hypothesis="The rewrite matches legacy rank, tuning, KD, BPW, and quality behavior with bounded memory.",
        baseline_run="legacy-experiment-018",
        tags=("gemma-3-1b-it", "parity", "diagonal", "model-kd"),
    ),
    allocation=config_delta(
        BASE_COMPRESSION_CONFIG.allocation,
        maximum_rank_layer_patterns=(),
    ),
)

__all__ = ["EXPERIMENT_018_CONFIG", "MODEL_REVISION"]
