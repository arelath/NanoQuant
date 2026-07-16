"""Canonical recipe for legacy Experiment 008's free residual-outlier ablation."""

from .._delta import config_delta
from .experiment013 import EXPERIMENT_013_CONFIG

_BASE = EXPERIMENT_013_CONFIG

EXPERIMENT_008_CONFIG = config_delta(
    _BASE,
    intent=config_delta(
        _BASE.intent,
        experiment_number=8,
        name="008-compress-gemma-3-1b-it-free-residual-outliers",
        purpose="Measure BF16 residual-selected outlier columns without charging them to the bit budget.",
        hypothesis="A free 0.1% residual side path improves quality enough to justify its uncharged storage ablation.",
        baseline_run="legacy-experiment-006",
        tags=("gemma-3-1b-it", "residual-outliers", "free-outliers", "ablation"),
    ),
    allocation=config_delta(
        _BASE.allocation,
        bounds=config_delta(
            _BASE.allocation.bounds,
            floor_fraction_of_uniform=0.8,
            ceiling_fraction_of_uniform=1.15,
        ),
    ),
    block_tuning=config_delta(
        _BASE.block_tuning,
        non_factorized=config_delta(
            _BASE.block_tuning.non_factorized,
            loop=config_delta(
                _BASE.block_tuning.non_factorized.loop,
                early_stop_relative_tolerance=1e-3,
            ),
        ),
    ),
)

__all__ = ["EXPERIMENT_008_CONFIG"]
