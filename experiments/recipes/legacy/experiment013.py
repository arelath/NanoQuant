"""Canonical recipe for legacy Experiment 013's improved free-residual run."""

from .._delta import config_delta
from .experiment018 import EXPERIMENT_018_CONFIG

_BASE = EXPERIMENT_018_CONFIG

EXPERIMENT_013_CONFIG = config_delta(
    _BASE,
    intent=config_delta(
        _BASE.intent,
        experiment_number=13,
        name="013-compress-gemma-3-1b-it-free-residual-Improved",
        purpose="Measure the improved free residual-outlier recipe with tighter rank bounds and full tuning batches.",
        hypothesis="Tighter rank allocation and no early stop improve the original free-residual ablation.",
        baseline_run="legacy-experiment-008",
        tags=("gemma-3-1b-it", "residual-outliers", "free-outliers", "improved"),
    ),
    allocation=config_delta(
        _BASE.allocation,
        retry=config_delta(
            _BASE.allocation.retry,
            thresholds=config_delta(
                _BASE.allocation.retry.thresholds,
                raw_normalized_error=None,
            ),
            allow_above_allocator_cap=False,
        ),
    ),
    block_tuning=config_delta(
        _BASE.block_tuning,
        non_factorized=config_delta(
            _BASE.block_tuning.non_factorized,
            epochs_by_layer_position=(),
        ),
        post_block_refit=config_delta(
            _BASE.block_tuning.post_block_refit,
            enabled=False,
            epochs=0,
            batch_size=None,
            scale_learning_rate=None,
        ),
    ),
)

__all__ = ["EXPERIMENT_013_CONFIG"]
