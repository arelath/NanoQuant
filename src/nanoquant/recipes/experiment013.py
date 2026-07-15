"""Canonical recipe for legacy Experiment 013's improved free-residual run."""

from dataclasses import replace

from nanoquant.config.schema import IntentConfig, PostBlockRefitConfig

from .experiment018 import EXPERIMENT_018_CONFIG

_BASE = EXPERIMENT_018_CONFIG

EXPERIMENT_013_CONFIG = replace(
    _BASE,
    intent=IntentConfig(
        experiment_number=13,
        name="013-compress-gemma-3-1b-it-free-residual-Improved",
        purpose="Measure the improved free residual-outlier recipe with tighter rank bounds and full tuning batches.",
        hypothesis="Tighter rank allocation and no early stop improve the original free-residual ablation.",
        baseline_run="legacy-experiment-008",
        tags=("gemma-3-1b-it", "residual-outliers", "free-outliers", "improved"),
    ),
    allocation=replace(
        _BASE.allocation,
        retry=replace(
            _BASE.allocation.retry,
            thresholds=replace(
                _BASE.allocation.retry.thresholds,
                weighted_normalized_error=0.5,
                raw_normalized_error=None,
            ),
            allow_above_allocator_cap=False,
        ),
    ),
    block_tuning=replace(
        _BASE.block_tuning,
        non_factorized=replace(
            _BASE.block_tuning.non_factorized,
            loop=replace(
                _BASE.block_tuning.non_factorized.loop,
                early_stop_relative_tolerance=None,
            ),
            epochs_by_layer_position=(),
        ),
        post_block_refit=PostBlockRefitConfig(),
    ),
)

__all__ = ["EXPERIMENT_013_CONFIG"]
