"""Canonical historical recipe for legacy Experiment 001."""

from dataclasses import replace

from nanoquant.config.schema import (
    IntentConfig,
    OutlierConfig,
    PostBlockRefitConfig,
)

from .experiment018 import EXPERIMENT_018_CONFIG

_BASE = EXPERIMENT_018_CONFIG

# Legacy 001 used the adapter's original order, which predates the MLP-first
# Phase-1 experiments.  Spell it out so future adapter-order changes cannot
# silently change this historical recipe.
_LEGACY_LAYER_ORDER = (
    "self_attn.q_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.k_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

EXPERIMENT_001_CONFIG = replace(
    _BASE,
    intent=IntentConfig(
        experiment_number=1,
        name="001-compress-gemma-3-1b-it",
        purpose="Preserve the original 1B sensitivity-ranked compression baseline as a native resident recipe.",
        hypothesis="The historical no-outlier recipe remains reproducible without its executable pickle output.",
        baseline_run="legacy-experiment-001",
        tags=("gemma-3-1b-it", "historical-baseline", "diagonal", "model-kd"),
    ),
    allocation=replace(
        _BASE.allocation,
        bounds=replace(
            _BASE.allocation.bounds,
            floor_fraction_of_uniform=0.8,
            ceiling_fraction_of_uniform=1.15,
        ),
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
    outliers=OutlierConfig(),
    block_tuning=replace(
        _BASE.block_tuning,
        layer_order=_LEGACY_LAYER_ORDER,
        non_factorized=replace(
            _BASE.block_tuning.non_factorized,
            loop=replace(
                _BASE.block_tuning.non_factorized.loop,
                early_stop_relative_tolerance=1e-3,
            ),
            epochs_by_layer_position=(),
        ),
        post_block_refit=PostBlockRefitConfig(),
    ),
)

__all__ = ["EXPERIMENT_001_CONFIG"]
