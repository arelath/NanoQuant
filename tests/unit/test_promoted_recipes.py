from __future__ import annotations

import runpy
from pathlib import Path
from typing import cast

import torch

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.recipes import (
    EXPERIMENT_001_CONFIG,
    EXPERIMENT_008_CONFIG,
    EXPERIMENT_013_CONFIG,
    EXPERIMENT_018_CONFIG,
)
from nanoquant.resident_workflow import ResolvedResidentInputs, resident_request_from_config


def _diff(left: object, right: object, prefix: str = "") -> set[str]:
    left = to_dict(left)
    right = to_dict(right)
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            paths.update(_diff(left.get(key), right.get(key), path))
        return paths
    return set() if left == right else {prefix}


def _inputs(config: RunConfig, tmp_path: Path) -> ResolvedResidentInputs:
    tokens = torch.zeros((config.calibration.sample_count, 8), dtype=torch.long)
    return ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=tmp_path / config.intent.name,
        registry_root=tmp_path,
        token_ids=tokens,
        quality_token_ids=tokens[:1],
        pad_token_id=0,
    )


def test_experiment013_preserves_pre_phase1_improved_recipe(tmp_path: Path) -> None:
    config = EXPERIMENT_013_CONFIG
    request = resident_request_from_config(config, _inputs(config, tmp_path))

    assert config.allocation.bounds.floor_fraction_of_uniform == 0.9
    assert config.allocation.bounds.ceiling_fraction_of_uniform == 1.1
    assert config.allocation.retry.thresholds.weighted_normalized_error == 0.5
    assert config.allocation.retry.thresholds.raw_normalized_error is None
    assert not config.allocation.retry.allow_above_allocator_cap
    assert not config.outliers.charge_to_bit_budget
    assert config.block_tuning.non_factorized.loop.early_stop_relative_tolerance is None
    assert request.nonfactorized_tuning_epochs == 8
    assert request.nonfactorized_tuning_epochs_by_layer == ()
    assert request.factorized_tuning_epochs == 8
    assert request.post_block_refit_epochs == 0
    assert request.tuning_microbatch_size == 8
    assert request.defer_run_completion


def test_experiment001_preserves_original_no_outlier_baseline(tmp_path: Path) -> None:
    config = EXPERIMENT_001_CONFIG
    request = resident_request_from_config(config, _inputs(config, tmp_path))

    assert _diff(EXPERIMENT_018_CONFIG, config) == {
        "intent.experiment_number",
        "intent.name",
        "intent.purpose",
        "intent.hypothesis",
        "intent.baseline_run",
        "intent.tags",
        "allocation.bounds.floor_fraction_of_uniform",
        "allocation.bounds.ceiling_fraction_of_uniform",
        "allocation.retry.thresholds.raw_normalized_error",
        "allocation.retry.allow_above_allocator_cap",
        "outliers.selector",
        "outliers.fraction",
        "outliers.charge_to_bit_budget",
        "block_tuning.layer_order",
        "block_tuning.non_factorized.loop.early_stop_relative_tolerance",
        "block_tuning.non_factorized.epochs_by_layer_position",
        "block_tuning.post_block_refit.enabled",
        "block_tuning.post_block_refit.epochs",
        "block_tuning.post_block_refit.batch_size",
        "block_tuning.post_block_refit.scale_learning_rate",
    }
    assert request.layer_order == (
        "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.k_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    assert config.allocation.bounds.floor_fraction_of_uniform == 0.8
    assert config.allocation.bounds.ceiling_fraction_of_uniform == 1.15
    assert config.outliers.fraction == 0.0
    assert request.outliers is not None
    assert request.outliers.selector.value == "none"
    assert request.outliers.fraction == 0.0
    assert request.nonfactorized_tuning_epochs == 8
    assert request.nonfactorized_tuning_epochs_by_layer == ()
    assert request.nonfactorized_tuning_early_stop_relative_tolerance == 1e-3
    assert request.factorized_tuning_epochs == 8
    assert request.post_block_refit_epochs == 0
    assert request.defer_run_completion


def test_experiment008_is_only_the_documented_recipe_delta_from_013(tmp_path: Path) -> None:
    config = EXPERIMENT_008_CONFIG
    request = resident_request_from_config(config, _inputs(config, tmp_path))

    assert _diff(EXPERIMENT_013_CONFIG, config) == {
        "intent.experiment_number",
        "intent.name",
        "intent.purpose",
        "intent.hypothesis",
        "intent.baseline_run",
        "intent.tags",
        "allocation.bounds.floor_fraction_of_uniform",
        "allocation.bounds.ceiling_fraction_of_uniform",
        "block_tuning.non_factorized.loop.early_stop_relative_tolerance",
    }
    assert config.allocation.bounds.floor_fraction_of_uniform == 0.8
    assert config.allocation.bounds.ceiling_fraction_of_uniform == 1.15
    assert config.block_tuning.non_factorized.loop.early_stop_relative_tolerance == 1e-3
    assert not request.outliers.charge_to_bit_budget
    assert request.nonfactorized_tuning_early_stop_relative_tolerance == 1e-3


def test_promoted_compression_runfiles_import_the_canonical_recipe_objects() -> None:
    cases = (
        (
            "experiments/001-compress-gemma-3-1b-it.py",
            EXPERIMENT_001_CONFIG,
        ),
        (
            "experiments/008-compress-gemma-3-1b-it-free-residual-outliers.py",
            EXPERIMENT_008_CONFIG,
        ),
        (
            "experiments/013-compress-gemma-3-1b-it-free-residual-Improved.py",
            EXPERIMENT_013_CONFIG,
        ),
    )
    for path, expected in cases:
        namespace = runpy.run_path(path)
        assert cast(RunConfig, namespace["CONFIG"]) is expected
