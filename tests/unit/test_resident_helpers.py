from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from nanoquant.config.schema import BinaryFactorSearchConfig
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    _clone_forward_metadata,
    _epoch_cooldown_observer,
    _execute_binary_factor_search,
    _resident_config_hash,
    _stored_search_error,
)


def test_numerical_batch_shapes_invalidate_resume_identity(tmp_path: Path) -> None:
    request = ResidentQuantizationRequest(
        tmp_path / "snapshot",
        tmp_path / "output",
        "fixture/model",
        "revision",
        ((1, 2),),
        device="cpu",
    )

    assert _resident_config_hash(replace(request, tuning_microbatch_size=2)) != _resident_config_hash(request)
    assert _resident_config_hash(replace(request, block_forward_batch_size=2)) != _resident_config_hash(request)
    assert _resident_config_hash(replace(request, restore_best_tuning_state=False)) != _resident_config_hash(request)
    assert _resident_config_hash(
        replace(request, factorized_tuning_epoch_cooldown_seconds=5.0)
    ) == _resident_config_hash(request)
    assert _resident_config_hash(
        replace(request, nonfactorized_tuning_epoch_cooldown_seconds=5.0)
    ) == _resident_config_hash(request)
    assert _resident_config_hash(
        replace(request, post_block_refit_epoch_cooldown_seconds=5.0)
    ) == _resident_config_hash(request)
    assert _resident_config_hash(replace(request, initial_cooldown_seconds=30.0)) == _resident_config_hash(request)


def test_epoch_cooldown_skips_initial_loss_and_sleeps_after_training_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("nanoquant.resident_quantization.time.sleep", sleeps.append)

    observer = _epoch_cooldown_observer(2.5)
    assert observer is not None
    observer(0, 10.0)
    observer(1, 9.0)
    observer(2, 8.0)

    assert sleeps == [2.5, 2.5]
    assert _epoch_cooldown_observer(0.0) is None


def test_forward_metadata_clone_isolates_nested_tensor_mutation() -> None:
    source = {
        "attention_mask": torch.tensor([[1.0, 2.0]]),
        "position_embeddings": (torch.tensor([3.0]), {"sin": torch.tensor([4.0])}),
        "flag": True,
    }

    cloned = _clone_forward_metadata(source)
    cast(torch.Tensor, cloned["attention_mask"]).zero_()
    position_embeddings = cast(tuple[torch.Tensor, dict[str, torch.Tensor]], cloned["position_embeddings"])
    position_embeddings[0].add_(10)
    position_embeddings[1]["sin"].mul_(0)

    assert torch.equal(cast(torch.Tensor, source["attention_mask"]), torch.tensor([[1.0, 2.0]]))
    source_positions = cast(tuple[torch.Tensor, dict[str, torch.Tensor]], source["position_embeddings"])
    assert torch.equal(source_positions[0], torch.tensor([3.0]))
    assert torch.equal(source_positions[1]["sin"], torch.tensor([4.0]))


def test_binary_factor_search_selects_the_best_stored_dtype_state() -> None:
    generator = torch.Generator().manual_seed(91)
    target = torch.randn((5, 5), generator=generator)
    left = torch.randint(0, 2, (5, 5), generator=generator).float().mul_(2).sub_(1)
    right = torch.randint(0, 2, (5, 5), generator=generator).float().mul_(2).sub_(1)
    scales = torch.ones(5, dtype=torch.bfloat16)
    importance = torch.ones(5)
    config = BinaryFactorSearchConfig(
        enabled=True,
        scale_passes=8,
        control_outer_passes=2,
        one_bit_passes=4,
        max_one_bit_vectors=5,
        variable_depth_passes=1,
        variable_depth_length=5,
        tabu_outer_passes=2,
        tabu_passes=1,
        tabu_steps=16,
        tabu_tenure=3,
        tabu_tenure_jitter=2,
    )

    selected, metrics = _execute_binary_factor_search(
        target,
        left,
        right,
        scales,
        scales,
        scales,
        importance,
        importance,
        config,
    )
    selected_error = _stored_search_error(
        target,
        selected,
        importance,
        importance,
        torch.bfloat16,
    )

    assert selected_error == pytest.approx(
        min(
            metrics.initial_weighted_squared_error,
            metrics.control_weighted_squared_error,
            metrics.tabu_weighted_squared_error,
        )
    )
    assert metrics.selected_initial == (
        metrics.initial_weighted_squared_error
        <= min(metrics.control_weighted_squared_error, metrics.tabu_weighted_squared_error)
    )
    assert metrics.selected_tabu == (
        metrics.tabu_weighted_squared_error
        < min(metrics.initial_weighted_squared_error, metrics.control_weighted_squared_error)
    )


def test_binary_factor_search_retains_the_incumbent_on_a_stored_dtype_tie() -> None:
    left = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    right = left.clone()
    scales = torch.zeros(2, dtype=torch.bfloat16)
    config = BinaryFactorSearchConfig(
        enabled=True,
        scale_passes=2,
        control_outer_passes=1,
        one_bit_passes=1,
        max_one_bit_vectors=2,
        variable_depth_passes=1,
        variable_depth_length=2,
        tabu_outer_passes=1,
        tabu_passes=1,
        tabu_steps=2,
        tabu_tenure=1,
        tabu_tenure_jitter=0,
    )

    selected, metrics = _execute_binary_factor_search(
        torch.zeros((2, 2)),
        left,
        right,
        scales,
        scales,
        scales,
        torch.ones(2),
        torch.ones(2),
        config,
    )

    assert metrics.initial_weighted_squared_error == 0.0
    assert metrics.selected_initial is True
    assert metrics.selected_tabu is False
    assert torch.equal(selected.left_binary, left)
    assert torch.equal(selected.right_binary, right)
