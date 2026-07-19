from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    _clone_forward_metadata,
    _epoch_cooldown_observer,
    _resident_config_hash,
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
