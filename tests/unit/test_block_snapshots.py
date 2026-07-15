from __future__ import annotations

import pytest
import torch
from torch import nn

from nanoquant.application.block_snapshots import (
    compare_block_snapshots,
    select_block_snapshot_tokens,
)
from nanoquant.domain.models import BlockId
from nanoquant.infrastructure.block_snapshot_probe import (
    capture_block_output_reference,
    measure_block_output_mse,
)


class _ToyBlocks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.layers = nn.ModuleList((nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def _forward(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens)


def test_snapshot_selection_is_bounded_versioned_and_content_identified() -> None:
    tokens = torch.arange(30).reshape(5, 6)
    first = select_block_snapshot_tokens(tokens, maximum_samples=3, maximum_tokens=4, pad_token_id=0)
    second = select_block_snapshot_tokens(tokens.clone(), maximum_samples=3, maximum_tokens=4, pad_token_id=0)
    different_padding = select_block_snapshot_tokens(
        tokens,
        maximum_samples=3,
        maximum_tokens=4,
        pad_token_id=None,
    )

    assert first.token_ids.shape == (3, 4)
    assert first.protocol == second.protocol
    assert first.protocol.version == "block-hidden-unweighted-mse-v1"
    assert first.protocol.semantic_key.startswith("sha256:")
    assert first.protocol.sample_count == 3
    assert first.protocol.sequence_length == 4
    assert first.protocol.reference_dtype == "bfloat16"
    assert first.protocol.accumulation_dtype == "float32"
    assert first.protocol.pad_token_id == 0
    assert first.protocol.semantic_key != different_padding.protocol.semantic_key


def test_probe_compares_same_teacher_reference_before_and_after_kd() -> None:
    torch.manual_seed(7)
    teacher = _ToyBlocks()
    student = _ToyBlocks()
    student.load_state_dict(teacher.state_dict())
    tokens = torch.tensor(((1, 2, 3, 0), (4, 5, 0, 0)))
    blocks = tuple(teacher.layers)
    reference = capture_block_output_reference(teacher, blocks, tokens, _forward, device="cpu")

    exact = measure_block_output_mse(
        student,
        tuple(student.layers),
        tokens,
        reference,
        _forward,
        device="cpu",
        pad_token_id=0,
    )
    with torch.no_grad():
        student.layers[0].weight.add_(0.25)
    changed = measure_block_output_mse(
        student,
        tuple(student.layers),
        tokens,
        reference,
        _forward,
        device="cpu",
        pad_token_id=0,
    )

    assert exact == (0.0, 0.0)
    assert changed[0] > 0
    assert changed[1] > 0


def test_named_post_kd_comparison_preserves_near_zero_relative_semantics() -> None:
    selection = select_block_snapshot_tokens(
        torch.tensor(((1, 2),)), maximum_samples=1, maximum_tokens=2, denominator_floor=1e-6
    )
    metrics = compare_block_snapshots(
        (BlockId(0), BlockId(1)),
        (0.0, 2.0),
        (1.0, 1.0),
        selection.protocol,
    )

    assert metrics[0].post_kd_vs_pre_kd.baseline_name == "final_frozen_pre_kd"
    assert metrics[0].post_kd_vs_pre_kd.candidate_name == "final_post_kd"
    assert metrics[0].post_kd_vs_pre_kd.absolute_delta == 1.0
    assert metrics[0].post_kd_vs_pre_kd.relative_delta is None
    assert metrics[1].post_kd_vs_pre_kd.relative_delta == -0.5


def test_snapshot_selection_rejects_invalid_limits_and_padding() -> None:
    tokens = torch.tensor(((1, 2),))

    with pytest.raises(ValueError, match="limits must be positive"):
        select_block_snapshot_tokens(tokens, maximum_samples=0, maximum_tokens=2)
    with pytest.raises(ValueError, match="pad token ID"):
        select_block_snapshot_tokens(tokens, maximum_samples=1, maximum_tokens=2, pad_token_id=-1)
