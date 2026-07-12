from copy import deepcopy

import pytest
import torch
from torch import nn

from nanoquant.application.distillation import (
    TopKDistillationConfig,
    cache_topk_teacher_epoch,
    cache_topk_teacher_targets,
    distill_topk,
    teacher_topk_logits,
    topk_distillation_loss,
)


class ToyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 6)
        self.projection = nn.Linear(6, 6, bias=False)
        self.norm = nn.LayerNorm(6)
        self.lm_head = nn.Linear(6, 17, bias=False)

    def hidden_states(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(self.embedding(token_ids)))


def _hidden(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    assert isinstance(model, ToyLanguageModel)
    return model.hidden_states(token_ids)


def test_chunked_teacher_topk_matches_dense_logits() -> None:
    generator = torch.Generator().manual_seed(4)
    head = nn.Linear(5, 19, bias=True)
    hidden = torch.randn(7, 5, generator=generator)

    values, indices = teacher_topk_logits(
        hidden,
        head,
        top_k=6,
        vocabulary_chunk_size=4,
        temperature=0.7,
    )
    expected_values, expected_indices = torch.topk(head(hidden) / 0.7, 6, dim=-1)

    assert torch.allclose(values, expected_values)
    assert torch.equal(indices, expected_indices)


def test_topk_loss_matches_selected_teacher_cross_entropy() -> None:
    generator = torch.Generator().manual_seed(5)
    head = nn.Linear(4, 13, bias=False)
    teacher_hidden = torch.randn(8, 4, generator=generator)
    student_hidden = torch.randn(8, 4, generator=generator)
    teacher_values, teacher_indices = torch.topk(head(teacher_hidden), 5, dim=-1)

    actual = topk_distillation_loss(
        student_hidden,
        teacher_values,
        teacher_indices,
        head,
        temperature=1.0,
        token_chunk_size=3,
    )
    selected_weights = head.weight.index_select(0, teacher_indices.reshape(-1)).view(8, 5, 4)
    selected_student_logits = torch.bmm(selected_weights, student_hidden.unsqueeze(-1)).squeeze(-1)
    expected = -(
        torch.softmax(teacher_values, dim=-1) * torch.log_softmax(selected_student_logits, dim=-1)
    ).sum(dim=-1).mean()

    assert float(actual.detach()) == pytest.approx(float(expected.detach()))


def test_cached_topk_distillation_is_bounded_deterministic_and_improves_student() -> None:
    torch.manual_seed(7)
    teacher = ToyLanguageModel()
    student = deepcopy(teacher)
    with torch.no_grad():
        student.projection.weight.add_(
            0.35 * torch.randn(student.projection.weight.shape, generator=torch.Generator().manual_seed(8))
        )
    tokens = torch.randint(1, 17, (8, 7), generator=torch.Generator().manual_seed(9))
    tokens[0, -2:] = 0
    config = TopKDistillationConfig(
        epochs=8,
        batch_size=2,
        learning_rate=0.04,
        top_k=8,
        vocabulary_chunk_size=5,
        token_chunk_size=4,
        maximum_tokens_per_batch=6,
        weight_decay=0.0,
        seed=10,
    )

    cache = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        device="cpu",
        pad_token_id=0,
    )
    repeated = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        device="cpu",
        pad_token_id=0,
    )
    resumed_epoch, resumed_bytes = cache_topk_teacher_epoch(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        epoch_index=5,
        device="cpu",
        pad_token_id=0,
    )
    untouched_embedding = student.embedding.weight.detach().clone()
    metrics = distill_topk(
        student,
        tokens,
        student.lm_head,
        _hidden,
        cache,
        config,
        lambda name, _parameter: name == "projection.weight",
        device="cpu",
    )

    assert cache.bytes > 0
    assert resumed_bytes == sum(
        value.numel() * value.element_size()
        for batch in cache.epochs[5]
        for value in (batch.token_indices, batch.top_values, batch.top_indices)
    )
    assert all(
        torch.equal(left.token_indices, right.token_indices)
        and torch.equal(left.top_values, right.top_values)
        and torch.equal(left.top_indices, right.top_indices)
        for left, right in zip(cache.epochs[5], resumed_epoch, strict=True)
    )
    assert all(target.token_indices.numel() <= 6 for epoch in cache.epochs for target in epoch)
    assert all(
        torch.equal(left.token_indices, right.token_indices)
        and torch.equal(left.top_values, right.top_values)
        and torch.equal(left.top_indices, right.top_indices)
        for left_epoch, right_epoch in zip(cache.epochs, repeated.epochs, strict=True)
        for left, right in zip(left_epoch, right_epoch, strict=True)
    )
    assert metrics.steps_completed == 32
    assert metrics.selected_parameter_count == 1
    assert metrics.epoch_losses[-1] < metrics.epoch_losses[0]
    assert torch.equal(student.embedding.weight, untouched_embedding)
