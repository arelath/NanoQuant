"""Persisted-scale and objective-gradient contracts for the continuation probe."""

import pytest
import torch
from objective_ablation_common import ScaleUnit, UnitSlice, bf16_ste, normalized_hidden_loss
from probe_distillation_objectives import topk_hidden_gradient

from nanoquant.application.distillation import topk_distillation_loss


def test_bf16_qat_uses_stored_values_and_master_gradient():
    value = torch.tensor([1.003, 0.01734], requires_grad=True)
    stored = bf16_ste(value)
    assert torch.equal(stored, value.detach().bfloat16().float())
    stored.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_fixed_sign_shared_unit_preserves_salient_columns_and_gradients():
    unit = ScaleUnit(torch.tensor([[1, -1], [-1, 1]]), torch.tensor([[1, 1, -1], [-1, 1, 1]]),
                     torch.tensor([0.2, 0.3, 0.4]), torch.tensor([0.5, 0.75]), torch.tensor([2., 3.]),
                     torch.tensor([1]), torch.tensor([[7.], [11.]]))
    left, right = UnitSlice(unit, 0, 1), UnitSlice(unit, 1, 2)
    inputs = torch.eye(3)
    actual = torch.cat((left(inputs), right(inputs)), dim=1).T
    assert torch.equal(actual[:, 1], torch.tensor([7., 11.]))
    actual.square().sum().backward()
    assert all(p.grad is not None for p in unit.parameters())
    assert unit.pre.grad[1] == 0
    assert unit.left.dtype == torch.int8 and unit.right.dtype == torch.int8
    assert all(value.dtype == torch.bfloat16 for value in unit.scales().values())


def test_hidden_anchor_normalizes_by_teacher_energy_and_excludes_embedding():
    teacher = (torch.ones(2), torch.tensor([2., 4.]), torch.tensor([3., 6.]))
    student = tuple((value * 1.25).requires_grad_(True) for value in teacher)
    loss = normalized_hidden_loss(student, teacher)
    assert float(loss.detach()) == pytest.approx(0.25**2)
    loss.backward()
    assert student[0].grad is None and student[1].grad is not None
    scaled_teacher = tuple(value * 17 for value in teacher)
    scaled_student = tuple(value.detach() * 17 for value in student)
    assert float(normalized_hidden_loss(scaled_student, scaled_teacher)) == pytest.approx(float(loss.detach()))


def test_chunked_topk_gradient_matches_production_full_sequence_loss():
    torch.manual_seed(2)
    head = torch.nn.Linear(4, 73, bias=False).requires_grad_(False)
    teacher_head = torch.nn.Linear(4, 73, bias=False).requires_grad_(False)
    hidden = torch.randn(2, 35, 4, requires_grad=True)
    teacher_hidden = torch.randn_like(hidden)
    values, indices = teacher_head(teacher_hidden[:, :-1]).reshape(-1, 73).topk(64)
    loss = topk_distillation_loss(hidden[:, :-1].reshape(-1, 4), values, indices, head,
                                 temperature=1, token_chunk_size=32)
    expected = torch.autograd.grad(loss, hidden)[0]
    gradient, measured = topk_hidden_gradient(hidden, teacher_hidden, head, teacher_head)
    torch.testing.assert_close(gradient, expected)
    assert measured == pytest.approx(float(loss.detach()))
