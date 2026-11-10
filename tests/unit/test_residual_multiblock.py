"""Fresh-objective and differentiable residual-stage research checks."""

import pytest
import torch
from probe_residual_multiblock_fit import normalized_importance
from probe_residual_multiblock_functional import FirstStageScaleOverlay, full_kl_hidden_gradient
from probe_residual_stage_budget import decode_payload, encode_payload, reconstruct


def test_fresh_importance_is_scale_invariant_and_positive():
    value = torch.tensor([0., 2., 4.])
    torch.testing.assert_close(normalized_importance(value), normalized_importance(value * 17))
    assert normalized_importance(value)[0] == pytest.approx(1e-6)
    for invalid in (torch.zeros(3), torch.tensor([-1., 2.]), torch.tensor([float("nan")])):
        with pytest.raises(ValueError, match="importance"):
            normalized_importance(invalid)


def test_chunked_full_kl_gradient_matches_whole_vocabulary_autograd():
    torch.manual_seed(3)
    student = torch.nn.Linear(4, 11, bias=False).requires_grad_(False)
    teacher = torch.nn.Linear(4, 11, bias=False).requires_grad_(False)
    hidden = torch.randn(2, 7, 4, requires_grad=True)
    teacher_hidden = torch.randn(2, 7, 4)
    expected_log = torch.log_softmax(teacher(teacher_hidden[:, :-1]), dim=-1)
    actual_log = torch.log_softmax(student(hidden[:, :-1]), dim=-1)
    loss = (expected_log.exp() * (expected_log - actual_log)).sum(-1).mean()
    expected = torch.autograd.grad(loss, hidden)[0]
    gradient, value = full_kl_hidden_gradient(hidden, teacher_hidden, student, teacher, chunk=2)
    torch.testing.assert_close(gradient, expected)
    assert value == pytest.approx(float(loss.detach()), abs=1e-7)
    assert torch.count_nonzero(gradient[:, -1]) == 0


def test_scale_overlay_preserves_frozen_correction_and_protected_values_after_persistence():
    state = dict(zip(("first.left_binary", "first.right_binary", "first.scale_pre",
                      "first.scale_mid", "first.scale_post", "protected.indices", "protected.values",
                      "correction.indices", "correction.int8", "correction.column_scale"),
                     (torch.tensor([[1, -1], [-1, 1]], dtype=torch.int8),
                      torch.tensor([[1, -1, 1], [-1, -1, 1]], dtype=torch.int8),
                      torch.tensor([2., 3., 4.]), torch.tensor([0.5, 0.75]), torch.tensor([2., 3.]),
                      torch.tensor([1], dtype=torch.int32), torch.tensor([[21.], [22.]]),
                      torch.tensor([0], dtype=torch.int32), torch.tensor([[1], [-2]], dtype=torch.int8),
                      torch.tensor([0.25])), strict=True))
    module = FirstStageScaleOverlay(state)
    original_bytes = len(encode_payload(state))
    with torch.no_grad():
        module.delta_pre.copy_(torch.tensor([0.02, -0.03, 0.04]))
        module.delta_mid.fill_(0.01)
        module.delta_post.fill_(-0.02)
    saved = decode_payload(encode_payload(module.persisted_state()))
    expected = reconstruct(saved).float()
    actual = module(torch.eye(3)).T
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 1], state["protected.values"][:, 0])
    assert len(encode_payload(saved)) == original_bytes
    actual.square().sum().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert sum(parameter.numel() for parameter in module.parameters()) == 7
    for name, value in state.items():
        if not name.startswith("first.scale_"):
            assert torch.equal(saved[name], value)
