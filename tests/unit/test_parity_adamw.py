import pytest
import torch

from nanoquant.application.parity_adamw import ParityAdamW, _debiased_beta


def test_debiased_beta_matches_closed_form() -> None:
    assert _debiased_beta(0.9, 1) == 0.0
    assert _debiased_beta(0.9, 2) == pytest.approx(0.4736842105263158)


def test_float32_update_matches_reference_recurrence() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = ParityAdamW([parameter], lr=1e-2)
    exp_avg = torch.zeros_like(parameter)
    exp_avg_sq = torch.zeros_like(parameter)
    expected = parameter.detach().clone()
    for step, gradient in enumerate((torch.tensor([0.25, -0.5]), torch.tensor([-0.1, 0.2])), start=1):
        parameter.grad = gradient.clone()
        optimizer.step()
        beta1 = _debiased_beta(0.9, step)
        beta2 = _debiased_beta(0.99, step)
        exp_avg.lerp_(gradient, weight=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        expected.addcdiv_(exp_avg, exp_avg_sq.sqrt().add(1e-6), value=-1e-2)
    assert torch.equal(parameter, expected)


def test_bfloat16_update_retains_sub_ulp_steps_with_kahan_compensation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = ParityAdamW([parameter], lr=1e-3)
    for _ in range(16):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    assert parameter.item() < 0.99
