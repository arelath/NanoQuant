import pytest
import torch

from nanoquant.application.parity_adamw import ParityAdamW, _debiased_beta


def _legacy_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    compensation: torch.Tensor | None,
    *,
    step: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    beta1 = _debiased_beta(0.9, step)
    beta2 = _debiased_beta(0.99, step)
    if weight_decay:
        parameter.mul_(1.0 - learning_rate * weight_decay)
    exp_avg.lerp_(gradient, weight=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
    denominator = exp_avg_sq.sqrt().add_(1e-6)
    if compensation is not None:
        compensation.addcdiv_(exp_avg, denominator, value=-learning_rate)
        gradient.copy_(parameter)
        parameter.add_(compensation)
        compensation.add_(gradient.sub_(parameter))
    else:
        parameter.addcdiv_(exp_avg, denominator, value=-learning_rate)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_foreach_update_is_bitwise_equal_to_legacy_loop(dtype: torch.dtype, weight_decay: float) -> None:
    generator = torch.Generator().manual_seed(17)
    parameters = [
        torch.nn.Parameter(torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype))
        for shape in ((3, 5), (7,), (2, 3, 2))
    ]
    expected = [parameter.detach().clone() for parameter in parameters]
    exp_avgs = [torch.zeros_like(parameter) for parameter in expected]
    exp_avg_sqs = [torch.zeros_like(parameter) for parameter in expected]
    compensations = [torch.zeros_like(parameter) if dtype == torch.bfloat16 else None for parameter in expected]
    optimizer = ParityAdamW(parameters, lr=3e-3, weight_decay=weight_decay)

    with torch.no_grad():
        for step in range(1, 9):
            gradients = [
                torch.randn(parameter.shape, generator=generator, dtype=torch.float32).to(dtype)
                for parameter in parameters
            ]
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.grad = gradient.clone()
            optimizer.step()
            for parameter, gradient, exp_avg, exp_avg_sq, compensation in zip(
                expected, gradients, exp_avgs, exp_avg_sqs, compensations, strict=True
            ):
                _legacy_step(
                    parameter,
                    gradient,
                    exp_avg,
                    exp_avg_sq,
                    compensation,
                    step=step,
                    learning_rate=3e-3,
                    weight_decay=weight_decay,
                )

    for actual, reference in zip(parameters, expected, strict=True):
        assert torch.equal(actual, reference)
