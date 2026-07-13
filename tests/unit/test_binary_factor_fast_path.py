import torch

from nanoquant.application.layers import TrainableFactorizedLinear


def _module(*, immutable_binary_factors: bool) -> TrainableFactorizedLinear:
    return TrainableFactorizedLinear(
        torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]),
        torch.tensor([0.75, 1.0, 1.25, 1.5]),
        torch.tensor([0.5, 1.5]),
        torch.tensor([1.25, 0.75, 1.5]),
        immutable_binary_factors=immutable_binary_factors,
    )


def test_immutable_binary_factor_fast_path_preserves_scale_training_bitwise() -> None:
    control = _module(immutable_binary_factors=False)
    optimized = _module(immutable_binary_factors=True)
    for module in (control, optimized):
        module.left_latent.requires_grad_(False)
        module.right_latent.requires_grad_(False)
    values = torch.tensor(
        [[[0.25, -0.5, 1.0, 0.75], [-1.0, 0.5, 0.125, -0.25]]],
        requires_grad=True,
    )

    control_output = control(values)
    control_output.square().sum().backward()
    optimized_output = optimized(values.detach().clone().requires_grad_(True))
    optimized_output.square().sum().backward()

    assert torch.equal(optimized_output, control_output)
    for name in ("scale_pre", "scale_mid", "scale_post"):
        assert torch.equal(getattr(optimized, name).grad, getattr(control, name).grad)


def test_binary_factor_marker_keeps_ste_when_factors_are_trainable() -> None:
    module = _module(immutable_binary_factors=True)
    values = torch.ones(1, 1, 4)

    module(values).sum().backward()

    assert module.left_latent.grad is not None
    assert module.right_latent.grad is not None
