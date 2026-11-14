from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["term", "host", "block"])
def test_packed_residual_training_keeps_forward_gradients_and_signs(dtype: torch.dtype, mode: str) -> None:
    control = _module(immutable_binary_factors=True).to(dtype)
    control.residual_stages.append(deepcopy(control))
    for name, parameter in control.named_parameters():
        if name.endswith("_latent"):
            parameter.requires_grad_(False)
    packed = deepcopy(control)
    for term in (packed, *packed.residual_terms):
        term.pack_fixed_signs()
        term.checkpoint_fixed_signs = mode != "block"
    assert all(not name.endswith("_latent") for name, _ in packed.named_parameters())
    assert not any("latent" in name for name in packed.state_dict())
    for original, compact in zip((control, *control.residual_terms), (packed, *packed.residual_terms), strict=True):
        assert compact.left_latent.dtype == torch.uint8
        for a, b in zip(original.export_signs(), compact.export_signs(), strict=True):
            assert torch.equal(a, b)
    values = torch.tensor([[[0.25, -0.5, 1.0, 0.75]]], dtype=dtype)
    x = values.clone().requires_grad_()
    y = values.clone().requires_grad_()
    expected = control(x)
    with torch.autograd.graph.save_on_cpu() if mode == "host" else nullcontext():
        actual = checkpoint(packed, y, use_reentrant=True) if mode == "block" else packed(y)
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, y.grad)
    original_parameters = dict(control.named_parameters())
    for name, parameter in packed.named_parameters():
        assert torch.equal(parameter.grad, original_parameters[name].grad)
    assert torch.equal(packed.dense_weight(), control.dense_weight())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_resident_signs_restore_the_unpacked_term_without_changing_training(dtype: torch.dtype) -> None:
    control = _module(immutable_binary_factors=True).to(dtype)
    control.residual_stages.append(deepcopy(control))
    for name, parameter in control.named_parameters():
        if name.endswith("_latent"):
            parameter.requires_grad_(False)
    resident = deepcopy(control)
    terms = (resident, *resident.residual_terms)
    assert all(term.fixed_sign_residency_bytes() == 0 for term in terms)
    for term in terms:
        term.pack_fixed_signs()
    # Six left and eight right signs decode to fourteen scalars and free two packed bytes.
    assert all(term.fixed_sign_residency_bytes() == 14 * control.scale_pre.element_size() - 2 for term in terms)
    for term in terms:
        term.unpack_fixed_signs()
        term.unpack_fixed_signs()

    assert all(term.fixed_sign_residency_bytes() == 0 for term in terms)
    assert {name for name, _ in resident.named_parameters()} == {
        name for name, _ in control.named_parameters() if not name.endswith("_latent")
    }
    assert not any("latent" in name for name in resident.state_dict())
    for original, restored in zip((control, *control.residual_terms), terms, strict=True):
        assert restored.left_latent.dtype is dtype
        assert torch.equal(restored.left_latent, original.left_latent)
        assert torch.equal(restored.right_latent, original.right_latent)
        for a, b in zip(original.export_signs(), restored.export_signs(), strict=True):
            assert torch.equal(a, b)
    values = torch.tensor([[[0.25, -0.5, 1.0, 0.75]]], dtype=dtype)
    x = values.clone().requires_grad_()
    y = values.clone().requires_grad_()
    expected = control(x)
    actual = checkpoint(resident, y, use_reentrant=True)
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, y.grad)
    original_parameters = dict(control.named_parameters())
    for name, parameter in resident.named_parameters():
        assert torch.equal(parameter.grad, original_parameters[name].grad)
    assert torch.equal(resident.dense_weight(), control.dense_weight())
