"""Tied scales update once, receive both gradients, and are stored once."""

import pytest
import torch
from probe_one_axis_shared_scales import restore_shared_payload, shared_cost, shared_payload, shared_split
from probe_residual_rank_split import frozen_payload, payload_weight, split_cost
from probe_residual_split_warm_start import split_module
from safetensors.torch import load_file, save_file

from nanoquant.application.layers import TrainableFactorizedLinear


@pytest.mark.parametrize("axis,dim", [("pre", 96), ("post", 64)])
def test_one_axis_gradient_and_persisted_cost(axis, dim, tmp_path):
    rng = torch.Generator().manual_seed(7)
    parent = TrainableFactorizedLinear(
        torch.randn(64, 64, generator=rng), torch.randn(64, 96, generator=rng),
        torch.randn(96, generator=rng), torch.randn(64, generator=rng), torch.randn(64, generator=rng),
    )
    control = split_module(parent, 32)
    tied = shared_split(parent, 32, axis)
    key = f"scale_{axis}"
    shared = getattr(tied, key)
    assert shared is getattr(tied.residual_terms[0], key)
    assert sum(p is shared for p in tied.parameters()) == 1
    x = torch.randn(3, 96, generator=rng)
    torch.testing.assert_close(tied(x), control(x))
    tied(x).square().mean().backward()
    control(x).square().mean().backward()
    torch.testing.assert_close(shared.grad, getattr(control, key).grad + getattr(control.residual_terms[0], key).grad)
    path = tmp_path / "shared.safetensors"
    save_file(shared_payload(tied, axis), str(path))
    saved = load_file(str(path))
    assert f"stage1.{key}" not in saved
    bits = sum(t.numel() * (1 if name.endswith("_binary") else 16) for name, t in saved.items())
    assert bits == shared_cost((64, 96), (32, 32), axis) == split_cost((64, 96), (32, 32)) - 16 * dim
    restored = restore_shared_payload(saved, axis)
    torch.testing.assert_close(payload_weight(restored, "cpu"), payload_weight(frozen_payload(tied), "cpu"))
    with pytest.raises(ValueError, match="untied"):
        shared_payload(control, axis)
