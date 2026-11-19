"""A split can replay one fitted stage without sharing trainable storage."""

import pytest
import torch
from probe_residual_rank_split import frozen_payload, payload_weight
from probe_residual_split_warm_start import replay, split_module

from nanoquant.application.layers import TrainableFactorizedLinear


def test_split_replays_nonsquare_factor_and_copies_scales():
    rng = torch.Generator().manual_seed(12)
    parent = TrainableFactorizedLinear(
        torch.randn(7, 5, generator=rng), torch.randn(5, 11, generator=rng),
        torch.randn(11, generator=rng), torch.randn(5, generator=rng), torch.randn(7, generator=rng),
    )
    divided = split_module(parent, 2)
    assert divided.left_latent.shape == (7, 2)
    assert divided.residual_terms[0].right_latent.shape == (3, 11)
    inputs = torch.randn(2, 3, 11, generator=rng)
    torch.testing.assert_close(divided(inputs), parent(inputs), rtol=1e-5, atol=1e-5)
    assert replay(parent, divided, inputs, "cpu")["relative_weight_error"] < 1e-6
    original = payload_weight(frozen_payload(parent), "cpu")
    with torch.no_grad():
        divided.residual_terms[0].scale_pre.add_(1)
    torch.testing.assert_close(divided.scale_pre, parent.scale_pre)
    torch.testing.assert_close(payload_weight(frozen_payload(parent), "cpu"), original)
    assert not torch.allclose(payload_weight(frozen_payload(divided), "cpu"), original)
    with pytest.raises(ValueError):
        split_module(divided, 1)
