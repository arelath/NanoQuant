"""Check fixed-outer residual fitting and the fused gain's joint gradients."""

import torch
from probe_residual_rank_split import frozen_payload, payload_weight
from probe_shared_scale_residual_fit import JointResidualGain, fit_residual, output_ratio

from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.domain.factorization import AdmmParameters


def test_residual_preserves_primary_and_gain_folds():
    primary = TrainableFactorizedLinear(
        torch.tensor([[1.], [-1.], [1.]]), torch.tensor([[1., -1., 1., 1.]]),
        torch.tensor([-0.5, 1., 1.5, 2.]), torch.tensor([0.125]), torch.tensor([1., 0.5, -2.]),
    )
    before = frozen_payload(primary)
    weight = payload_weight(before, "cpu")
    target = (weight + torch.tensor([[0.125, 0.5, 0.25, -0.25]])).bfloat16()
    left, right, fit, original = fit_residual(
        primary, target, torch.ones(4), torch.ones(3), 1, 3, AdmmParameters(outer_iterations=6),
    )
    assert fit.accepted and fit.after_error < fit.before_error
    assert all(torch.equal(t, before[k]) for k, t in frozen_payload(primary).items())
    torch.testing.assert_close(original, weight)
    x = torch.eye(4)
    ratio = output_ratio(x, weight, weight + fit.correction)
    module = JointResidualGain(primary, left, right, fit.scale_mid, 0.03 / ratio)
    folded = module.folded_payload()
    combined = payload_weight(folded, "cpu")
    torch.testing.assert_close(module(x.bfloat16()).float(), x @ combined.mT, atol=0.005, rtol=0.01)
    assert abs(output_ratio(x, weight, combined) - 0.03) < 0.001
    loss = (module(x.bfloat16()).float() - x @ target.float().mT).square().sum()
    loss.backward()
    assert module.gain.grad is not None and module.gain.grad.abs() > 0
    assert all(p.grad is not None for p in module.factor.parameters())
    assert module.factor.scale_mid.grad[:1].abs().sum() > 0
    assert module.factor.scale_mid.grad[1:].abs().sum() > 0
