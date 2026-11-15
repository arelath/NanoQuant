import torch
from probe_weight_fisher_imatrix import exact_gradient_square, paired_interval, quantize, tiles, untile


def test_fisher_sums_context_uses_before_square_and_targets_after_square():
    weight = torch.tensor([[2.0]], requires_grad=True)
    # Two contextual uses cancel within target 1; target 2 has gradient +3.
    first = (weight * 2).sum() - (weight * 2).sum()
    second = (weight * 3).sum()
    a, = exact_gradient_square(first, (weight,))
    b, = exact_gradient_square(second, (weight,))
    assert torch.equal((a + b) / 2, torch.tensor([[4.5]]))
    # Across targets, +3 and -3 must NOT cancel before squaring.
    c, = exact_gradient_square(-(weight * 3).sum(), (weight,))
    assert torch.equal((b + c) / 2, torch.tensor([[9.0]]))


def test_tiles_roundtrip_and_equal_rate_weighted_objective():
    generator = torch.Generator().manual_seed(14)
    weight = torch.randn(16, 24, generator=generator)
    importance = torch.rand(16, 24, generator=generator).square()
    assert torch.equal(untile(tiles(weight), weight.shape), weight)
    uniform = quantize(weight, torch.ones_like(weight), 3)
    weighted = quantize(weight, importance, 3)
    assert ((weighted - weight).square() * importance).sum() <= ((uniform - weight).square() * importance).sum()
    assert all(len(row.unique()) <= 8 for row in tiles(weighted))
    assert torch.equal(quantize(torch.zeros_like(weight), importance, 4), torch.zeros_like(weight))


def test_paired_bootstrap_constant_difference():
    assert paired_interval([1, 2, 3], [2, 3, 4]) == dict(mean=-1, low=-1, high=-1)


def test_model_target_fisher_matches_predictive_kl_curvature():
    weight = torch.tensor(0.7, requires_grad=True)
    logits = torch.stack((weight * 2, weight * 0))
    reference = logits.detach().softmax(-1)
    fisher = 0.0
    for target in range(2):
        square, = exact_gradient_square(-logits.log_softmax(-1)[target], (weight,), retain_graph=True)
        fisher = fisher + reference[target] * square

    def predictive_kl(value):
        candidate = torch.stack((value * 2, value * 0)).log_softmax(-1)
        return (reference * (reference.log() - candidate)).sum()

    curvature = torch.autograd.functional.hessian(predictive_kl, weight)
    assert torch.allclose(fisher, curvature, atol=1e-6)
