import math

import pytest
import torch
from compare_imatrix_shaped_noise import MODES, prediction_metrics, residual_metrics, shaped_noise


@pytest.mark.parametrize("mode", MODES)
def test_shaped_noise_preserves_requested_norm_and_is_reproducible(mode):
    original = torch.arange(1, 257).float().reshape(8, 32)
    importance = torch.linspace(0, 2, 32)
    kwargs = dict(mode=mode, seed=17, norm=3.5, floor=0.01, block_size=8)
    noise = shaped_noise(original, importance, **kwargs)
    assert torch.isfinite(noise).all()
    assert float(noise.norm()) == pytest.approx(3.5, rel=1e-6)
    assert torch.equal(noise, shaped_noise(original, importance, **kwargs))


def test_inverse_importance_moves_energy_to_less_important_columns():
    original = torch.ones(4096, 2)
    importance = torch.tensor([1., 100.])
    kwargs = dict(seed=5, norm=1., floor=0.01, block_size=1)
    plain = shaped_noise(original, importance, mode="isotropic", **kwargs)
    shaped = shaped_noise(original, importance, mode="imatrix", **kwargs)
    raw_ratio = float(plain[:, 0].square().sum() / plain[:, 1].square().sum())
    shaped_ratio = float(shaped[:, 0].square().sum() / shaped[:, 1].square().sum())
    assert shaped_ratio / raw_ratio == pytest.approx(100, rel=1e-5)
    assert residual_metrics(original, original + shaped, importance)["weighted_error"] < (
        residual_metrics(original, original + plain, importance)["weighted_error"])


def test_uniform_importance_and_constant_local_scales_reduce_to_isotropic():
    original = torch.ones(8, 32)
    kwargs = dict(seed=3, norm=0.1, floor=0.01, block_size=8)
    probes = [shaped_noise(original, torch.ones(32), mode=mode, **kwargs) for mode in MODES]
    assert all(torch.equal(probes[0], probe) for probe in probes[1:])


def test_block_scaling_preserves_local_weight_energy_ratio():
    original = torch.tensor([[1., 1., 10., 10.]]).repeat(1024, 1)
    kwargs = dict(seed=8, norm=1., floor=0.01, block_size=2)
    plain = shaped_noise(original, torch.ones(4), mode="isotropic", **kwargs)
    block = shaped_noise(original, torch.ones(4), mode="block", **kwargs)
    def ratio(x):
        return float(x[:, 2:].square().sum() / x[:, :2].square().sum())

    assert ratio(block) / ratio(plain) == pytest.approx(100, rel=1e-5)


@pytest.mark.parametrize("importance", [torch.zeros(4), torch.tensor([1., -1., 1., 1.]),
                                        torch.tensor([1., math.nan, 1., 1.])])
def test_invalid_importance_is_rejected(importance):
    with pytest.raises(ValueError, match="invalid"):
        shaped_noise(torch.ones(2, 4), importance, mode="imatrix", seed=0,
                     norm=1., floor=0.01, block_size=2)


def test_aggregate_agreement_does_not_hide_individual_prediction_error():
    metrics = prediction_metrics([2., 1.], [1., 2.])
    assert metrics["sum_ratio"] == 1
    assert metrics["median_fold_error"] == 2
    assert metrics["spearman"] == pytest.approx(-1)
