"""CPU checks for the standalone split screen (no production policy change)."""

import pytest
import torch
from probe_residual_rank_split import (
    candidate_splits,
    fit_split,
    frozen_payload,
    metrics,
    payload_weight,
    split_cost,
    tune_joint,
)
from safetensors.torch import load_file, save_file
from score_residual_rank_split import selected_pairs

from nanoquant.domain.factorization import AdmmParameters


def test_equal_total_and_stage_count_cost_and_single_stage_overhead():
    shape = (1152, 6912)
    splits = candidate_splits(shape, (1152, 768))
    assert splits == ((1152, 768), (960, 960), (768, 1152))
    assert len({split_cost(shape, ranks) for ranks in splits}) == 1
    single = candidate_splits(shape, (1152,))
    assert len(single) == 3
    assert all(split_cost(shape, r) <= split_cost(shape, (1152,)) for r in single)
    assert all(sum(r) < 1152 for r in single if len(r) == 2)
    assert candidate_splits(shape, (1152, 1152)) == ((1152, 1152),)
    three = candidate_splits(shape, (1152, 1152, 576))
    assert all(split_cost(shape, r) == split_cost(shape, three[0]) for r in three)
    with pytest.raises(ValueError):
        split_cost(shape, (33, 64))


def test_explicit_fit_roundtrip_and_all_stages_jointly_tuned(tmp_path):
    torch.manual_seed(4)
    target = torch.randn(16, 12) * 0.05
    v, u = torch.ones(12), torch.ones(16)
    module = fit_split(target, v, u, (6, 4), 1, AdmmParameters(outer_iterations=3))
    assert module.left_latent.shape == (16, 6)
    assert module.residual_terms[0].left_latent.shape == (16, 4)
    x = torch.randn(3, 8, 12)
    with torch.no_grad():
        torch.testing.assert_close(module(x), x @ module.dense_weight().T, atol=1e-6, rtol=1e-5)
    result = tune_joint(module, target, x, 2)
    assert result["steps"] == 6
    assert all(delta > 0 for delta in result["max_parameter_changes"].values())
    path = tmp_path / "payload.safetensors"
    save_file(frozen_payload(module), str(path))
    saved = load_file(str(path))
    assert all(t.dtype == torch.bfloat16 for k, t in saved.items() if "scale" in k)
    assert all(set(t.unique().tolist()) <= {-1, 1} for k, t in saved.items() if "binary" in k)
    prediction = payload_weight(saved, "cpu")
    measured = metrics(prediction, target, v, u, x)
    assert len(measured["activation_nmse_per_sequence"]) == 3
    assert measured["weighted_nmse"] >= 0
    assert metrics(target, target, v, u, x)["activation_nmse"] == 0


def test_nonfinite_score_fails_closed():
    with pytest.raises(FloatingPointError):
        metrics(torch.full((2, 2), float("nan")), torch.ones(2, 2),
                torch.ones(2), torch.ones(2), torch.ones(1, 3, 2))


def test_confirmation_pairs_preserve_seed_and_budget():
    rows = [dict(block=25, owner="mlp.down_proj", seed=seed, ranks=ranks,
                 bits=100, is_baseline=ranks == [64, 32])
            for seed in (0, 1) for ranks in ([64, 32], [32, 64])]
    pairs = selected_pairs(rows, 25, "mlp.down_proj", [32, 64])
    assert set(pairs) == {0, 1}
    assert all(a["seed"] == b["seed"] == seed for seed, (a, b) in pairs.items())
    rows[1]["bits"] = 101
    with pytest.raises(ValueError, match="not cost more"):
        selected_pairs(rows, 25, "mlp.down_proj", [32, 64])
