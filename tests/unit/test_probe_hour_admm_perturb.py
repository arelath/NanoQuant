"""CPU guards for candidate initialization and production ADMM parity."""

import time
from argparse import Namespace

import pytest
import torch
from probe_hour_admm_perturb import admm_fit, eligible_improvement, lift_candidate, perturb, run
from probe_hour_factor_fit import STATE_KEYS, BudgetExpired, state_of
from test_probe_hour_factor_fit import _case

from nanoquant.domain.factorization import factorize_admm


@pytest.mark.parametrize("iterations", [0, 1, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_random_adapter_exactly_matches_production(iterations, dtype):
    fixture, _ = _case()
    expected = factorize_admm(
        fixture["target"].to(dtype), fixture["input_importance"], fixture["output_importance"], 2,
        torch.Generator().manual_seed(77), outer_iterations=iterations,
    )
    actual, trace = admm_fit(fixture, 2, 77, iterations=iterations, dtype=dtype)
    for key, tensor in state_of(expected).items():
        assert torch.equal(actual[key], tensor), key
    assert len(trace) == len(expected.trace)
    for row, point in zip(trace, expected.trace, strict=True):
        assert row["primal_residual"] == point.primal_residual
        assert row["dual_residual"] == point.dual_residual


def test_lift_preserves_signed_scale_prediction_in_weighted_coordinates():
    fixture, state = _case()
    state["scale_pre"][0] *= -1
    state["scale_post"][0] *= -1
    left, right = lift_candidate(fixture, state)
    prediction = (state["left_binary"] * state["scale_post"][:, None]) @ (
        state["right_binary"] * state["scale_mid"][:, None] * state["scale_pre"][None, :])
    weighted = prediction * fixture["output_importance"].sqrt()[:, None]
    weighted *= fixture["input_importance"].sqrt()[None, :]
    torch.testing.assert_close(left @ right, weighted)
    torch.testing.assert_close(left.norm(dim=0), torch.ones(2))


def test_zero_iteration_warm_start_preserves_candidate_not_random_initialization():
    fixture, state = _case()
    actual, _ = admm_fit(fixture, 2, 12, initial=state, iterations=0, dtype=torch.float32)
    def reconstruct(x):
        left, right, pre, mid, post = (x[k] for k in STATE_KEYS)
        return (left * post[:, None]) @ (right * mid[:, None] * pre[None, :])
    torch.testing.assert_close(reconstruct(actual), reconstruct(state), rtol=2e-5, atol=1e-6)


def test_perturbation_is_reproducible_retains_scales_and_changes_both_factors():
    _, small = _case()
    state = {**small, "left_binary": torch.ones(100, 100), "right_binary": torch.ones(100, 300)}
    first, counts = perturb(state, 68000)
    second, other = perturb(state, 68000)
    assert counts == other
    for key in STATE_KEYS:
        assert torch.equal(first[key], second[key])
        assert torch.equal(state[key], small[key] if key in STATE_KEYS[2:] else torch.ones_like(state[key]))
    for key in STATE_KEYS[:2]:
        assert 0.08 < counts[key] / state[key].numel() < 0.12
    for key in STATE_KEYS[2:]:
        assert torch.equal(first[key], state[key])


def test_deadline_never_returns_partial_admm_as_completed_fit():
    fixture, state = _case()
    with pytest.raises(BudgetExpired):
        admm_fit(fixture, 2, 12, initial=state, deadline=time.perf_counter() - 1)


def test_overbudget_improvement_cannot_replace_best():
    assert eligible_improvement(1, 2, 3599, 3600)
    assert not eligible_improvement(1, 2, 3601, 3600)
    assert not eligible_improvement(3, 2, 3599, 3600)


def test_cancelled_directory_does_not_restart_or_acquire_cuda(tmp_path, capsys):
    (tmp_path / "cancellation.json").write_text('{}')
    # No source, device, or other arguments are needed: cancellation is checked first.
    run(Namespace(output=tmp_path))
    assert "cancelled" in capsys.readouterr().out
