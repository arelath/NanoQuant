"""CPU mathematical guards for the basin-search proposals and final certificate."""

import pytest
import torch
from probe_hour_basin_search import accept_walker, flip_costs, neighborhood_audit, perturb
from probe_hour_factor_fit import STATE_KEYS, score, validate_state


def case():
    generator = torch.Generator().manual_seed(67)
    state = {
        "left_binary": torch.where(torch.randn(3, 2, generator=generator) > 0, 1.0, -1.0),
        "right_binary": torch.where(torch.randn(2, 4, generator=generator) > 0, 1.0, -1.0),
        "scale_pre": torch.randn(4, generator=generator),
        "scale_mid": torch.randn(2, generator=generator),
        "scale_post": torch.randn(3, generator=generator),
    }
    fixture = {
        "target": torch.randn(3, 4, generator=generator),
        "input_importance": torch.rand(4, generator=generator),
        "output_importance": torch.rand(3, generator=generator),
    }
    return fixture, state


def test_every_analytic_flip_cost_matches_independent_reconstruction():
    fixture, state = case()
    before = score(fixture, state, double=True)["weighted_error"]
    costs = flip_costs(fixture, state, double=True)
    improving = 0
    for key, cost in zip(STATE_KEYS[:2], costs, strict=True):
        for index in range(state[key].numel()):
            candidate = {k: v.clone() for k, v in state.items()}
            candidate[key].view(-1)[index] *= -1
            delta = score(fixture, candidate, double=True)["weighted_error"] - before
            assert float(cost.flatten()[index]) == pytest.approx(delta, abs=1e-12)
            improving += delta < -1e-10
    audit = neighborhood_audit(fixture, state)
    assert audit["improving_bits"] == improving
    assert audit["bits_examined"] == 14


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11])
def test_perturbations_are_reproducible_preserve_representation_and_do_not_mutate_parent(index):
    fixture, state = case()
    original = {k: v.clone() for k, v in state.items()}
    first, details = perturb(fixture, state, index)
    second, repeated = perturb(fixture, state, index)
    validate_state(first, (3, 4), 2)
    assert details == repeated
    for key in STATE_KEYS:
        assert torch.equal(first[key], second[key])
        assert torch.equal(state[key], original[key])
    for key in STATE_KEYS[2:]:
        assert torch.equal(first[key], state[key])


def test_walker_can_accept_uphill_without_altering_best_and_always_accepts_improvement():
    assert accept_walker(0.9, 1.0, 0.8, 0.5, 67)
    assert accept_walker(1.00000001, 1.0, 0.8, 0.5, 67)
    assert not accept_walker(2.0, 1.0, 0.8, 0.5, 67)


def test_large_kick_changes_about_five_percent_of_signs_without_changing_scales():
    state = {
        "left_binary": torch.ones(100, 100), "right_binary": torch.ones(100, 1000),
        "scale_pre": torch.ones(1000), "scale_mid": torch.ones(100), "scale_post": torch.ones(100),
    }
    candidate, details = perturb({}, state, 291, large=True)
    flips = details["left_flips"] + details["right_flips"]
    assert 0.045 < flips / 110000 < 0.055
    validate_state(candidate, (100, 1000), 100)
    for key in STATE_KEYS[2:]:
        assert torch.equal(candidate[key], state[key])
    assert bool((state["left_binary"] == 1).all())
    assert bool((state["right_binary"] == 1).all())
