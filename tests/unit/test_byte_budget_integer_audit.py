import itertools

import pytest
from audit_byte_budget_allocation import integer_allocate
from probe_byte_budget_allocation import Option, allocate

pytest.importorskip("scipy")


def test_retains_discrete_option_above_convex_hull():
    units = {"a": [Option("low", 1, 10), Option("middle", 6, 6), Option("high", 11, 0)]}
    assert allocate(units, 6)["a"].label == "low"
    result = integer_allocate(units, 6)
    assert result["choices"] == {"a": "middle"}
    assert result["kl"] == pytest.approx(6)
    assert result["relative_gap"] == pytest.approx(0)


def test_matches_brute_force_for_indivisible_upgrades():
    units = {
        "a": [Option("low", 1, 10), Option("high", 7, 3)],
        "b": [Option("low", 1, 10), Option("high", 6, 4.5)],
        "c": [Option("low", 1, 10), Option("high", 6, 4.5)],
    }
    budget = 13
    expected = min(sum(option.kl for option in choices)
                   for choices in itertools.product(*units.values())
                   if sum(option.bytes for option in choices) <= budget)
    result = integer_allocate(units, budget)
    assert result["kl"] == pytest.approx(expected)
    assert result["projection_bytes"] <= budget
    assert result["kl"] < sum(option.kl for option in allocate(units, budget).values())
