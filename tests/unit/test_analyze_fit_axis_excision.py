"""Independent checks of axis-only optimum and overlapping union coverage."""

from itertools import combinations

import numpy as np
import pytest
from analyze_fit_axis_excision import coverage_curve, summarize_thresholds


@pytest.mark.parametrize("mode,axis", [("rows", 0), ("columns", 1)])
def test_single_axis_curve_matches_exhaustive_best_subsets(mode, axis):
    energy = np.array([[12., 1., 4., 2.], [0., 8., 3., 1.], [6., 2., 1., 0.]])
    curve = coverage_curve(energy, mode)
    for count in range(1, energy.shape[axis]+1):
        best = max(float(np.take(energy, indices, axis=axis).sum())
                   for indices in combinations(range(energy.shape[axis]), count))
        assert curve["removed"][count] == pytest.approx(best)


def test_mixed_curve_matches_explicit_union_and_maximum_marginal_density():
    energy = np.array([[20., 20., 20., 20.], [1., 1., 19., 1.], [1., 1., 19., 1.]])
    curve = coverage_curve(energy, "mixed")
    m, n = energy.shape
    previous = np.zeros_like(energy, dtype=bool)
    for step, (r, c) in enumerate(curve["counts"]):
        union = np.zeros_like(energy, dtype=bool)
        union[curve["rows"][:r], :] = True
        union[:, curve["columns"][:c]] = True
        assert curve["entries"][step] == union.sum()
        assert curve["removed"][step] == pytest.approx(energy[union].sum())
        if step:
            added = union & ~previous
            chosen_density = energy[added].sum()/added.sum()
            alternatives = []
            for axis, count in ((0, m), (1, n)):
                for index in range(count):
                    candidate = np.zeros_like(union)
                    if axis == 0:
                        candidate[index, :] = True
                    else:
                        candidate[:, index] = True
                    candidate &= ~previous
                    if candidate.any():
                        alternatives.append(energy[candidate].mean())
            assert chosen_density == pytest.approx(max(alternatives))
        previous = union
    assert curve["rows"] and curve["columns"]
    assert curve["entries"][-1] == energy.size


def test_thresholds_use_original_column_ids_and_first_crossing():
    energy = np.array([[9., 4., 2., 1.], [9., 4., 2., 1.]])
    curve = coverage_curve(energy, "columns")
    rows = summarize_thresholds(energy, curve, np.array([0, 2, 5, 7]), 8)
    assert [x["columns"] for x in rows] == [1, 1, 2]
    assert rows[-1]["original_column_indices"] == [0, 2]
    assert rows[-1]["separate_entries"] == 4
    assert rows[-1]["bf16_payload_bytes"] == 8
    assert rows[-1]["active_entry_fraction"] == 0.5
    assert rows[-1]["full_matrix_entry_fraction"] == 0.25


@pytest.mark.parametrize("energy", [np.zeros((2, 3)), np.ones(3), [[-1., 2.]], [[np.nan]], [[np.inf]]])
def test_invalid_energy_rejected(energy):
    with pytest.raises(ValueError, match="energy"):
        coverage_curve(energy, "mixed")
