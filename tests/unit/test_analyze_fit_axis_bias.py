"""Independent least-squares checks for the additive-bias diagnostic."""

import numpy as np
import pytest
from analyze_fit_axis_bias import fit_biases


def test_closed_form_matches_full_weighted_least_squares_design():
    residual = np.random.default_rng(72).normal(size=(4, 7))
    u = np.array([0.1, 3.0, 0.5, 2.0])
    v = np.array([2.0, 1.0, 0.3, 0.1, 4.0, 2.0, 0.7])
    row, col, grand = fit_biases(residual, u, v)
    correction = (row-grand)[:, None] + col[None, :]
    design = np.zeros((28, 11))
    for i in range(4):
        for j in range(7):
            design[i*7+j, i] = 1
            design[i*7+j, 4+j] = 1
    weights = np.sqrt(u[:, None] * v[None, :]).ravel()
    solution = np.linalg.lstsq(design * weights[:, None], residual.ravel() * weights, rcond=None)[0]
    np.testing.assert_allclose(correction, (design @ solution).reshape(4, 7), atol=1e-14)
    corrected = residual-correction
    np.testing.assert_allclose(corrected @ v, np.zeros(4), atol=1e-14)
    np.testing.assert_allclose(u @ corrected, np.zeros(7), atol=1e-14)


def test_pure_row_plus_column_residual_is_removed_without_double_counting_mean():
    residual = np.array([2.0, -3.0, 1.0])[:, None] + np.array([0.3, 5.0, -1.0, 2.0])[None, :]
    row, column, grand = fit_biases(residual, np.array([1.0, 2.0, 4.0]), np.array([5.0, 0.2, 3.0, 1.0]))
    np.testing.assert_allclose((row-grand)[:, None] + column[None, :], residual, atol=1e-14)


def test_projection_energy_matches_error_reduction():
    residual = np.random.default_rng(2).normal(size=(3, 5))
    u, v = np.array([0.5, 2.0, 4.0]), np.array([1.0, 0.2, 0.4, 3.0, 2.0])
    row, col, grand = fit_biases(residual, u, v)
    correction = (row-grand)[:, None] + col[None, :]
    weights = u[:, None] * v[None, :]
    improvement = np.sum(weights * (residual**2 - (residual-correction)**2))
    assert improvement == pytest.approx(np.sum(weights * correction**2), abs=1e-13)
    expected = v.sum() * np.dot(u, row**2) + u.sum() * np.dot(v, col**2) - u.sum()*v.sum()*grand**2
    assert improvement == pytest.approx(expected, abs=1e-13)


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        fit_biases(np.ones((2, 3)), np.array([0.0, 1.0]), np.ones(3))
    with pytest.raises(ValueError, match="dimensions"):
        fit_biases(np.ones((2, 3)), np.ones(3), np.ones(3))
