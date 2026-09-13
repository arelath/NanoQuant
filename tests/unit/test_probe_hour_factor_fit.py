"""CPU guards for the timed research runner's score, budget, and saved fit."""

import time

import pytest
import torch
from probe_hour_factor_fit import (
    BudgetExpired,
    DeadlineRecorder,
    budgeted_iterations,
    choose_mode,
    save_state,
    score,
    validate_state,
)
from safetensors.torch import load_file


def _case():
    state = {
        "left_binary": torch.tensor([[1.0, -1.0], [-1.0, -1.0]]),
        "right_binary": torch.tensor([[1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
        "scale_pre": torch.tensor([0.5, 2.0, 3.0]),
        "scale_mid": torch.tensor([0.7, -0.2]),
        "scale_post": torch.tensor([1.2, 0.4]),
    }
    fixture = {
        "target": torch.tensor([[0.2, -0.5, 0.8], [0.1, 0.0, -0.2]]),
        "input_importance": torch.tensor([1.0, 2.0, 3.0]),
        "output_importance": torch.tensor([0.25, 2.0]),
    }
    return fixture, state


def test_score_matches_independent_elementwise_sum():
    fixture, state = _case()
    actual = score(fixture, state, double=True)
    error = 0.0
    for i in range(2):
        for j in range(3):
            prediction = (
                float(state["scale_post"][i])
                * float(state["scale_pre"][j])
                * sum(
                    float(state["left_binary"][i, k])
                    * float(state["scale_mid"][k])
                    * float(state["right_binary"][k, j])
                    for k in range(2)
                )
            )
            error += (prediction - float(fixture["target"][i, j])) ** 2 * float(
                fixture["input_importance"][j] * fixture["output_importance"][i]
            )
    assert actual["weighted_error"] == pytest.approx(error, rel=1e-12)


def test_checkpoint_replays_exactly_and_rejects_nonbinary(tmp_path):
    fixture, state = _case()
    validate_state(state, (2, 3), 2)
    path = tmp_path / "best.safetensors"
    assert len(save_state(path, state)) == 64
    restored = load_file(str(path))
    assert score(fixture, restored, double=True) == score(fixture, state, double=True)
    restored["left_binary"][0, 0] = 0
    with pytest.raises(ValueError, match="nonbinary"):
        validate_state(restored, (2, 3), 2)


def test_deadline_checked_before_next_admm_iteration():
    recorder = DeadlineRecorder(time.perf_counter() - 1)
    with pytest.raises(BudgetExpired):
        recorder.phase("iteration")


def test_fastest_pilot_must_pass_common_precision_error_guard():
    pilots = [
        {"mode": "fp32-ieee", "weighted_error": 1.0, "admm_seconds": 3.0},
        {"mode": "fast-bad", "weighted_error": 1.06, "admm_seconds": 0.1},
        {"mode": "fast-valid", "weighted_error": 1.02, "admm_seconds": 0.5},
    ]
    assert choose_mode(pilots) == "fast-valid"


def test_budget_reduces_horizon_and_reserves_finish_time():
    assert budgeted_iterations(12800, 2000, 0.1) == 12800
    assert budgeted_iterations(12800, 150, 0.1) == 800
    assert budgeted_iterations(12800, 10, 0.1) == 0
