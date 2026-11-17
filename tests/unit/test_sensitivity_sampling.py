import pytest
from summarize_sensitivity_sampling import nested_curves


def test_nested_curve_uses_the_same_selected_windows_for_the_clean_control():
    report = {
        "clean_baseline": {"sequences": [{"kl_nats_per_token": 0.2}, {"kl_nats_per_token": 0.8}]},
        "trials": [
            {
                "target": "0:mlp.up_proj", "requested_error": error, "actual_error": error,
                "sequences": [
                    {"kl_nats_per_token": 0.2 + 3 * error**2},
                    {"kl_nats_per_token": 0.8 + 9 * error**2},
                ],
            }
            for error in (0.01, 0.025, 0.05, 0.10, 0.20, 0.35)
        ],
    }
    subset = nested_curves(report, [1])["0:mlp.up_proj"]
    assert subset["coefficient"] == pytest.approx(9)
    assert subset["exponent"] == pytest.approx(2)
    combined = nested_curves(report, [0, 1])["0:mlp.up_proj"]
    assert combined["coefficient"] == pytest.approx(6)
    assert combined["exponent"] == pytest.approx(2)
