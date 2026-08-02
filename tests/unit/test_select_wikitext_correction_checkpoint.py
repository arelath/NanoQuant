from __future__ import annotations

from dataclasses import asdict

from tools.probe_wikitext_kd_quality import SequenceMetrics
from tools.select_wikitext_correction_checkpoint import select_correction_checkpoint


def _sequence(nll: float, full_kl: float, tail_kl: float) -> dict[str, object]:
    return asdict(SequenceMetrics(nll, full_kl, 0.1, tail_kl, 0.9, 0.8, 0.1, 10))


def _payload(values: dict[str, tuple[float, float, float]]) -> tuple[dict, dict]:
    protocol = {"slice": "frozen"}
    quality = {
        "status": "completed",
        "protocol": protocol,
        "results": {
            name: {
                "means": {
                    "negative_log_likelihood": metrics[0],
                    "full_kl": metrics[1],
                    "topk_plus_tail_kl": metrics[2],
                }
            }
            for name, metrics in values.items()
        },
    }
    checkpoint = {
        "status": "completed",
        "protocol": protocol,
        "sequences": {
            name: [_sequence(*metrics) for _ in range(4)]
            for name, metrics in values.items()
        },
    }
    return quality, checkpoint


def test_tail_regression_makes_checkpoint_ineligible() -> None:
    quality, checkpoint = _payload(
        {"base": (5.0, 2.0, 1.0), "arm": (4.0, 1.0, 1.01)}
    )

    result = select_correction_checkpoint(
        quality,
        checkpoint,
        baseline="base",
        ordered_arms=("arm",),
        tolerance=0.02,
        resamples=100,
        seed=0,
    )

    assert result["eligible_arms"] == []
    assert result["selected_arm"] is None
    assert result["decision"] == "no survivor"


def test_selects_earliest_three_metric_plateau_member() -> None:
    quality, checkpoint = _payload(
        {
            "base": (5.0, 2.0, 1.9),
            "early": (4.01, 1.01, 0.91),
            "late": (4.0, 1.0, 0.9),
        }
    )

    result = select_correction_checkpoint(
        quality,
        checkpoint,
        baseline="base",
        ordered_arms=("early", "late"),
        tolerance=0.02,
        resamples=100,
        seed=0,
    )

    assert result["eligible_arms"] == ["early", "late"]
    assert result["plateau_arms"] == ["early", "late"]
    assert result["selected_arm"] == "early"
    assert result["decision"] == "select early"
