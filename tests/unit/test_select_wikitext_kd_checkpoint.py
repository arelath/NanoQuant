from __future__ import annotations

from dataclasses import asdict

from tools.probe_wikitext_kd_quality import SequenceMetrics
from tools.select_wikitext_kd_checkpoint import select_checkpoint


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


def test_earliest_plateau_does_not_replace_better_incumbent() -> None:
    quality, checkpoint = _payload(
        {"pre": (5.0, 2.0, 1.9), "early": (4.1, 1.1, 1.0), "late": (4.0, 1.0, 0.9)}
    )

    result = select_checkpoint(
        quality,
        checkpoint,
        baseline="pre",
        incumbent="late",
        ordered_arms=("early", "late"),
        tolerance=0.11,
        resamples=100,
        seed=0,
    )

    assert result["selected_arm"] == "early"
    assert result["replace_incumbent"] is False
    assert result["decision"] == "retain late"


def test_earlier_checkpoint_replaces_worse_incumbent() -> None:
    quality, checkpoint = _payload(
        {"pre": (5.0, 2.0, 1.9), "early": (3.9, 1.0, 0.9), "late": (4.1, 1.0, 0.9)}
    )

    result = select_checkpoint(
        quality,
        checkpoint,
        baseline="pre",
        incumbent="late",
        ordered_arms=("early", "late"),
        tolerance=0.02,
        resamples=100,
        seed=0,
    )

    assert result["selected_arm"] == "early"
    assert result["replace_incumbent"] is True
    assert result["decision"] == "replace late with early"
