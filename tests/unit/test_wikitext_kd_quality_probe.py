from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.probe_wikitext_kd_quality import (
    SequenceMetrics,
    _aggregate,
    _bootstrap_interval,
    _paired_interval,
    _parse_arm,
    _slice_reservation,
)


def test_checkpoint_arm_parser_keeps_windows_paths() -> None:
    assert _parse_arm(r"tail=checkpoint;D:\frozen;D:\checkpoints;8") == (
        "tail",
        "checkpoint",
        Path(r"D:\frozen"),
        Path(r"D:\checkpoints"),
        8,
        "global-distillation",
    )


def test_materialized_arm_parser_remains_compatible() -> None:
    assert _parse_arm(r"tail=D:\materialized") == (
        "tail",
        "postkd",
        Path(r"D:\materialized"),
        None,
        None,
        "global-distillation",
    )


def test_pre_kd_arm_parser_supports_same_factorization_control() -> None:
    assert _parse_arm(r"baseline=prekd;D:\materialized") == (
        "baseline",
        "prekd",
        Path(r"D:\materialized"),
        None,
        None,
        "global-distillation",
    )


def test_tuning_arm_parser_keeps_immutable_pointer() -> None:
    assert _parse_arm(r"long=tuning;D:\frozen;D:\conditional.json") == (
        "long",
        "tuning",
        Path(r"D:\frozen"),
        Path(r"D:\conditional.json"),
        None,
        "global-distillation",
    )


def test_checkpoint_arm_parser_accepts_canonical_correction_namespace() -> None:
    assert _parse_arm(
        r"tail=checkpoint;D:\frozen;D:\checkpoints;3;global-distillation-mass-floor"
    ) == (
        "tail",
        "checkpoint",
        Path(r"D:\frozen"),
        Path(r"D:\checkpoints"),
        3,
        "global-distillation-mass-floor",
    )


def _sequence(nll: float, kl: float, tail_kl: float, tokens: int) -> SequenceMetrics:
    return SequenceMetrics(nll, kl, 0.1, tail_kl, 0.9, 0.8, 0.1, tokens)


def test_aggregate_uses_token_weighted_means() -> None:
    result = _aggregate((_sequence(2.0, 1.0, 0.8, 1), _sequence(4.0, 2.0, 1.2, 3)))

    assert result["negative_log_likelihood"] == pytest.approx(3.5)
    assert result["full_kl"] == pytest.approx(1.75)
    assert result["topk_plus_tail_kl"] == pytest.approx(1.1)
    assert result["token_count"] == 4


def test_paired_interval_detects_uniform_improvement() -> None:
    baseline = tuple(_sequence(3.0 + index, 2.0, 1.0, 10) for index in range(4))
    candidate = tuple(_sequence(2.5 + index, 1.5, 0.8, 10) for index in range(4))

    result = _paired_interval(
        baseline,
        candidate,
        "negative_log_likelihood",
        resamples=1_000,
        seed=0,
    )

    assert result["point_delta"] == pytest.approx(-0.5)
    assert result["lower_delta"] == pytest.approx(-0.5)
    assert result["upper_delta"] == pytest.approx(-0.5)
    assert result["improved_with_confidence"] is True


def test_bootstrap_interval_reports_absolute_selected_mass_uncertainty() -> None:
    values = tuple(
        SequenceMetrics(2.0, 1.0, 0.1, 0.8, 0.9, mass, 0.1, 10)
        for mass in (0.76, 0.77, 0.78, 0.79)
    )

    result = _bootstrap_interval(
        values,
        "student_teacher_topk_mass",
        resamples=1_000,
        seed=0,
    )

    assert result["point"] == pytest.approx(0.775)
    assert float(result["lower"]) < 0.775 < float(result["upper"])


def test_slice_reservation_rejects_overlap(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slices": [
                    {
                        "id": "retired",
                        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
                        "split": "validation",
                        "offset": 0,
                        "samples": 48,
                        "sequence_length": 512,
                        "token_hash": "old",
                        "status": "retired",
                    },
                    {
                        "id": "candidate",
                        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
                        "split": "validation",
                        "offset": 24,
                        "samples": 48,
                        "sequence_length": 512,
                        "token_hash": "new",
                        "status": "reserved",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps"):
        _slice_reservation(
            registry,
            "candidate",
            split="validation",
            offset=24,
            samples=48,
            sequence_length=512,
            token_hash="new",
        )
