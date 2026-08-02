from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tools.select_c4_capability_correction_checkpoint import (
    _arm_steps,
    run,
    select_c4_capability_checkpoint,
)


def _payloads(
    values: dict[str, tuple[int, float, float]],
) -> tuple[dict[str, object], dict[str, object]]:
    arms = list(values)
    protocol = {
        "arms": [
            {
                "name": name,
                "mode": "postkd" if index == 0 else "checkpoint",
                "expected_steps": steps,
            }
            for index, (name, (steps, _nll, _kl)) in enumerate(values.items())
        ],
        "slice": "fixed",
    }
    sequence_payload = {
        name: [
            {
                "negative_log_likelihood": nll + offset,
                "kl_nats_per_token": kl + offset,
                "token_count": 511,
            }
            for offset in (0.0, 0.1, -0.1, 0.0)
        ]
        for name, (_steps, nll, kl) in values.items()
    }
    quality = {
        "schema_version": 1,
        "status": "completed",
        "protocol": protocol,
        "arms": {
            name: {
                "mode": "postkd" if index == 0 else "checkpoint",
                "steps_completed": steps,
                "checkpoint": None if index == 0 else {"steps": steps},
            }
            for index, (name, (steps, _nll, _kl)) in enumerate(values.items())
        },
        "results": {
            name: {
                "negative_log_likelihood": nll,
                "kl_nats_per_token": kl,
            }
            for name, (_steps, nll, kl) in values.items()
        },
    }
    checkpoint = {
        "schema_version": 1,
        "status": "completed",
        "protocol": deepcopy(protocol),
        "sequences": sequence_payload,
    }
    assert arms[0] == "base"
    return quality, checkpoint


def _select(
    values: dict[str, tuple[int, float, float]],
    *,
    tolerance: float = 0.01,
) -> dict[str, object]:
    quality, checkpoint = _payloads(values)
    return select_c4_capability_checkpoint(
        quality,
        checkpoint,
        baseline=("base", 256),
        ordered_arms=tuple(
            (name, fields[0]) for name, fields in list(values.items())[1:]
        ),
        tolerance=tolerance,
        resamples=1_000,
        seed=0,
    )


def test_arm_steps_parser_rejects_nonpositive_values() -> None:
    assert _arm_steps("candidate=96") == ("candidate", 96)
    with pytest.raises(argparse.ArgumentTypeError):
        _arm_steps("candidate=0")


def test_selects_earliest_checkpoint_on_joint_plateau() -> None:
    result = _select(
        {
            "base": (256, 5.0, 2.0),
            "epoch1": (32, 4.95, 1.95),
            "epoch2": (64, 4.905, 1.905),
            "epoch3": (96, 4.90, 1.90),
        }
    )
    assert result["eligible_arms"] == ["epoch1", "epoch2", "epoch3"]
    assert result["plateau_arms"] == ["epoch2", "epoch3"]
    assert result["selected_arm"] == "epoch2"
    assert result["selected_steps"] == 64
    assert result["correction_applied"] is True


def test_falls_back_to_baseline_when_no_arm_improves_both_metrics() -> None:
    result = _select(
        {
            "base": (256, 5.0, 2.0),
            "epoch1": (32, 4.9, 2.1),
            "epoch2": (64, 5.1, 1.9),
        }
    )
    assert result["eligible_arms"] == []
    assert result["selected_arm"] == "base"
    assert result["correction_applied"] is False


def test_falls_back_when_eligible_minima_have_no_joint_plateau() -> None:
    result = _select(
        {
            "base": (256, 5.0, 2.0),
            "epoch1": (32, 4.8, 1.95),
            "epoch2": (64, 4.95, 1.8),
        }
    )
    assert result["eligible_arms"] == ["epoch1", "epoch2"]
    assert result["plateau_arms"] == []
    assert result["selected_arm"] == "base"


def test_rejects_step_and_protocol_mismatches() -> None:
    quality, checkpoint = _payloads(
        {"base": (256, 5.0, 2.0), "epoch1": (32, 4.9, 1.9)}
    )
    quality["arms"]["epoch1"]["steps_completed"] = 31  # type: ignore[index]
    with pytest.raises(ValueError, match="step identity"):
        select_c4_capability_checkpoint(
            quality,
            checkpoint,
            baseline=("base", 256),
            ordered_arms=(("epoch1", 32),),
            tolerance=0.01,
            resamples=100,
            seed=0,
        )
    quality, checkpoint = _payloads(
        {"base": (256, 5.0, 2.0), "epoch1": (32, 4.9, 1.9)}
    )
    checkpoint["protocol"]["slice"] = "changed"  # type: ignore[index]
    with pytest.raises(ValueError, match="protocol is invalid"):
        select_c4_capability_checkpoint(
            quality,
            checkpoint,
            baseline=("base", 256),
            ordered_arms=(("epoch1", 32),),
            tolerance=0.01,
            resamples=100,
            seed=0,
        )


def test_rejects_extra_arm_inventory() -> None:
    quality, checkpoint = _payloads(
        {"base": (256, 5.0, 2.0), "epoch1": (32, 4.9, 1.9)}
    )
    quality["results"]["extra"] = quality["results"]["epoch1"]  # type: ignore[index]
    with pytest.raises(ValueError, match="protocol is invalid"):
        select_c4_capability_checkpoint(
            quality,
            checkpoint,
            baseline=("base", 256),
            ordered_arms=(("epoch1", 32),),
            tolerance=0.01,
            resamples=100,
            seed=0,
        )


def test_rejects_aggregate_metrics_that_differ_from_paired_sequences() -> None:
    quality, checkpoint = _payloads(
        {"base": (256, 5.0, 2.0), "epoch1": (32, 4.9, 1.9)}
    )
    quality["results"]["epoch1"]["negative_log_likelihood"] = 4.8  # type: ignore[index]
    with pytest.raises(ValueError, match="differs from its sequence results"):
        select_c4_capability_checkpoint(
            quality,
            checkpoint,
            baseline=("base", 256),
            ordered_arms=(("epoch1", 32),),
            tolerance=0.01,
            resamples=100,
            seed=0,
        )


def test_run_binds_both_inputs_by_hash(tmp_path: Path) -> None:
    quality, checkpoint = _payloads(
        {"base": (256, 5.0, 2.0), "epoch1": (32, 4.9, 1.9)}
    )
    quality_path = tmp_path / "quality.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    output = tmp_path / "selection.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    assert (
        run(
            argparse.Namespace(
                quality_output=quality_path,
                checkpoint_output=checkpoint_path,
                output=output,
                baseline=("base", 256),
                ordered_arm=[("epoch1", 32)],
                tolerance=0.01,
                resamples=100,
                seed=0,
            )
        )
        == 0
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert len(receipt["protocol"]["quality_sha256"]) == 64
    assert len(receipt["protocol"]["checkpoint_sha256"]) == 64
    assert (
        receipt["protocol"]["quality_sha256"]
        != receipt["protocol"]["checkpoint_sha256"]
    )
    assert receipt["selected_arm"] == "epoch1"
