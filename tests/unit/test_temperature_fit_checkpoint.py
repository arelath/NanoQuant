from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoquant.application.temperature_calibration import TemperatureNllStatistics, fit_logit_temperature
from nanoquant.infrastructure.temperature_fit_checkpoint import (
    complete_temperature_fit_receipt,
    load_temperature_fit_iterations,
    load_temperature_fit_receipt,
    write_temperature_fit_progress,
)


def _protocol() -> dict[str, object]:
    return {
        "arm": {"name": "candidate", "artifact_id": "sha256-" + "1" * 64},
        "slice": {"id": "c4-calibration", "token_hash": "sha256:" + "2" * 64},
        "solver": {
            "version": 1,
            "initial_logit_scale": 1.0,
            "minimum_logit_scale": 0.5,
            "maximum_logit_scale": 1.5,
            "maximum_update_passes": 4,
            "convergence_tolerance": 1e-4,
            "hessian_floor": 1e-12,
        },
    }


def _result():
    return fit_logit_temperature(
        lambda scale: TemperatureNllStatistics(8, 3.0, scale - 1.2, 1.0)
    )


def test_temperature_fit_progress_is_protocol_bound(tmp_path: Path) -> None:
    path = tmp_path / "fit.checkpoint.json"
    result = _result()
    write_temperature_fit_progress(path, _protocol(), result.iterations[:1])

    assert load_temperature_fit_iterations(path, _protocol()) == result.iterations[:1]
    changed = {**_protocol(), "slice": {"id": "different"}}
    with pytest.raises(ValueError, match="protocol identity"):
        load_temperature_fit_iterations(path, changed)


def test_temperature_fit_receipt_is_immutable_and_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "fit.json"
    checkpoint = tmp_path / "fit.checkpoint.json"
    result = _result()

    first = complete_temperature_fit_receipt(output, checkpoint, _protocol(), result)
    second = complete_temperature_fit_receipt(output, checkpoint, _protocol(), result)
    protocol, restored = load_temperature_fit_receipt(output)

    assert first == second
    assert protocol == _protocol()
    assert restored == result
    assert json.loads(checkpoint.read_text()) == first

    altered = _result()
    changed = {**_protocol(), "slice": {"id": "different"}}
    with pytest.raises(ValueError, match="protocol identity"):
        complete_temperature_fit_receipt(output, checkpoint, changed, altered)


def test_temperature_fit_receipt_rejects_solver_mismatch(tmp_path: Path) -> None:
    protocol = _protocol()
    protocol["solver"] = {**protocol["solver"], "maximum_update_passes": 8}

    with pytest.raises(ValueError, match="solver protocol"):
        complete_temperature_fit_receipt(
            tmp_path / "fit.json",
            tmp_path / "fit.checkpoint.json",
            protocol,
            _result(),
        )
