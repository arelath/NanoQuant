"""Identity-bound checkpoints and immutable receipts for scalar temperature fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from nanoquant.application.temperature_calibration import (
    TEMPERATURE_CALIBRATION_VERSION,
    TemperatureFitIteration,
    TemperatureFitResult,
)
from nanoquant.config.codec import from_dict, semantic_hash, to_dict
from nanoquant.infrastructure.io_utils import atomic_write_json

TEMPERATURE_FIT_RECEIPT_SCHEMA_VERSION = 1


def temperature_fit_protocol_hash(protocol: dict[str, object]) -> str:
    return semantic_hash(protocol)


def _validate_protocol_result(
    protocol: dict[str, object],
    result: TemperatureFitResult,
) -> None:
    solver = protocol.get("solver")
    expected = {
        "version": result.version,
        "initial_logit_scale": result.initial_logit_scale,
        "minimum_logit_scale": result.minimum_logit_scale,
        "maximum_logit_scale": result.maximum_logit_scale,
        "maximum_update_passes": result.maximum_update_passes,
        "convergence_tolerance": result.convergence_tolerance,
        "hessian_floor": result.hessian_floor,
    }
    if (
        not isinstance(solver, dict)
        or any(solver.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("temperature-fit result differs from its solver protocol")


def _validated_payload(path: Path, protocol: dict[str, object] | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != TEMPERATURE_FIT_RECEIPT_SCHEMA_VERSION
        or payload.get("status") not in {"in_progress", "completed"}
        or not isinstance(payload.get("protocol"), dict)
        or payload.get("protocol_hash")
        != temperature_fit_protocol_hash(cast(dict[str, object], payload["protocol"]))
        or not isinstance(payload.get("iterations"), list)
    ):
        raise ValueError("temperature-fit checkpoint or receipt is invalid")
    if protocol is not None and (
        payload["protocol"] != protocol
        or payload["protocol_hash"] != temperature_fit_protocol_hash(protocol)
    ):
        raise ValueError("temperature-fit protocol identity differs")
    return cast(dict[str, Any], payload)


def load_temperature_fit_iterations(
    path: Path,
    protocol: dict[str, object],
) -> tuple[TemperatureFitIteration, ...]:
    if not path.is_file():
        return ()
    payload = _validated_payload(path, protocol)
    return tuple(
        from_dict(TemperatureFitIteration, item, path=f"temperature_fit.iterations[{index}]")
        for index, item in enumerate(payload["iterations"])
    )


def write_temperature_fit_progress(
    path: Path,
    protocol: dict[str, object],
    iterations: tuple[TemperatureFitIteration, ...],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": TEMPERATURE_FIT_RECEIPT_SCHEMA_VERSION,
            "status": "in_progress",
            "protocol": protocol,
            "protocol_hash": temperature_fit_protocol_hash(protocol),
            "iterations": to_dict(iterations),
            "result": None,
        },
    )


def complete_temperature_fit_receipt(
    output: Path,
    checkpoint: Path,
    protocol: dict[str, object],
    result: TemperatureFitResult,
) -> dict[str, Any]:
    if result.version != TEMPERATURE_CALIBRATION_VERSION or not result.converged:
        raise ValueError("temperature-fit result is not a completed current-version fit")
    _validate_protocol_result(protocol, result)
    payload: dict[str, Any] = {
        "schema_version": TEMPERATURE_FIT_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "protocol": protocol,
        "protocol_hash": temperature_fit_protocol_hash(protocol),
        "iterations": to_dict(result.iterations),
        "result": to_dict(result),
        "calibration_metrics": {
            "raw_negative_log_likelihood": result.iterations[0].mean_negative_log_likelihood,
            "fitted_negative_log_likelihood": result.final_mean_negative_log_likelihood,
        },
    }
    if output.exists():
        existing = _validated_payload(output, protocol)
        if existing != payload:
            raise FileExistsError(f"refusing to replace a different temperature-fit receipt: {output}")
        return existing
    atomic_write_json(output, payload)
    atomic_write_json(checkpoint, payload)
    return payload


def load_temperature_fit_receipt(path: Path) -> tuple[dict[str, object], TemperatureFitResult]:
    payload = _validated_payload(path)
    if payload["status"] != "completed" or not isinstance(payload.get("result"), dict):
        raise ValueError("temperature-fit receipt is incomplete")
    result = from_dict(TemperatureFitResult, payload["result"], path="temperature_fit.result")
    if result.version != TEMPERATURE_CALIBRATION_VERSION or not result.converged:
        raise ValueError("temperature-fit receipt result is invalid")
    _validate_protocol_result(cast(dict[str, object], payload["protocol"]), result)
    if tuple(payload["iterations"]) != tuple(to_dict(result.iterations)):
        raise ValueError("temperature-fit receipt iteration inventory differs from its result")
    return cast(dict[str, object], payload["protocol"]), result


__all__ = [
    "TEMPERATURE_FIT_RECEIPT_SCHEMA_VERSION",
    "complete_temperature_fit_receipt",
    "load_temperature_fit_iterations",
    "load_temperature_fit_receipt",
    "temperature_fit_protocol_hash",
    "write_temperature_fit_progress",
]
