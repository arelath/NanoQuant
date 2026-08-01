"""Immutable post-distillation calibration of Gemma's final RMSNorm scale."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from nanoquant.config.codec import to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.global_tuning import (
    CommittedGlobalTuning,
    activate_global_tuning,
    activate_global_tuning_stage,
    active_global_tuning_stage,
    commit_global_tuning,
    load_global_tuning,
)
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.tensor_store import LocalTensorStore

CALIBRATION_VERSION = "gemma-final-rmsnorm-effective-weight-scale-v1"
FINAL_NORM_NAME = "model.norm.weight"
STATE_NAMESPACE = "global-distillation-final-norm"


def calibrated_protocol_hash(base_protocol_hash: str, scale: float) -> str:
    payload = json.dumps(
        {
            "base_protocol_hash": base_protocol_hash,
            "calibration": {
                "name": FINAL_NORM_NAME,
                "scale": scale,
                "version": CALIBRATION_VERSION,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def apply_gemma_final_norm_scale(
    parameter: torch.Tensor,
    source: torch.Tensor,
    scale: float,
) -> None:
    if parameter.shape != source.shape:
        raise ValueError("Gemma final RMSNorm source shape differs")
    with torch.no_grad():
        parameter.copy_(((1.0 + source.float()) * scale - 1.0).to(parameter))


def calibrate_global_tuning_final_norm(
    run_output: str | Path,
    source_reference: ArtifactRef,
    scale: float,
) -> CommittedGlobalTuning:
    """Derive and activate a content-addressed final-norm-calibrated tuning result."""

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("folded final RMSNorm scale must be positive and finite")
    output = Path(run_output)
    artifacts = LocalArtifactStore(output / "artifacts")
    source = load_global_tuning(source_reference, artifacts)
    protocol_hash = calibrated_protocol_hash(source.result.protocol_hash, scale)
    completed_reference = active_global_tuning_stage(
        output,
        state_namespace=STATE_NAMESPACE,
    )
    if completed_reference is not None:
        completed = load_global_tuning(completed_reference, artifacts)
        if (
            completed.result.source_blocks != source.result.source_blocks
            or completed.result.token_hash != source.result.token_hash
            or completed.result.protocol_hash != protocol_hash
        ):
            raise ValueError("completed final-norm calibration does not match its requested source")
        activate_global_tuning(output, completed.reference)
        return completed

    auxiliary = dict(source.result.auxiliary_parameters)
    if FINAL_NORM_NAME not in auxiliary:
        raise ValueError("global tuning result has no Gemma final RMSNorm parameter")
    tensors = LocalTensorStore(artifacts)
    with tensors.read(auxiliary[FINAL_NORM_NAME]) as value:
        calibrated = value.clone()
        apply_gemma_final_norm_scale(calibrated, value, scale)
    references = tensors.put(
        "global-tuning-parameters",
        {FINAL_NORM_NAME: calibrated},
    )
    auxiliary[FINAL_NORM_NAME] = references[FINAL_NORM_NAME]
    result = replace(
        source.result,
        auxiliary_parameters=tuple(
            (name, auxiliary[name])
            for name, _reference in source.result.auxiliary_parameters
        ),
        protocol_hash=protocol_hash,
    )
    committed = commit_global_tuning(result, artifacts)
    activate_global_tuning_stage(
        output,
        committed.reference,
        state_namespace=STATE_NAMESPACE,
    )
    activate_global_tuning(output, committed.reference)
    atomic_write_json(
        output / "final-norm-calibration.json",
        {
            "schema_version": 1,
            "version": CALIBRATION_VERSION,
            "source_global_tuning": to_dict(source_reference),
            "derived_global_tuning": to_dict(committed.reference),
            "parameter": FINAL_NORM_NAME,
            "scale": scale,
            "base_protocol_hash": source.result.protocol_hash,
            "calibrated_protocol_hash": protocol_hash,
        },
    )
    return committed


__all__ = [
    "CALIBRATION_VERSION",
    "FINAL_NORM_NAME",
    "STATE_NAMESPACE",
    "apply_gemma_final_norm_scale",
    "calibrate_global_tuning_final_norm",
    "calibrated_protocol_hash",
]
