"""Derive a zero-copy global-tuning run with a scale folded into Gemma's final RMSNorm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import _paths  # noqa: F401
from materialize_topk_tail_checkpoint import _hardlink_tree
from probe_distillation_checkpoint_tail_mass import _apply_gemma_final_norm_scale

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.global_tuning import (
    activate_global_tuning,
    active_global_tuning,
    commit_global_tuning,
    load_global_tuning,
)
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json
from nanoquant.infrastructure.tensor_store import LocalTensorStore

CALIBRATION_VERSION = "gemma-final-rmsnorm-effective-weight-scale-v1"
FINAL_NORM_NAME = "model.norm.weight"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--derived-run-output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    return parser


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


def run(args: argparse.Namespace) -> int:
    if not math.isfinite(args.scale) or args.scale <= 0.0:
        raise ValueError("folded final RMSNorm scale must be positive and finite")
    source = args.run_output.resolve()
    destination = args.derived_run_output.resolve()
    if destination.exists():
        raise ValueError(f"derived run output already exists: {destination}")
    active = active_global_tuning(source)
    if active is None:
        raise ValueError("source run has no active global tuning result")
    source_artifacts = LocalArtifactStore(source / "artifacts")
    committed = load_global_tuning(active, source_artifacts)
    auxiliary = dict(committed.result.auxiliary_parameters)
    if FINAL_NORM_NAME not in auxiliary:
        raise ValueError("global tuning result has no Gemma final RMSNorm parameter")
    with LocalTensorStore(source_artifacts).read(auxiliary[FINAL_NORM_NAME]) as value:
        calibrated = value.clone()
        _apply_gemma_final_norm_scale(calibrated, value, args.scale)
    protocol_hash = calibrated_protocol_hash(committed.result.protocol_hash, args.scale)

    with atomic_workspace(destination) as temporary:
        linked_files = _hardlink_tree(source, temporary)
        destination_artifacts = LocalArtifactStore(temporary / "artifacts")
        references = LocalTensorStore(destination_artifacts).put(
            "global-tuning-parameters",
            {FINAL_NORM_NAME: calibrated},
        )
        auxiliary[FINAL_NORM_NAME] = references[FINAL_NORM_NAME]
        result = replace(
            committed.result,
            auxiliary_parameters=tuple(
                (name, auxiliary[name])
                for name, _reference in committed.result.auxiliary_parameters
            ),
            protocol_hash=protocol_hash,
        )
        derived = commit_global_tuning(result, destination_artifacts)
        activate_global_tuning(temporary, derived.reference)
        atomic_write_json(
            temporary / "final-norm-calibration.json",
            {
                "schema_version": 1,
                "version": CALIBRATION_VERSION,
                "source_run_output": str(source),
                "source_global_tuning": to_dict(active),
                "derived_global_tuning": to_dict(derived.reference),
                "parameter": FINAL_NORM_NAME,
                "scale": args.scale,
                "base_protocol_hash": committed.result.protocol_hash,
                "calibrated_protocol_hash": protocol_hash,
                "linked_source_files": linked_files,
            },
        )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
