"""Bind a transferred correction policy to its actual development initializer regime.

This preflight is deliberately separate from checkpoint replay. Replay proves
that an implementation can reproduce a retained trajectory; this receipt proves
that the retained trajectory used to develop a correction has the same primary
KD protocol and observed horizon as the deployment launcher.
"""

from __future__ import annotations

import argparse
import json
import runpy
from dataclasses import dataclass
from pathlib import Path

import _paths  # noqa: F401
from recipes import ExperimentDefinition

from nanoquant.compression_quality_workflow import CompressionQualityExperiment
from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.domain.models import GlobalTuningResult
from nanoquant.global_distillation import distillation_protocol_hash
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.resident_workflow import primary_distillation_config_from_run_config


@dataclass(frozen=True, slots=True)
class PolicyInitializerRegime:
    protocol_hash: str
    observed_steps: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--deployment-launcher", type=Path, required=True)
    parser.add_argument("--selection-slice-id", required=True)
    parser.add_argument("--confirmation-slice-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    return "sha256:" + hash_file(path)


def _load_config(launcher: Path) -> RunConfig:
    namespace = runpy.run_path(str(launcher), run_name="nanoquant_policy_transfer_preflight")
    definition = namespace.get("EXPERIMENT")
    if not isinstance(definition, ExperimentDefinition) or not isinstance(
        definition.workflow, CompressionQualityExperiment
    ):
        raise TypeError("deployment launcher does not expose a compression-quality experiment")
    return definition.config


def validate_policy_initializer_transfer(
    development: PolicyInitializerRegime,
    deployment: RunConfig,
    *,
    selection_slice_id: str,
    confirmation_slice_id: str,
) -> dict[str, object]:
    """Fail closed unless a transferred correction is developed in the deployment regime."""

    distillation = deployment.distillation
    correction = distillation.mass_floor_correction
    if not distillation.enabled or not correction.enabled:
        raise ValueError("policy transfer requires primary KD and correction to be enabled")
    if distillation.maximum_batches_per_epoch is None:
        raise ValueError("transferred policy requires an explicit primary batches-per-epoch cap")
    configured_steps = distillation.epochs * distillation.maximum_batches_per_epoch
    deployment_hash = distillation_protocol_hash(
        primary_distillation_config_from_run_config(deployment)
    )
    if (
        development.protocol_hash != deployment_hash
        or development.observed_steps != configured_steps
    ):
        raise ValueError(
            "development initializer regime differs from deployment primary KD; "
            "treat the regime change as a new experiment and re-derive the policy"
        )
    if (
        correction.expected_initializer_protocol_hash != development.protocol_hash
        or correction.expected_initializer_steps != development.observed_steps
    ):
        raise ValueError("correction initializer declaration differs from development evidence")
    if distillation.final_norm_calibration.enabled:
        raise ValueError(
            "a fixed final-norm calibration value may not transfer across trajectories; "
            "fit calibration per arm on calibration-only data"
        )
    if (
        not selection_slice_id
        or not confirmation_slice_id
        or selection_slice_id == confirmation_slice_id
    ):
        raise ValueError("policy selection and final confirmation require distinct slice identities")
    return {
        "development_initializer": {
            "protocol_hash": development.protocol_hash,
            "observed_steps": development.observed_steps,
        },
        "deployment_primary": {
            "protocol_hash": deployment_hash,
            "configured_epochs": distillation.epochs,
            "configured_maximum_batches_per_epoch": distillation.maximum_batches_per_epoch,
            "expected_steps": configured_steps,
        },
        "correction_initializer_expectation": {
            "protocol_hash": correction.expected_initializer_protocol_hash,
            "steps": correction.expected_initializer_steps,
        },
        "calibration_policy": "per-arm-held-out-fit-only",
        "selection_policy": "same-run-adaptive-checkpoint-selection",
        "selection_slice_id": selection_slice_id,
        "confirmation_slice_id": confirmation_slice_id,
    }


def load_development_initializer(
    run_output: Path,
) -> tuple[GlobalTuningResult, dict[str, object]]:
    manifest_path = run_output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("development run manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = active_global_tuning(run_output)
    if manifest.get("status") != "completed" or reference is None:
        raise ValueError("development initializer run is incomplete")
    if reference.artifact_id not in manifest.get("artifacts", []):
        raise ValueError("development initializer is not authorized by its completed manifest")
    artifacts = LocalArtifactStore(
        run_output / "artifacts", use_persistent_validation_cache=False
    )
    result = load_global_tuning(reference, artifacts).result
    return result, {
        "run_output": str(run_output),
        "manifest_sha256": _sha256(manifest_path),
        "global_tuning": to_dict(reference),
    }


def run(args: argparse.Namespace) -> int:
    development_run = args.development_run.resolve()
    launcher = args.deployment_launcher.resolve()
    result, development_binding = load_development_initializer(development_run)
    config = _load_config(launcher)
    validation = validate_policy_initializer_transfer(
        PolicyInitializerRegime(result.protocol_hash, result.steps_completed),
        config,
        selection_slice_id=args.selection_slice_id,
        confirmation_slice_id=args.confirmation_slice_id,
    )
    atomic_write_json(
        args.output.resolve(),
        {
            "schema_version": 1,
            "status": "validated",
            "role": "correction-policy initializer-regime transfer preflight",
            "claim_scope": (
                "This proves regime identity only. Same-run held-out selection and untouched "
                "confirmation remain necessary policy evidence."
            ),
            "development_evidence": development_binding,
            "deployment_evidence": {
                "launcher": str(launcher),
                "launcher_sha256": _sha256(launcher),
                "config_hash": config_hash(config),
            },
            "validation": validation,
        },
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
