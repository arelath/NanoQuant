"""Prove byte-identical pinned-Gemma calibration and allocation in two runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
from recipes import ExperimentDefinition

from nanoquant.config.codec import config_hash
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    execute_resident_workflow,
    resolve_resident_experiment_inputs,
)

_COMPARE_MODULE = "tools.compare_preprocessing_runs" if __package__ else "compare_preprocessing_runs"
compare_runs = cast(Any, importlib.import_module(_COMPARE_MODULE)).compare_runs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-wddm-shared-gib", type=float, default=0.75)
    parser.add_argument("--single-run-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _load_definition(path: Path) -> ExperimentDefinition[Any]:
    namespace = runpy.run_path(str(path.resolve()), run_name="nanoquant_preprocessing_replay")
    definition = namespace.get("EXPERIMENT")
    if not isinstance(definition, ExperimentDefinition):
        raise TypeError(f"launcher does not expose an ExperimentDefinition named EXPERIMENT: {path}")
    return definition


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validated_preprocessing_receipt(output: Path) -> dict[str, object]:
    state_path = output / "state" / "preprocessing.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    artifacts = LocalArtifactStore(
        output / "artifacts",
        use_persistent_validation_cache=False,
    )
    validated: dict[str, dict[str, object]] = {}
    for name in ("calibration", "objectives", "plan"):
        reference = state[name]
        artifact_id = str(reference["artifact_id"])
        descriptor = artifacts.validate(artifact_id)
        validated[name] = {
            "artifact_id": artifact_id,
            "artifact_type": descriptor.artifact_type,
            "file_count": len(descriptor.files),
            "bytes": sum(item.bytes for item in descriptor.files),
        }

    plan_id = str(state["plan"]["artifact_id"])
    plan_path = artifacts.path_for(plan_id) / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    profile_reference = plan.get("reconstruction_profile")
    if not isinstance(profile_reference, dict):
        raise ValueError("preprocessing plan has no reconstruction-rank profile")
    profile_id = str(profile_reference["artifact_id"])
    profile_descriptor = artifacts.validate(profile_id)
    profile_path = artifacts.path_for(profile_id) / "reconstruction-rank-profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    unit_results = profile.get("unit_results")
    if not isinstance(unit_results, list):
        raise ValueError("reconstruction-rank profile has no unit result inventory")
    for index, reference in enumerate(unit_results):
        if not isinstance(reference, dict) or "artifact_id" not in reference:
            raise ValueError(f"rank-profile unit result {index} is invalid")
        artifacts.validate(str(reference["artifact_id"]))

    calibration_input_path = output / "calibration-input.json"
    calibration_input = json.loads(calibration_input_path.read_text(encoding="utf-8"))
    return {
        "output": str(output.resolve()),
        "preprocessing_state_sha256": _sha256(state_path),
        "resident_config_hash": state["resident_config_hash"],
        "calibration_input": {
            "receipt_sha256": _sha256(calibration_input_path),
            "artifact_id": calibration_input.get("artifact_id"),
            "fingerprint": calibration_input.get("fingerprint"),
            "sample_count": calibration_input.get("sample_count"),
            "sequence_length": calibration_input.get("sequence_length"),
        },
        "artifacts": validated,
        "reconstruction_profile": {
            "artifact_id": profile_id,
            "artifact_type": profile_descriptor.artifact_type,
            "unit_count": len(unit_results),
        },
    }


def _run_one(
    definition: ExperimentDefinition[Any],
    launcher: Path,
    output: Path,
    maximum_shared_bytes: int,
) -> dict[str, object]:
    inputs = resolve_resident_experiment_inputs(
        definition.config,
        launcher_path=launcher,
        output_override=output,
    )
    try:
        execute_resident_workflow(
            definition.config,
            inputs,
            ResidentExecutionOptions(
                interrupt_after_preprocessing=True,
                maximum_wddm_shared_bytes=maximum_shared_bytes,
            ),
        )
    except InterruptedError as exc:
        if "after durable preprocessing commit" not in str(exc):
            raise
    else:
        raise RuntimeError("preprocessing replay unexpectedly continued into compression")
    return _validated_preprocessing_receipt(output)


def _run_in_independent_process(
    launcher: Path,
    output_root: Path,
    output: Path,
    maximum_wddm_shared_gib: float,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--launcher",
            str(launcher),
            "--output-root",
            str(output_root),
            "--maximum-wddm-shared-gib",
            str(maximum_wddm_shared_gib),
            "--single-run-output",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.maximum_wddm_shared_gib < 0:
        raise ValueError("maximum WDDM shared memory must be non-negative")
    launcher = args.launcher.resolve()
    output_root = args.output_root.resolve()
    definition = _load_definition(launcher)
    maximum_shared_bytes = int(args.maximum_wddm_shared_gib * 2**30)
    if args.single_run_output is not None:
        receipt = _run_one(
            definition,
            launcher,
            args.single_run_output.resolve(),
            maximum_shared_bytes,
        )
        print(json.dumps(receipt, sort_keys=True, indent=2))
        return 0

    receipt_path = output_root / "preprocessing-reproducibility.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable replay receipt: {receipt_path}")
    run_paths = [output_root / label for label in ("run-a", "run-b")]
    for output in run_paths:
        _run_in_independent_process(
            launcher,
            output_root,
            output,
            args.maximum_wddm_shared_gib,
        )
    runs = [_validated_preprocessing_receipt(output) for output in run_paths]
    compared_fields = (
        ("resident_config_hash",),
        ("calibration_input", "artifact_id"),
        ("calibration_input", "fingerprint"),
        ("calibration_input", "sample_count"),
        ("calibration_input", "sequence_length"),
        ("artifacts", "calibration", "artifact_id"),
        ("artifacts", "objectives", "artifact_id"),
        ("artifacts", "plan", "artifact_id"),
        ("reconstruction_profile", "artifact_id"),
        ("reconstruction_profile", "unit_count"),
    )

    def value(run: dict[str, object], path: tuple[str, ...]) -> object:
        current: object = run
        for part in path:
            current = cast(dict[str, object], current)[part]
        return current

    comparisons = {
        ".".join(path): {
            "equal": value(runs[0], path) == value(runs[1], path),
            "run_a": value(runs[0], path),
            "run_b": value(runs[1], path),
        }
        for path in compared_fields
    }
    passed = all(bool(item["equal"]) for item in comparisons.values())
    artifact_comparison = compare_runs(*run_paths)
    passed = passed and bool(artifact_comparison["passed"])
    receipt = {
        "schema_version": 1,
        "producer": {
            "name": "replay-gemma-preprocessing-reproducibility",
            "version": "1",
        },
        "launcher": str(launcher),
        "experiment": definition.identity.canonical_name,
        "canonical_config_hash": config_hash(definition.config),
        "maximum_wddm_shared_bytes": maximum_shared_bytes,
        "runs": runs,
        "comparisons": comparisons,
        "exact_artifact_comparison": artifact_comparison,
        "passed": passed,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
