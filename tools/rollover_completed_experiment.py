"""Archive an incompatible completed or explicitly accepted non-complete experiment.

This is intentionally a dry-run tool by default. If a newly prepared
calibration receipt matching the requested configuration coexists with the old
resident manifest, its validated artifact closure is retained. Otherwise the
fresh run starts without calibration. Failed and interrupted runs require
explicit flags so an operator cannot archive incomplete evidence accidentally.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nanoquant.config.codec import semantic_hash
from nanoquant.infrastructure.artifact_gc import ARTIFACT_ID_PATTERN, TEXT_SUFFIXES
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.io_utils import atomic_write_json


@dataclass(frozen=True, slots=True)
class RolloverPlan:
    run_output: Path
    run_archive: Path
    outputs_root: Path
    outputs_archive: Path
    results_root: Path
    results_archive: Path
    stored_config_hash: str
    expected_config_hash: str
    stored_status: str
    calibration_artifact_id: str | None
    calibration_artifact_count: int
    calibration_logical_bytes: int
    teacher_trace_state_present: bool


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unavailable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _preparation_config_hash(manifest: dict[str, Any]) -> str:
    resolved_config = manifest.get("resolved_config")
    if isinstance(resolved_config, dict) and "canonical_run_config" in resolved_config:
        canonical_run_config = resolved_config["canonical_run_config"]
        if not isinstance(canonical_run_config, dict):
            raise ValueError("resident manifest canonical_run_config must contain a JSON object")
        return semantic_hash(canonical_run_config)
    return str(manifest.get("config_hash") or "")


def _artifact_references(store: LocalArtifactStore, artifact_id: str) -> set[str]:
    descriptor = store.validate(artifact_id)
    root = store.path_for(artifact_id)
    references: set[str] = set()
    for member in descriptor.files:
        path = root / member.path
        if path.suffix.lower() in TEXT_SUFFIXES:
            references.update(
                ARTIFACT_ID_PATTERN.findall(path.read_text(encoding="utf-8", errors="ignore"))
            )
    return references


def _validated_closure(store: LocalArtifactStore, root_artifact: str) -> tuple[str, ...]:
    reachable: set[str] = set()
    pending = [root_artifact]
    while pending:
        artifact_id = pending.pop()
        if artifact_id in reachable:
            continue
        store.validate(artifact_id)
        reachable.add(artifact_id)
        for reference in _artifact_references(store, artifact_id) - reachable:
            if not store.path_for(reference).is_dir():
                raise ValueError(
                    f"calibration artifact {artifact_id} references absent artifact {reference}"
                )
            pending.append(reference)
    return tuple(sorted(reachable))


def plan_rollover(
    run_output: str | Path,
    outputs_root: str | Path,
    results_root: str | Path,
    *,
    expected_config_hash: str,
    allow_failed: bool = False,
    allow_interrupted: bool = False,
) -> RolloverPlan:
    run = Path(run_output).resolve()
    outputs = Path(outputs_root).resolve()
    results = Path(results_root).resolve()
    if not expected_config_hash.startswith("sha256:") or len(expected_config_hash) != 71:
        raise ValueError("expected config hash must be a canonical sha256: digest")
    if run.parent == run or outputs.parent == outputs or results.parent == results:
        raise ValueError("rollover targets must not be filesystem roots")
    if (run / ".active-lease.json").exists():
        raise ValueError(f"run is actively leased and cannot be rolled over: {run}")

    manifest = _read_object(run / "manifest.json", "resident manifest")
    stored_hash = _preparation_config_hash(manifest)
    stored_status = str(manifest.get("status") or "")
    accepted_incomplete = (allow_failed and stored_status == "failed") or (
        allow_interrupted and stored_status == "interrupted"
    )
    if stored_status != "completed" and not accepted_incomplete:
        raise ValueError(
            "rollover requires a completed resident manifest or explicit acceptance of a failed "
            "or interrupted run"
        )
    if not stored_hash.startswith("sha256:") or len(stored_hash) != 71:
        raise ValueError("resident manifest has an invalid config hash")
    if stored_hash == expected_config_hash:
        raise ValueError("resident manifest already matches the expected configuration")

    calibration_artifact: str | None = None
    closure: tuple[str, ...] = ()
    logical_bytes = 0
    receipt_path = run / "calibration-input.json"
    if receipt_path.is_file():
        receipt = _read_object(receipt_path, "calibration receipt")
        if receipt.get("preparation_id") == expected_config_hash:
            calibration_artifact = str(receipt.get("artifact_id") or "")
            source_store = LocalArtifactStore(
                run / "artifacts",
                use_persistent_validation_cache=False,
            )
            closure = _validated_closure(source_store, calibration_artifact)
            logical_bytes = sum(
                sum(member.bytes for member in source_store.validate(artifact_id).files)
                for artifact_id in closure
            )

    suffix = stored_hash[7:19]
    run_archive = run.with_name(f"{run.name}--archive-{suffix}")
    outputs_archive = outputs.with_name(f"{outputs.name}--archive-{suffix}")
    results_archive = results.with_name(f"{results.name}--archive-{suffix}")
    for archive in (run_archive, outputs_archive, results_archive):
        if archive.exists():
            raise FileExistsError(f"rollover archive already exists: {archive}")

    return RolloverPlan(
        run,
        run_archive,
        outputs,
        outputs_archive,
        results,
        results_archive,
        stored_hash,
        expected_config_hash,
        stored_status,
        calibration_artifact,
        len(closure),
        logical_bytes,
        calibration_artifact is not None and (run / "state" / "teacher-traces").is_dir(),
    )


def execute_rollover(plan: RolloverPlan) -> Path:
    """Apply a validated rollover and return the fresh canonical run directory."""

    # Re-plan immediately before mutation so a newly acquired lease, changed
    # receipt, corrupt artifact, or occupied archive fails before any move.
    current = plan_rollover(
        plan.run_output,
        plan.outputs_root,
        plan.results_root,
        expected_config_hash=plan.expected_config_hash,
        allow_failed=plan.stored_status == "failed",
        allow_interrupted=plan.stored_status == "interrupted",
    )
    if current != plan:
        raise ValueError("rollover inputs changed after planning")

    source_store = LocalArtifactStore(
        plan.run_output / "artifacts",
        use_persistent_validation_cache=False,
    )
    closure = (
        _validated_closure(source_store, plan.calibration_artifact_id)
        if plan.calibration_artifact_id is not None
        else ()
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{plan.run_output.name}.rollover-",
            dir=plan.run_output.parent,
        )
    ).resolve()
    if staging.parent != plan.run_output.parent.resolve():
        raise RuntimeError("rollover staging escaped the run registry")

    moved_companions: list[tuple[Path, Path]] = []
    run_moved = False
    try:
        destination_store = LocalArtifactStore(
            staging / "artifacts",
            use_persistent_validation_cache=False,
        )
        for artifact_id in closure:
            destination_store.import_validated(source_store, artifact_id)

        if plan.calibration_artifact_id is not None:
            receipt = _read_object(
                plan.run_output / "calibration-input.json",
                "calibration receipt",
            )
            atomic_write_json(staging / "calibration-input.json", receipt)
            teacher_state = plan.run_output / "state" / "teacher-traces"
            if teacher_state.is_dir():
                shutil.copytree(teacher_state, staging / "state" / "teacher-traces")
        atomic_write_json(
            staging / "state" / "config-rollover.json",
            {
                "schema_version": 1,
                "stored_config_hash": plan.stored_config_hash,
                "expected_config_hash": plan.expected_config_hash,
                "stored_status": plan.stored_status,
                "archived_run": str(plan.run_archive),
                "archived_outputs": str(plan.outputs_archive),
                "archived_results": str(plan.results_archive),
                "calibration_artifact_id": plan.calibration_artifact_id,
                "calibration_artifact_count": plan.calibration_artifact_count,
                "calibration_logical_bytes": plan.calibration_logical_bytes,
            },
        )

        for source, archive in (
            (plan.outputs_root, plan.outputs_archive),
            (plan.results_root, plan.results_archive),
        ):
            if source.exists():
                source.rename(archive)
                moved_companions.append((source, archive))
        plan.run_output.rename(plan.run_archive)
        run_moved = True
        staging.rename(plan.run_output)
        return plan.run_output
    except BaseException:
        if run_moved and plan.run_archive.exists() and not plan.run_output.exists():
            plan.run_archive.rename(plan.run_output)
        for source, archive in reversed(moved_companions):
            if archive.exists() and not source.exists():
                archive.rename(source)
        raise
    finally:
        if staging.exists():
            # The staging path was created by this function and its resolved
            # parent/prefix were checked before any recursive cleanup.
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--expected-config-hash", required=True)
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="archive a failed resident run while preserving it as the rollover source",
    )
    parser.add_argument(
        "--allow-interrupted",
        action="store_true",
        help="archive an interrupted resident run while preserving it as the rollover source",
    )
    parser.add_argument("--apply", action="store_true", help="apply the rollover; default is dry-run")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = plan_rollover(
        args.run_output,
        args.outputs_root,
        args.results_root,
        expected_config_hash=args.expected_config_hash,
        allow_failed=args.allow_failed,
        allow_interrupted=args.allow_interrupted,
    )
    print(
        json.dumps(
            {
                **asdict(plan),
                "run_output": str(plan.run_output),
                "run_archive": str(plan.run_archive),
                "outputs_root": str(plan.outputs_root),
                "outputs_archive": str(plan.outputs_archive),
                "results_root": str(plan.results_root),
                "results_archive": str(plan.results_archive),
                "apply": bool(args.apply),
            },
            sort_keys=True,
            indent=2,
        )
    )
    if args.apply:
        execute_rollover(plan)
        print(f"rollover completed; fresh run directory: {plan.run_output}")
    else:
        print("dry run only; pass --apply after reviewing the plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
