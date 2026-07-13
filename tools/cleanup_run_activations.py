"""Dry-run or retire superseded activation generations from one resumable run."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from nanoquant.infrastructure.artifacts import LocalArtifactStore


@dataclass(frozen=True, slots=True)
class ActivationCleanupPlan:
    run_output: Path
    active_config_hash: str
    preserved_artifact: str
    candidates: tuple[str, ...]
    candidate_bytes: int
    warnings: tuple[str, ...]


def _artifact_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def plan_activation_cleanup(
    run_output: str | Path,
    *,
    active_config_hash: str | None = None,
) -> ActivationCleanupPlan:
    run_output = Path(run_output).resolve()
    journal = run_output / "state" / "journal.jsonl"
    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    block_records = [record for record in records if record.get("kind") == "block"]
    if not block_records:
        raise ValueError("run journal contains no committed blocks")
    if active_config_hash is None:
        active_config_hash = str(block_records[-1]["identity"]["config_hash"])
    active = [
        record
        for record in block_records
        if str(record.get("identity", {}).get("config_hash")) == active_config_hash
    ]
    if not active:
        raise ValueError(f"run journal contains no blocks for active config hash: {active_config_hash}")

    artifacts = LocalArtifactStore(run_output / "artifacts")
    warnings: list[str] = []
    generations: set[str] = set()
    active_generations: list[str] = []
    for record in block_records:
        block_id = str(record["artifact_id"])
        block_path = artifacts.path_for(block_id) / "block-result.json"
        try:
            payload = json.loads(block_path.read_text(encoding="utf-8"))
            generation = str(payload["activation_generation"]["artifact_id"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"could not inspect block artifact {block_id}: {exc}")
            continue
        generations.add(generation)
        if str(record.get("identity", {}).get("config_hash")) == active_config_hash:
            active_generations.append(generation)
    if not active_generations:
        raise ValueError("active block records contain no activation generation")
    preserved = active_generations[-1]
    candidates = []
    candidate_bytes = 0
    for artifact_id in sorted(generations - {preserved}):
        path = artifacts.path_for(artifact_id)
        if not path.exists():
            continue
        try:
            descriptor = artifacts.validate(artifact_id)
        except (OSError, ValueError) as exc:
            warnings.append(f"could not validate activation artifact {artifact_id}: {exc}")
            continue
        if descriptor.artifact_type != "activation-generation":
            warnings.append(
                f"refusing non-activation artifact {artifact_id}: {descriptor.artifact_type}"
            )
            continue
        candidates.append(artifact_id)
        candidate_bytes += _artifact_bytes(path)
    return ActivationCleanupPlan(
        run_output,
        active_config_hash,
        preserved,
        tuple(candidates),
        candidate_bytes,
        tuple(warnings),
    )


def apply_activation_cleanup(plan: ActivationCleanupPlan) -> tuple[int, int]:
    artifacts = LocalArtifactStore(plan.run_output / "artifacts")
    deleted = 0
    deleted_bytes = 0
    for artifact_id in plan.candidates:
        path = artifacts.path_for(artifact_id)
        size = _artifact_bytes(path)
        removed = artifacts.remove_artifact(artifact_id, expected_type="activation-generation")
        if removed:
            deleted += 1
            deleted_bytes += size
    return deleted, deleted_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--active-config-hash")
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Delete candidates; omission is always a dry run.")
    args = parser.parse_args()
    plan = plan_activation_cleanup(args.run_output, active_config_hash=args.active_config_hash)
    payload: dict[str, object] = {
        "mode": "apply" if args.apply else "dry-run",
        "run_output": str(plan.run_output),
        "active_config_hash": plan.active_config_hash,
        "preserved_artifact": plan.preserved_artifact,
        "candidate_artifact_count": len(plan.candidates),
        "candidate_logical_bytes": plan.candidate_bytes,
        "warnings": plan.warnings,
    }
    if args.list_candidates:
        payload["candidate_artifacts"] = plan.candidates
    if args.apply:
        deleted, deleted_bytes = apply_activation_cleanup(plan)
        payload["deleted_artifact_count"] = deleted
        payload["deleted_logical_bytes"] = deleted_bytes
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
