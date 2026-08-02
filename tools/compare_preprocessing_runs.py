"""Verify exact preprocessing reproducibility between two independent resident runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401

from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

ROOT_TYPES = {
    "calibration": "calibration-stats",
    "objectives": "objective-specs",
    "plan": "quantization-plan",
}
RULE = "exact-resident-preprocessing-artifact-graph-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _artifact_ids(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        artifact_id = value.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id.startswith("sha256-"):
            found.add(artifact_id)
        for child in value.values():
            found.update(_artifact_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_artifact_ids(child))
    return found


def _validate_graph(store: LocalArtifactStore, roots: tuple[str, ...]) -> tuple[str, ...]:
    pending = list(roots)
    validated: set[str] = set()
    while pending:
        artifact_id = pending.pop()
        if artifact_id in validated:
            continue
        descriptor = store.validate(artifact_id)
        validated.add(artifact_id)
        artifact_root = store.path_for(artifact_id)
        for member in descriptor.files:
            if Path(member.path).suffix.lower() != ".json":
                continue
            payload = json.loads((artifact_root / member.path).read_text(encoding="utf-8"))
            pending.extend(sorted(_artifact_ids(payload) - validated))
    return tuple(sorted(validated))


def _load_run(run: Path) -> dict[str, Any]:
    resolved = run.resolve()
    state_path = resolved / "state" / "preprocessing.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported preprocessing state schema in {resolved}")
    config_hash = payload.get("resident_config_hash")
    if not isinstance(config_hash, str) or not config_hash.startswith("sha256:"):
        raise ValueError(f"invalid resident config hash in {resolved}")

    roots: dict[str, str] = {}
    for name, expected_type in ROOT_TYPES.items():
        reference = payload.get(name)
        if not isinstance(reference, dict):
            raise ValueError(f"missing {name} preprocessing reference in {resolved}")
        if reference.get("artifact_type") != expected_type or reference.get("schema_version") != 1:
            raise ValueError(f"invalid {name} preprocessing reference in {resolved}")
        artifact_id = reference.get("artifact_id")
        if not isinstance(artifact_id, str):
            raise ValueError(f"invalid {name} preprocessing artifact id in {resolved}")
        roots[name] = artifact_id

    store = LocalArtifactStore(resolved / "artifacts", use_persistent_validation_cache=False)
    validated = _validate_graph(store, tuple(roots.values()))
    for name, expected_type in ROOT_TYPES.items():
        descriptor = store.validate(roots[name])
        if descriptor.artifact_type != expected_type or descriptor.schema_version != 1:
            raise ValueError(f"{name} descriptor disagrees with preprocessing state in {resolved}")
    return {
        "run": str(resolved),
        "state_sha256": "sha256:" + hash_file(state_path),
        "resident_config_hash": config_hash,
        "roots": roots,
        "validated_artifact_count": len(validated),
        "validated_artifacts": list(validated),
    }


def compare_runs(run_a: Path, run_b: Path) -> dict[str, Any]:
    first = _load_run(run_a)
    second = _load_run(run_b)
    comparisons = {
        "preprocessing_state": first["state_sha256"] == second["state_sha256"],
        "resident_config_hash": first["resident_config_hash"] == second["resident_config_hash"],
        **{
            f"{name}_artifact": first["roots"][name] == second["roots"][name]
            for name in ROOT_TYPES
        },
        "transitive_artifact_graph": (
            first["validated_artifacts"] == second["validated_artifacts"]
        ),
    }
    return {
        "schema_version": 1,
        "rule": RULE,
        "passed": all(comparisons.values()),
        "comparisons": comparisons,
        "run_a": first,
        "run_b": second,
    }


def main() -> int:
    args = _parser().parse_args()
    result = compare_runs(args.run_a, args.run_b)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable comparison receipt: {args.output}")
    atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
