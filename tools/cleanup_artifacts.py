"""Dry-run or apply conservative garbage collection to NanoQuant artifact stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanoquant.infrastructure.artifact_gc import apply_artifact_gc, plan_artifact_gc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, action="append")
    parser.add_argument(
        "--ignore-evidence-path",
        type=Path,
        action="append",
        default=[],
        help="Ignore references below a retired run/file while leaving the evidence itself untouched.",
    )
    parser.add_argument("--keep-artifact", action="append", default=[])
    parser.add_argument("--minimum-age-hours", type=float, default=24.0)
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Delete candidates; omission is always a dry run.")
    args = parser.parse_args()

    evidence_roots = tuple(args.evidence_root or (Path("evidence"),))
    plan = plan_artifact_gc(
        args.artifact_root,
        evidence_roots,
        ignored_evidence_paths=tuple(args.ignore_evidence_path),
        keep_artifacts=tuple(args.keep_artifact),
        minimum_age_seconds=args.minimum_age_hours * 60 * 60,
    )
    payload: dict[str, object] = {
        "mode": "apply" if args.apply else "dry-run",
        "artifact_root": str(plan.artifact_root),
        "evidence_roots": [str(path.resolve()) for path in evidence_roots],
        "ignored_evidence_paths": [str(path.resolve()) for path in args.ignore_evidence_path],
        "root_artifact_count": len(plan.root_artifacts),
        "external_evidence_reference_count": plan.external_evidence_reference_count,
        "reachable_artifact_count": len(plan.reachable_artifacts),
        "candidate_artifact_count": len(plan.candidate_artifacts),
        "retained_for_age_count": len(plan.retained_for_age),
        "candidate_logical_bytes": plan.candidate_logical_bytes,
        "warnings": plan.warnings,
    }
    if args.list_candidates:
        payload["candidate_artifacts"] = plan.candidate_artifacts
    if args.apply:
        result = apply_artifact_gc(plan)
        payload["deleted_artifact_count"] = len(result.deleted_artifacts)
        payload["deleted_logical_bytes"] = result.deleted_logical_bytes
        if args.list_candidates:
            payload["deleted_artifacts"] = result.deleted_artifacts
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
