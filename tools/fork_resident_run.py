"""Create an isolated, zero-copy hard-link fork of a resident run directory."""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import _paths  # noqa: F401

from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-output", type=Path, required=True)
    parser.add_argument("--derived-run-output", type=Path, required=True)
    parser.add_argument("--purpose", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardlink_tree(source: Path, destination: Path) -> int:
    linked = 0
    for root, directories, filenames in os.walk(source):
        root_path = Path(root)
        target_root = destination / root_path.relative_to(source)
        target_root.mkdir(parents=True, exist_ok=True)
        directories.sort()
        for filename in sorted(filenames):
            source_path = root_path / filename
            if source_path.is_symlink():
                raise ValueError(f"resident run contains a symbolic link: {source_path}")
            os.link(source_path, target_root / filename)
            linked += 1
    return linked


def fork_resident_run(source: Path, destination: Path, *, purpose: str) -> int:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise ValueError(f"source resident run does not exist: {source}")
    if not (source / "manifest.json").is_file():
        raise ValueError(f"source resident run has no manifest: {source}")
    if destination.exists():
        raise ValueError(f"derived run output already exists: {destination}")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("source and derived run must not contain one another")
    if not purpose.strip():
        raise ValueError("derived run purpose must be non-empty")

    source_manifest_hash = _sha256(source / "manifest.json")
    with atomic_workspace(destination) as temporary:
        linked_files = _hardlink_tree(source, temporary)
        atomic_write_json(
            temporary / "derived-run-provenance.json",
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_run_output": str(source),
                "source_manifest_sha256": source_manifest_hash,
                "purpose": purpose,
                "linked_source_files": linked_files,
            },
        )
    return linked_files


def main() -> int:
    args = _parser().parse_args()
    linked_files = fork_resident_run(
        args.source_run_output,
        args.derived_run_output,
        purpose=args.purpose,
    )
    print(f"created {args.derived_run_output} with {linked_files} hard-linked source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
