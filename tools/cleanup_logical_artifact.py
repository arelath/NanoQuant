"""Dry-run or remove one validated generated logical runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path

from nanoquant.runtime import open_logical_artifact


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-descriptor-sha256")
    parser.add_argument("--apply", action="store_true", help="Delete after validation; omission is a dry run.")
    args = parser.parse_args()
    candidate = args.artifact
    attributes = getattr(candidate.lstat(), "st_file_attributes", 0) if candidate.exists() else 0
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if candidate.is_symlink() or (os.name == "nt" and bool(attributes & reparse_flag)):
        raise ValueError("logical artifact cleanup refuses a link or reparse-point target")
    root = candidate.resolve()
    if not root.is_dir() or root.parent == root or len(root.parts) < 3:
        raise ValueError(f"logical artifact cleanup target is not a safe directory: {root}")
    artifact = open_logical_artifact(root, verify_hashes=True)
    descriptor_hash = _hash_file(root / "nanoquant-model.json")
    if args.expected_descriptor_sha256 is not None:
        expected = args.expected_descriptor_sha256.lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("expected logical artifact descriptor hash must be 64 hexadecimal characters")
        if descriptor_hash != expected:
            raise ValueError(
                f"logical artifact descriptor hash differs: {descriptor_hash} != {expected}"
            )
    if args.apply and args.expected_descriptor_sha256 is None:
        raise ValueError("logical artifact cleanup apply requires --expected-descriptor-sha256")
    logical_bytes = _tree_bytes(root)
    payload = {
        "mode": "apply" if args.apply else "dry-run",
        "artifact": str(root),
        "descriptor_sha256": descriptor_hash,
        "block_count": len(artifact.manifest.blocks),
        "layer_count": artifact.manifest.layer_count,
        "logical_bytes": logical_bytes,
        "deleted": False,
    }
    if args.apply:
        shutil.rmtree(root)
        payload["deleted"] = True
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
