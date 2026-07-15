from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from nanoquant.runtime import (
    LogicalLayerState,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    write_logical_artifact,
)


def _artifact(tmp_path: Path) -> Path:
    spec = QuantizedLinearSpec("blocks.0.linear", "nanoquant-v1", 2, 2, 1, "float32", "float32")
    state = LogicalLayerState(
        spec,
        torch.ones(2, 1),
        torch.tensor([[1.0, -1.0]]),
        torch.ones(2),
        torch.ones(1),
        torch.ones(2),
    )
    return write_logical_artifact(
        tmp_path / "logical",
        RuntimeModelMetadata("fixture", "revision", "family", "config", "tokenizer"),
        {0: (state,)},
    ).root


def _command(root: Path, expected: str, *, apply: bool = False) -> list[str]:
    script = Path(__file__).parents[2] / "tools" / "cleanup_logical_artifact.py"
    command = [
        sys.executable,
        str(script),
        "--artifact",
        str(root),
        "--expected-descriptor-sha256",
        expected,
    ]
    if apply:
        command.append("--apply")
    return command


def test_logical_artifact_cleanup_is_dry_run_and_hash_guarded(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    digest = hashlib.sha256((root / "nanoquant-model.json").read_bytes()).hexdigest()

    dry_run = subprocess.run(_command(root, digest), check=True, capture_output=True, text=True)
    assert json.loads(dry_run.stdout)["deleted"] is False
    assert root.is_dir()

    mismatch = subprocess.run(_command(root, "0" * 64, apply=True), capture_output=True, text=True)
    assert mismatch.returncode != 0
    assert root.is_dir()

    applied = subprocess.run(_command(root, digest, apply=True), check=True, capture_output=True, text=True)
    assert json.loads(applied.stdout)["deleted"] is True
    assert not root.exists()


def test_logical_artifact_cleanup_apply_requires_well_formed_expected_hash(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    script = Path(__file__).parents[2] / "tools" / "cleanup_logical_artifact.py"

    missing = subprocess.run(
        [sys.executable, str(script), "--artifact", str(root), "--apply"],
        capture_output=True,
        text=True,
    )
    malformed = subprocess.run(_command(root, "not-a-hash", apply=True), capture_output=True, text=True)

    assert missing.returncode != 0
    assert malformed.returncode != 0
    assert root.is_dir()
