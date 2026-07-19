from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nanoquant.runtime import (
    LogicalLayerState,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    write_logical_artifact,
)
from tools.cleanup_logical_artifact import main


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


def _arguments(root: Path, expected: str | None, *, apply: bool = False) -> list[str]:
    arguments = [
        "--artifact",
        str(root),
    ]
    if expected is not None:
        arguments.extend(("--expected-descriptor-sha256", expected))
    if apply:
        arguments.append("--apply")
    return arguments


def _command(root: Path, expected: str | None, *, apply: bool = False) -> list[str]:
    script = Path(__file__).parents[2] / "tools" / "cleanup_logical_artifact.py"
    return [sys.executable, str(script), *_arguments(root, expected, apply=apply)]


@pytest.mark.subprocess
def test_logical_artifact_cleanup_is_dry_run_and_hash_guarded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _artifact(tmp_path)
    digest = hashlib.sha256((root / "nanoquant-model.json").read_bytes()).hexdigest()

    dry_run = subprocess.run(_command(root, digest), check=True, capture_output=True, text=True)
    assert json.loads(dry_run.stdout)["deleted"] is False
    assert root.is_dir()

    with pytest.raises(ValueError, match="descriptor hash differs"):
        main(_arguments(root, "0" * 64, apply=True))
    assert root.is_dir()

    assert main(_arguments(root, digest, apply=True)) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] is True
    assert not root.exists()


def test_logical_artifact_cleanup_apply_requires_well_formed_expected_hash(tmp_path: Path) -> None:
    root = _artifact(tmp_path)

    with pytest.raises(ValueError, match="requires --expected-descriptor-sha256"):
        main(_arguments(root, None, apply=True))
    with pytest.raises(ValueError, match="64 hexadecimal"):
        main(_arguments(root, "not-a-hash", apply=True))
    assert root.is_dir()
