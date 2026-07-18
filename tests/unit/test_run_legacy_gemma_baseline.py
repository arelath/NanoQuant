import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tools.run_legacy_gemma_baseline import (
    _atomic_json,
    _calibration_tensor_path,
    _detach,
)

CALIBRATION_ARTIFACT = "sha256-" + "2" * 64


def _calibration_fixture(root: Path, shape: tuple[int, int] = (256, 2048)) -> Path:
    tensor_artifact = "sha256-" + "1" * 64
    tensor_root = root / "artifacts" / "11" / tensor_artifact
    tensor_root.mkdir(parents=True)
    save_file({"input_ids": torch.zeros(shape, dtype=torch.long)}, tensor_root / "tensors.safetensors")
    manifest_root = root / "artifacts" / CALIBRATION_ARTIFACT[7:9] / CALIBRATION_ARTIFACT
    manifest_root.mkdir(parents=True)
    (manifest_root / "manifest.json").write_text(
        json.dumps({"tensor_artifact": tensor_artifact, "fingerprint": "fixture"}),
        encoding="utf-8",
    )
    return tensor_root / "tensors.safetensors"


def test_calibration_tensor_path_validates_pinned_shape(tmp_path: Path) -> None:
    expected = _calibration_fixture(tmp_path)

    actual, manifest = _calibration_tensor_path(tmp_path, CALIBRATION_ARTIFACT)

    assert actual == expected.resolve()
    assert manifest["fingerprint"] == "fixture"


def test_calibration_tensor_path_rejects_wrong_shape(tmp_path: Path) -> None:
    _calibration_fixture(tmp_path, (2, 3))

    with pytest.raises(ValueError, match="pinned calibration shape"):
        _calibration_tensor_path(tmp_path, CALIBRATION_ARTIFACT)


def test_atomic_json_replaces_existing_payload(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    _atomic_json(path, {"status": "running"})
    _atomic_json(path, {"status": "complete"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "complete"}
    assert not path.with_suffix(".json.tmp").exists()


def test_detach_option_is_not_forwarded_to_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **options: object) -> SimpleNamespace:
        captured.update({"command": command, "options": options})
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(sys, "argv", ["tool", "--output", "run", "--detach", "--validate-only"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert _detach(tmp_path / "run") == 0

    assert "--detach" not in captured["command"]  # type: ignore[operator]
    assert "--validate-only" in captured["command"]  # type: ignore[operator]
    assert (tmp_path / "run.launcher.stdout.log").exists()
    assert (tmp_path / "run.launcher.stderr.log").exists()
