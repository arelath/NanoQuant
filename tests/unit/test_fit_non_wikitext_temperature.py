from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nanoquant.infrastructure.io_utils import hash_file
from tools.fit_non_wikitext_temperature import (
    C4_DATASET,
    _retired_slice,
    _selection_arm_identity,
)


def _selection(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    baseline = {"mode": "postkd", "run_output": "baseline", "steps_completed": 256}
    selected = {"mode": "checkpoint", "run_output": "candidate", "steps_completed": 96}
    quality = {
        "status": "completed",
        "protocol": {"token_hash": "sha256:selection"},
        "arms": {"base": baseline, "candidate": selected},
    }
    quality_path = tmp_path / "quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    decision = {
        "schema_version": 1,
        "rule": "frozen-rule",
        "baseline": {"name": "base", "steps": 256},
        "selected_arm": "candidate",
        "selected_steps": 96,
        "selected_identity": selected,
        "protocol": {
            "quality_output": str(quality_path),
            "quality_sha256": hash_file(quality_path),
        },
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    return decision_path, selected


def test_selection_identity_binds_selected_arm_and_quality_evidence(tmp_path: Path) -> None:
    decision_path, selected = _selection(tmp_path)

    _decision, identity, quality_protocol = _selection_arm_identity(
        decision_path,
        role="selected",
        name="candidate",
        expected_steps=96,
    )

    assert identity == selected
    assert quality_protocol == {"token_hash": "sha256:selection"}
    with pytest.raises(ValueError, match="selected checkpoint"):
        _selection_arm_identity(
            decision_path,
            role="selected",
            name="candidate",
            expected_steps=64,
        )


def test_selection_identity_rejects_changed_quality_evidence(tmp_path: Path) -> None:
    decision_path, _selected = _selection(tmp_path)
    decision = json.loads(decision_path.read_text())
    Path(decision["protocol"]["quality_output"]).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        _selection_arm_identity(
            decision_path,
            role="baseline",
            name="base",
            expected_steps=256,
        )


def test_temperature_fit_requires_exact_retired_slice(tmp_path: Path) -> None:
    registry = tmp_path / "slices.json"
    entry = {
        "id": "selection",
        "dataset": C4_DATASET,
        "split": "validation",
        "offset": 10,
        "samples": 2,
        "sequence_length": 8,
        "token_start": 80,
        "token_end": 96,
        "token_hash": "sha256:tokens",
        "status": "retired",
        "consumer": "selector",
    }
    registry.write_text(json.dumps({"schema_version": 1, "slices": [entry]}), encoding="utf-8")

    selected, registry_hash = _retired_slice(
        registry,
        "selection",
        offset=10,
        samples=2,
        sequence_length=8,
        token_hash="sha256:tokens",
    )

    assert selected == entry
    assert registry_hash.startswith("sha256:")
    entry["status"] = "reserved"
    registry.write_text(json.dumps({"schema_version": 1, "slices": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="retired selection slice"):
        _retired_slice(
            registry,
            "selection",
            offset=10,
            samples=2,
            sequence_length=8,
            token_hash="sha256:tokens",
        )


def test_temperature_fit_script_entry_point_loads() -> None:
    script = Path(__file__).parents[2] / "tools" / "fit_non_wikitext_temperature.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
