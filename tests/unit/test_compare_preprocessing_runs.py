from __future__ import annotations

import json
from pathlib import Path

from nanoquant.infrastructure.artifacts import LocalArtifactStore
from tools.compare_preprocessing_runs import compare_runs


def _commit(store: LocalArtifactStore, artifact_type: str, payload: object) -> str:
    with store.begin_write(artifact_type) as writer:
        (writer.path / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        return writer.commit().artifact_id


def _run(root: Path, *, plan_value: int = 1) -> None:
    store = LocalArtifactStore(root / "artifacts")
    tensor = _commit(store, "tensor-bundle", {"values": [1, 2, 3]})
    calibration = _commit(store, "calibration-stats", {"tensor": {"artifact_id": tensor}})
    objectives = _commit(
        store,
        "objective-specs",
        {"calibration": {"artifact_id": calibration}, "tensor": {"artifact_id": tensor}},
    )
    plan = _commit(
        store,
        "quantization-plan",
        {"objectives": {"artifact_id": objectives}, "value": plan_value},
    )
    state = {
        "schema_version": 1,
        "resident_config_hash": "sha256:" + "1" * 64,
        "calibration": {
            "artifact_type": "calibration-stats",
            "artifact_id": calibration,
            "schema_version": 1,
        },
        "objectives": {
            "artifact_type": "objective-specs",
            "artifact_id": objectives,
            "schema_version": 1,
        },
        "plan": {
            "artifact_type": "quantization-plan",
            "artifact_id": plan,
            "schema_version": 1,
        },
    }
    state_path = root / "state" / "preprocessing.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")


def test_compare_runs_requires_exact_root_and_transitive_artifact_identity(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first)
    _run(second)

    result = compare_runs(first, second)

    assert result["passed"]
    assert all(result["comparisons"].values())
    assert result["run_a"]["validated_artifact_count"] == 4


def test_compare_runs_rejects_a_different_plan(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first)
    _run(second, plan_value=2)

    result = compare_runs(first, second)

    assert not result["passed"]
    assert not result["comparisons"]["plan_artifact"]
    assert not result["comparisons"]["transitive_artifact_graph"]
