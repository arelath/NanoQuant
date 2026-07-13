import json
from pathlib import Path

import pytest

from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from tools.validate_resident_run import validate_resident_run

IDENTITY = {
    "config_hash": "sha256:config",
    "model_hash": "sha256:model",
    "plan_hash": "sha256-plan",
}


def _artifact(store: LocalArtifactStore, artifact_type: str, filename: str, payload: object) -> str:
    with store.begin_write(artifact_type) as writer:
        (writer.path / filename).write_text(json.dumps(payload), encoding="utf-8")
        return writer.commit().artifact_id


def _reference(artifact_id: str, artifact_type: str) -> dict[str, object]:
    return {"artifact_id": artifact_id, "artifact_type": artifact_type, "schema_version": 1}


def _write_journal(output: Path, records: list[dict[str, object]]) -> None:
    state = output / "state"
    state.mkdir(parents=True)
    (state / "journal.jsonl").write_text(
        "\n".join(json.dumps({"sequence": index, **record}) for index, record in enumerate(records, 1)) + "\n",
        encoding="utf-8",
    )


def test_validate_resident_run_follows_artifact_graph_and_allows_retired_activations(tmp_path: Path) -> None:
    output = tmp_path / "run"
    store = LocalArtifactStore(output / "artifacts")
    frozen = _artifact(store, "frozen-layer", "state.json", {"value": 1})
    old_activation = _artifact(store, "activation-generation", "activation-generation.json", {"block": 0})
    active_activation = _artifact(store, "activation-generation", "activation-generation.json", {"block": 1})
    layer = _artifact(
        store,
        "layer-result",
        "layer-result.json",
        {
            "identity": IDENTITY,
            "result": {
                "layer": {"block": {"index": 0}, "path": "mlp.gate_proj"},
                "frozen": _reference(frozen, "frozen-layer"),
            },
        },
    )
    block0 = _artifact(
        store,
        "block-result",
        "block-result.json",
        {
            "identity": IDENTITY,
            "block": {"index": 0},
            "activation_generation": _reference(old_activation, "activation-generation"),
            "frozen": _reference(frozen, "frozen-layer"),
        },
    )
    block1 = _artifact(
        store,
        "block-result",
        "block-result.json",
        {
            "identity": IDENTITY,
            "block": {"index": 1},
            "activation_generation": _reference(active_activation, "activation-generation"),
            "frozen": _reference(frozen, "frozen-layer"),
        },
    )
    _write_journal(
        output,
        [
            {
                "kind": "layer",
                "block": 0,
                "layer": "mlp.gate_proj",
                "artifact_id": layer,
                "identity": IDENTITY,
                "timestamp": "1",
            },
            {
                "kind": "block",
                "block": 0,
                "layer": None,
                "artifact_id": block0,
                "identity": IDENTITY,
                "timestamp": "2",
            },
            {
                "kind": "block",
                "block": 1,
                "layer": None,
                "artifact_id": block1,
                "identity": IDENTITY,
                "timestamp": "3",
            },
        ],
    )
    store.remove_artifact(old_activation, expected_type="activation-generation")
    cache = store.root / ".validation-cache.json"
    cache_before = cache.read_bytes()

    result = validate_resident_run(output, expected_blocks=2, require_complete=True)

    assert result.complete is True
    assert result.completed_blocks == (0, 1)
    assert result.journal_records == 3
    assert result.layer_records == 1
    assert result.block_records == 2
    assert result.artifacts_validated == 5
    assert result.artifacts_by_type == {
        "activation-generation": 1,
        "block-result": 2,
        "frozen-layer": 1,
        "layer-result": 1,
    }
    assert result.retired_activation_generations == (old_activation,)
    assert cache.read_bytes() == cache_before


def test_validate_resident_run_rejects_missing_durable_reference_and_incomplete_run(tmp_path: Path) -> None:
    output = tmp_path / "run"
    store = LocalArtifactStore(output / "artifacts")
    missing = "sha256-" + "f" * 64
    block = _artifact(
        store,
        "block-result",
        "block-result.json",
        {
            "identity": IDENTITY,
            "block": {"index": 0},
            "frozen": _reference(missing, "frozen-layer"),
        },
    )
    _write_journal(
        output,
        [
            {
                "kind": "block",
                "block": 0,
                "layer": None,
                "artifact_id": block,
                "identity": IDENTITY,
                "timestamp": "1",
            }
        ],
    )

    with pytest.raises(ValueError, match="incomplete"):
        validate_resident_run(output, expected_blocks=2, require_complete=True)
    with pytest.raises(ArtifactCorruptionError, match="unavailable"):
        validate_resident_run(output, expected_blocks=1, require_complete=True)
