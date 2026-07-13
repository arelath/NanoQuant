import json
from pathlib import Path

from nanoquant.infrastructure.artifacts import LocalArtifactStore
from tools.cleanup_run_activations import apply_activation_cleanup, plan_activation_cleanup


def _artifact(store: LocalArtifactStore, artifact_type: str, filename: str, payload: object) -> str:
    with store.begin_write(artifact_type) as writer:
        (writer.path / filename).write_text(json.dumps(payload), encoding="utf-8")
        return writer.commit().artifact_id


def test_cleanup_preserves_latest_active_generation_and_run_evidence(tmp_path: Path) -> None:
    output = tmp_path / "run"
    store = LocalArtifactStore(output / "artifacts")
    old_generation = _artifact(store, "activation-generation", "activation-generation.json", {"generation": 1})
    active_generation = _artifact(store, "activation-generation", "activation-generation.json", {"generation": 2})
    old_block = _artifact(
        store,
        "block-result",
        "block-result.json",
        {"activation_generation": {"artifact_id": old_generation}},
    )
    active_block = _artifact(
        store,
        "block-result",
        "block-result.json",
        {"activation_generation": {"artifact_id": active_generation}},
    )
    state = output / "state"
    state.mkdir(parents=True)
    journal = state / "journal.jsonl"
    records = (
        {"kind": "block", "artifact_id": old_block, "identity": {"config_hash": "old"}},
        {"kind": "block", "artifact_id": active_block, "identity": {"config_hash": "active"}},
    )
    journal.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    plan = plan_activation_cleanup(output)
    assert plan.active_config_hash == "active"
    assert plan.preserved_artifact == active_generation
    assert plan.candidates == (old_generation,)
    deleted, _deleted_bytes = apply_activation_cleanup(plan)

    assert deleted == 1
    assert not store.path_for(old_generation).exists()
    assert store.path_for(active_generation).exists()
    assert store.path_for(old_block).exists()
    assert journal.exists()
