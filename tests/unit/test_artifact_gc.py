import json
from pathlib import Path

from nanoquant.infrastructure.artifact_gc import apply_artifact_gc, plan_artifact_gc
from nanoquant.infrastructure.artifacts import LocalArtifactStore


def _artifact(store: LocalArtifactStore, name: str, payload: object) -> str:
    with store.begin_write(name) as writer:
        (writer.path / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        return writer.commit().artifact_id


def test_artifact_gc_preserves_transitive_references_and_evidence_files(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    leaf = _artifact(store, "leaf", {"value": 1})
    parent = _artifact(store, "parent", {"child": {"artifact_id": leaf}})
    garbage = _artifact(store, "garbage", {"value": 2})
    retired = _artifact(store, "retired", {"value": 3})
    evidence = tmp_path / "evidence"
    keep = evidence / "keep"
    discard = evidence / "retired-run"
    keep.mkdir(parents=True)
    discard.mkdir(parents=True)
    (keep / "journal.jsonl").write_text(json.dumps({"artifact_id": parent}), encoding="utf-8")
    retired_evidence = discard / "journal.jsonl"
    retired_evidence.write_text(json.dumps({"artifact_id": retired}), encoding="utf-8")
    external = "sha256-" + "f" * 64
    (keep / "other-store.json").write_text(
        json.dumps({"artifact_id": external}), encoding="utf-8"
    )

    plan = plan_artifact_gc(
        store.root,
        (evidence,),
        ignored_evidence_paths=(discard,),
        minimum_age_seconds=0,
    )

    assert set(plan.reachable_artifacts) == {parent, leaf}
    assert plan.external_evidence_reference_count == 1
    assert plan.warnings == ()
    assert set(plan.candidate_artifacts) == {garbage, retired}
    result = apply_artifact_gc(plan)
    assert set(result.deleted_artifacts) == {garbage, retired}
    assert store.path_for(parent).is_dir()
    assert store.path_for(leaf).is_dir()
    assert not store.path_for(garbage).exists()
    assert not store.path_for(retired).exists()
    assert retired_evidence.is_file()
    cache = json.loads((store.root / ".validation-cache.json").read_text(encoding="utf-8"))
    assert garbage not in cache and retired not in cache


def test_artifact_gc_age_and_explicit_keep_protect_unreferenced_artifacts(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    recent = _artifact(store, "recent", {"value": 1})
    explicit = _artifact(store, "explicit", {"value": 2})
    evidence = tmp_path / "evidence"
    evidence.mkdir()

    plan = plan_artifact_gc(
        store.root,
        (evidence,),
        keep_artifacts=(explicit,),
        minimum_age_seconds=24 * 60 * 60,
    )

    assert plan.candidate_artifacts == ()
    assert recent in plan.retained_for_age
    assert explicit in plan.reachable_artifacts


def test_artifact_gc_warns_for_broken_reference_inside_reachable_artifact(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    missing = "sha256-" + "e" * 64
    parent = _artifact(store, "parent", {"artifact_id": missing})
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "journal.json").write_text(
        json.dumps({"artifact_id": parent}), encoding="utf-8"
    )

    plan = plan_artifact_gc(store.root, (evidence,), minimum_age_seconds=0)

    assert plan.external_evidence_reference_count == 0
    assert plan.warnings == (
        f"reachable artifact {parent} references absent artifact: {missing}",
    )
