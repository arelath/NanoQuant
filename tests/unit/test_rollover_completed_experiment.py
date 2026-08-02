import json
from pathlib import Path

import pytest

from nanoquant.config.codec import semantic_hash
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from tools.rollover_completed_experiment import execute_rollover, plan_rollover


def _artifact(store: LocalArtifactStore, artifact_type: str, payload: object) -> str:
    with store.begin_write(artifact_type) as writer:
        (writer.path / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        return writer.commit().artifact_id


def test_rollover_archives_old_run_and_preserves_exact_calibration_closure(
    tmp_path: Path,
) -> None:
    run = tmp_path / "evidence" / "030" / "030-fixture"
    outputs = tmp_path / "outputs" / "030"
    results = tmp_path / "Results" / "030"
    store = LocalArtifactStore(run / "artifacts")
    tensor = _artifact(store, "calibration-token-dataset", {"tokens": [1, 2, 3]})
    trace = _artifact(store, "teacher-trace-dataset", {"responses": ["complete"]})
    calibration = _artifact(
        store,
        "calibration-dataset-manifest",
        {
            "tensor_artifact": tensor,
            "teacher_trace_artifacts": [{"artifact_id": trace}],
        },
    )
    unrelated = _artifact(store, "frozen-block", {"legacy": True})
    old_hash = "sha256:" + "a" * 64
    new_hash = "sha256:" + "b" * 64
    (run / "manifest.json").write_text(
        json.dumps({"status": "completed", "config_hash": old_hash}),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 2,
        "sample_count": 2,
        "sequence_length": 4,
        "seed": 0,
        "preparation_id": new_hash,
        "artifact_id": calibration,
        "fingerprint": "sha256:fixture",
    }
    (run / "calibration-input.json").write_text(json.dumps(receipt), encoding="utf-8")
    teacher_state = run / "state" / "teacher-traces"
    teacher_state.mkdir(parents=True)
    (teacher_state / "trace.jsonl").write_text('{"kind":"header"}\n', encoding="utf-8")
    outputs.mkdir(parents=True)
    results.mkdir(parents=True)
    (outputs / "old-output.json").write_text("old", encoding="utf-8")
    (results / "old-result.md").write_text("old", encoding="utf-8")

    plan = plan_rollover(run, outputs, results, expected_config_hash=new_hash)

    assert run.is_dir()
    assert plan.calibration_artifact_count == 3
    assert not plan.run_archive.exists()
    fresh = execute_rollover(plan)

    assert fresh == run
    assert plan.run_archive.is_dir()
    assert plan.outputs_archive.joinpath("old-output.json").read_text(encoding="utf-8") == "old"
    assert plan.results_archive.joinpath("old-result.md").read_text(encoding="utf-8") == "old"
    assert not (run / "manifest.json").exists()
    assert json.loads((run / "calibration-input.json").read_text(encoding="utf-8")) == receipt
    assert (run / "state" / "teacher-traces" / "trace.jsonl").is_file()
    rollover = json.loads((run / "state" / "config-rollover.json").read_text(encoding="utf-8"))
    assert rollover["stored_config_hash"] == old_hash
    assert rollover["expected_config_hash"] == new_hash

    fresh_store = LocalArtifactStore(run / "artifacts", use_persistent_validation_cache=False)
    for artifact_id in (calibration, tensor, trace):
        fresh_store.validate(artifact_id)
    assert not fresh_store.path_for(unrelated).exists()
    LocalArtifactStore(
        plan.run_archive / "artifacts",
        use_persistent_validation_cache=False,
    ).validate(unrelated)


def test_rollover_without_new_calibration_starts_with_an_empty_fresh_run(
    tmp_path: Path,
) -> None:
    run = tmp_path / "evidence" / "030" / "030-fixture"
    outputs = tmp_path / "outputs" / "030"
    results = tmp_path / "Results" / "030"
    store = LocalArtifactStore(run / "artifacts")
    old_calibration = _artifact(store, "calibration-token-dataset", {"tokens": [1, 2, 3]})
    old_hash = "sha256:" + "a" * 64
    new_hash = "sha256:" + "b" * 64
    (run / "manifest.json").write_text(
        json.dumps({"status": "completed", "config_hash": old_hash}),
        encoding="utf-8",
    )
    (run / "calibration-input.json").write_text(
        json.dumps(
            {
                "preparation_id": old_hash,
                "artifact_id": old_calibration,
            }
        ),
        encoding="utf-8",
    )

    plan = plan_rollover(run, outputs, results, expected_config_hash=new_hash)

    assert plan.calibration_artifact_id is None
    assert plan.calibration_artifact_count == 0
    assert plan.calibration_logical_bytes == 0
    fresh = execute_rollover(plan)

    assert not (fresh / "calibration-input.json").exists()
    assert not LocalArtifactStore(fresh / "artifacts").path_for(old_calibration).exists()
    assert (fresh / "state" / "config-rollover.json").is_file()
    LocalArtifactStore(
        plan.run_archive / "artifacts",
        use_persistent_validation_cache=False,
    ).validate(old_calibration)


def test_rollover_compares_expected_hash_with_embedded_canonical_run_config(
    tmp_path: Path,
) -> None:
    run = tmp_path / "evidence" / "030" / "030-fixture"
    run.mkdir(parents=True)
    canonical = {"schema_version": 1, "model": {"source": "fixture/model"}}
    expected_hash = semantic_hash(canonical)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "config_hash": "sha256:" + "a" * 64,
                "resolved_config": {"canonical_run_config": canonical},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already matches"):
        plan_rollover(
            run,
            tmp_path / "outputs" / "030",
            tmp_path / "Results" / "030",
            expected_config_hash=expected_hash,
        )


def test_rollover_requires_explicit_acceptance_and_preserves_failed_run(tmp_path: Path) -> None:
    run = tmp_path / "evidence" / "054" / "054-fixture"
    outputs = tmp_path / "outputs" / "054"
    results = tmp_path / "Results" / "054"
    run.mkdir(parents=True)
    old_hash = "sha256:" + "a" * 64
    new_hash = "sha256:" + "b" * 64
    (run / "manifest.json").write_text(
        json.dumps({"status": "failed", "config_hash": old_hash}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicit acceptance"):
        plan_rollover(run, outputs, results, expected_config_hash=new_hash)

    plan = plan_rollover(
        run,
        outputs,
        results,
        expected_config_hash=new_hash,
        allow_failed=True,
    )
    fresh = execute_rollover(plan)

    assert plan.stored_status == "failed"
    assert plan.run_archive.joinpath("manifest.json").is_file()
    assert not fresh.joinpath("manifest.json").exists()
    rollover = json.loads((fresh / "state" / "config-rollover.json").read_text(encoding="utf-8"))
    assert rollover["stored_status"] == "failed"
