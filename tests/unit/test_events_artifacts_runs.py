import json
import os
from pathlib import Path

import pytest

from nanoquant.config.schema import IntentConfig, ModelConfig, RunConfig
from nanoquant.domain.runs import RunStatus
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.cache import explain_reuse, semantic_key
from nanoquant.infrastructure.environment import capture_environment
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.runs import (
    RunDirectory,
    initial_manifest,
    launcher_provenance,
    transition,
    validate_launcher_number,
)


def test_events_are_monotonic_across_reopen_and_spans(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(path, "run_test")
    sink.emit("one", "info", "first")
    with sink.span("two", "work"):
        sink.emit("two", "warning", "middle", code="TEST001")
    JsonlEventSink(path, "run_test").emit("three", "info", "last")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, 6))
    assert events[1]["span_id"] == events[3]["span_id"]


def test_environment_is_allowlisted_and_secrets_are_redacted() -> None:
    captured = capture_environment(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_TOKEN": "must-not-leak",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    assert captured["environment"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    assert "must-not-leak" not in json.dumps(captured)


def test_content_addressed_commit_deduplicates_and_detects_corruption(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    ids = []
    for _ in range(2):
        with store.begin_write("fixture") as writer:
            (writer.path / "value.txt").write_text("same", encoding="utf-8")
            ids.append(writer.commit().artifact_id)
    assert ids[0] == ids[1]
    store.validate(ids[0])
    (store.path_for(ids[0]) / "value.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ArtifactCorruptionError, match="ART001"):
        store.validate(ids[0])


def test_validation_cache_uses_unique_temporary_files_across_store_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    stores = (LocalArtifactStore(root), LocalArtifactStore(root))
    real_replace = os.replace
    cache_temporaries: list[Path] = []

    def record_cache_temporary(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name == ".validation-cache.json":
            cache_temporaries.append(Path(source))
        real_replace(source, destination)

    monkeypatch.setattr("nanoquant.infrastructure.artifacts.os.replace", record_cache_temporary)
    for index, store in enumerate(stores):
        with store.begin_write("fixture") as writer:
            (writer.path / "value.txt").write_text(str(index), encoding="utf-8")
            writer.commit()

    assert len(cache_temporaries) == 2
    assert cache_temporaries[0] != cache_temporaries[1]
    assert not list(root.glob(".validation-cache-*.tmp"))


def test_validation_cache_failure_does_not_fail_durable_artifact_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    real_replace = os.replace

    def deny_cache_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name == ".validation-cache.json":
            raise PermissionError("simulated Windows sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr("nanoquant.infrastructure.artifacts.os.replace", deny_cache_replace)
    with store.begin_write("fixture") as writer:
        (writer.path / "value.txt").write_text("durable", encoding="utf-8")
        artifact_id = writer.commit().artifact_id

    assert store.path_for(artifact_id).is_dir()
    assert store.validate(artifact_id).artifact_id == artifact_id
    assert not list(store.root.glob(".validation-cache-*.tmp"))


def test_uncommitted_writer_is_not_discoverable(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(RuntimeError):
        with store.begin_write("fixture") as writer:
            (writer.path / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("crash")
    assert not list(store.root.glob("??/sha256-*"))


def test_run_manifest_lifecycle_and_atomic_replacement(tmp_path: Path) -> None:
    launcher = tmp_path / "001_test.py"
    launcher.write_text("# fixture\n", encoding="utf-8")
    config = RunConfig(ModelConfig("x"), intent=IntentConfig(experiment_number=1))
    provenance = launcher_provenance(launcher, 1)
    manifest = initial_manifest(config, provenance, {}, run_id="run_fixture")
    directory = RunDirectory(tmp_path / "runs", manifest.run_id)
    directory.write_manifest(manifest)
    running = transition(manifest, RunStatus.RUNNING)
    directory.write_manifest(running)
    completed = transition(running, RunStatus.COMPLETED, artifacts=("artifact",))
    directory.write_manifest(completed)
    assert directory.read_manifest()["status"] == "completed"
    assert not list(directory.root.glob("manifest-*.tmp"))
    with pytest.raises(ValueError, match="invalid run transition"):
        transition(completed, RunStatus.RUNNING)


def test_failed_run_can_return_to_running_without_losing_identity(tmp_path: Path) -> None:
    launcher_path = tmp_path / "001_test.py"
    launcher_path.write_text("# fixture\n", encoding="utf-8")
    launcher = launcher_provenance(launcher_path, 1)
    manifest = initial_manifest(
        RunConfig(ModelConfig("x")),
        launcher,
        {},
        run_id="run_fixture",
    )
    running = transition(manifest, RunStatus.RUNNING)
    failed = transition(
        running,
        RunStatus.FAILED,
        artifacts=("artifact",),
        failure={"type": "RuntimeError", "message": "injected"},
    )

    resumed = transition(failed, RunStatus.RUNNING)

    assert resumed.run_id == failed.run_id
    assert resumed.artifacts == ("artifact",)
    assert resumed.failure is None


def test_launcher_number_validation() -> None:
    config = RunConfig(ModelConfig("x"), intent=IntentConfig(experiment_number=19))
    validate_launcher_number(config, "019_baseline.py")
    with pytest.raises(ValueError, match="does not match"):
        validate_launcher_number(config, "018_other.py")

    canonical = RunConfig(
        ModelConfig("x"),
        intent=IntentConfig(experiment_number=19, name="019-baseline"),
    )
    validate_launcher_number(canonical, "019-baseline.py")
    with pytest.raises(ValueError, match="launcher name"):
        validate_launcher_number(canonical, "019-other.py")


def test_semantic_cache_explains_reuse_and_precise_invalidation() -> None:
    first = semantic_key("calibrate", "online-fisher", "1", {"sample_count": 8, "presentation": "ignored"})
    same = semantic_key("calibrate", "online-fisher", "1", {"sample_count": 8, "presentation": "ignored"})
    changed = semantic_key("calibrate", "online-fisher", "1", {"sample_count": 9, "presentation": "ignored"})
    assert explain_reuse(first, same).reusable
    explanation = explain_reuse(first, changed)
    assert not explanation.reusable
    assert explanation.changed_paths == ("request.sample_count",)
