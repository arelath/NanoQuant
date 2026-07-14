import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from nanoquant.config.schema import ModelConfig, RunConfig
from nanoquant.domain.runs import RunStatus
from nanoquant.infrastructure.run_registry import (
    RegistryLock,
    discover_runs,
    read_registry,
    rebuild_registry,
    register_external_run,
    select_run,
)
from nanoquant.infrastructure.runs import RunDirectory, initial_manifest, launcher_provenance, transition


def _manifest(tmp_path: Path, run_id: str, created_at: str, experiment: int | None = None):
    launcher = tmp_path / f"{run_id}.py"
    launcher.write_text("# fixture\n", encoding="utf-8")
    manifest = initial_manifest(
        RunConfig(ModelConfig("fixture")),
        launcher_provenance(launcher, experiment),
        {},
        run_id=run_id,
    )
    return replace(manifest, created_at=created_at, updated_at=created_at)


def _write(output: Path, manifest) -> None:
    RunDirectory(output.parent, output.name).write_manifest(manifest)


def test_discovery_scans_managed_runs_and_reads_external_status_live(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    managed = _manifest(tmp_path, "run_managed", "2026-01-01T00:00:00+00:00", 1)
    external = _manifest(tmp_path, "run_external", "2026-01-02T00:00:00+00:00", 2)
    _write(root / managed.run_id, managed)
    external_path = tmp_path / "evidence" / "candidate"
    _write(external_path, external)

    register_external_run(root, external_path, external)
    register_external_run(root, external_path, external)

    assert len(read_registry(root)) == 1
    raw_record = json.loads((root / ".nanoquant" / "registry.jsonl").read_text(encoding="utf-8"))
    assert set(raw_record) == {
        "schema_version",
        "run_id",
        "path",
        "created_at",
        "component",
        "experiment_number",
    }
    found = {item.run_id: item for item in discover_runs(root)}
    assert set(found) == {"run_managed", "run_external"}
    assert found["run_managed"].source == "scan"
    assert found["run_external"].source == "registry"
    assert select_run(root, "latest").run_id == "run_external"
    assert select_run(root, "exp:1").run_id == "run_managed"

    running = transition(external, RunStatus.RUNNING)
    completed = transition(running, RunStatus.COMPLETED)
    _write(external_path, completed)
    assert select_run(root, "run_external").status == "completed"
    assert select_run(root, "latest", status=RunStatus.COMPLETED).run_id == "run_external"


def test_concurrent_registration_preserves_both_immutable_records(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    values = []
    for index in range(2):
        manifest = _manifest(tmp_path, f"run_{index}", f"2026-01-0{index + 1}T00:00:00+00:00")
        output = tmp_path / "evidence" / str(index)
        _write(output, manifest)
        values.append((output, manifest))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(register_external_run, root, output, manifest) for output, manifest in values]
        for future in futures:
            future.result()

    assert {record.run_id for record in read_registry(root)} == {"run_0", "run_1"}


def test_registry_skips_torn_and_mismatched_records_and_rebuilds(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    external_root = tmp_path / "evidence"
    manifest = _manifest(tmp_path, "run_good", "2026-01-01T00:00:00+00:00")
    output = external_root / "candidate"
    _write(output, manifest)
    register_external_run(root, output, manifest)
    path = root / ".nanoquant" / "registry.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '{"torn"\n'
        + json.dumps(
            {
                "schema_version": 1,
                "run_id": "wrong",
                "path": "../evidence/candidate",
                "created_at": manifest.created_at,
                "component": "resident-quantization",
                "experiment_number": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    found = discover_runs(root)
    assert any(item.run_id == "wrong" and item.integrity == "mismatched" for item in found)
    assert rebuild_registry(root, (external_root,)) == 1
    assert rebuild_registry(root, (external_root,)) == 1
    assert [record.run_id for record in read_registry(root)] == ["run_good"]


def test_registry_lock_takes_over_only_after_stale_window(tmp_path: Path) -> None:
    path = tmp_path / "registry.lock"
    owner = {"pid": 1, "hostname": "old", "acquired_at": 0}
    path.write_text(json.dumps(owner), encoding="utf-8")
    old = time.time() - 120
    os.utime(path, (old, old))

    with RegistryLock(path, timeout_seconds=0.1, stale_seconds=60) as lock:
        assert lock.taken_over_owner == owner
        assert path.is_file()
    assert not path.exists()
