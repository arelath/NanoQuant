import json
import socket
from pathlib import Path

import pytest

from nanoquant.config.schema import ModelConfig, ObservabilityConfig, RunConfig
from nanoquant.infrastructure.run_session import open_run_session
from nanoquant.infrastructure.runs import RunLease, initial_manifest, launcher_provenance


def _manifest(tmp_path: Path, run_id: str = "run_fixture"):
    launcher = tmp_path / "launcher.py"
    launcher.write_text("# fixture\n", encoding="utf-8")
    return initial_manifest(
        RunConfig(ModelConfig("fixture")),
        launcher_provenance(launcher, None),
        {},
        run_id=run_id,
    )


def test_run_session_owns_writer_identity_and_renders_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest = _manifest(tmp_path)
    observability = ObservabilityConfig(event_level="debug", console_level="warning")

    with open_run_session(
        output,
        manifest=manifest,
        observability=observability,
        registry_root=None,
        console=False,
    ) as session:
        assert session.run_id == "run_fixture"
        assert session.resumed is False
        event = session.events.emit("run", "info", "run.started", message="first\nsecond")
        assert event is not None and event.sequence == 1
        assert (output / ".active-lease.json").is_file()

    assert not (output / ".active-lease.json").exists()
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["run_id"] == "run_fixture"
    rendered = (output / "run.log").read_text(encoding="utf-8")
    assert rendered.count("\n") == 1
    assert 'message="first\\nsecond"' in rendered

    with open_run_session(
        output,
        manifest=_manifest(tmp_path, "ignored_new_id"),
        observability=observability,
        registry_root=None,
        console=False,
    ) as resumed:
        assert resumed.run_id == "run_fixture"
        assert resumed.resumed is True
        event = resumed.events.emit("run", "info", "run.resumed")
        assert event is not None and event.sequence == 2

    assert len((output / "run.log").read_text(encoding="utf-8").splitlines()) == 2


def test_run_lease_takes_over_dead_same_host_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / ".active-lease.json"
    owner = {"pid": 999_999, "hostname": socket.gethostname(), "acquired_at": "old"}
    path.write_text(json.dumps(owner), encoding="utf-8")
    monkeypatch.setattr(RunLease, "_pid_is_alive", staticmethod(lambda _pid: False))

    with RunLease(path) as lease:
        assert lease.taken_over_owner == owner
        current = json.loads(path.read_text(encoding="utf-8"))
        assert current["hostname"] == socket.gethostname()
        assert current["pid"] != owner["pid"]
    assert not path.exists()


def test_run_lease_rejects_live_or_foreign_owner(tmp_path: Path) -> None:
    path = tmp_path / ".active-lease.json"
    owner = {"pid": 1, "hostname": "different-host", "acquired_at": "now"}
    path.write_text(json.dumps(owner), encoding="utf-8")

    with pytest.raises(RuntimeError, match="RUN001.*different-host"):
        RunLease(path).acquire()
    assert json.loads(path.read_text(encoding="utf-8")) == owner


def test_registry_failure_degrades_to_canonical_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_registration(*_args, **_kwargs):
        raise PermissionError("injected registry failure")

    monkeypatch.setattr("nanoquant.infrastructure.run_registry.register_external_run", fail_registration)
    output = tmp_path / "external" / "run"
    with open_run_session(
        output,
        manifest=_manifest(tmp_path),
        observability=ObservabilityConfig(),
        registry_root=tmp_path / "runs",
        console=False,
    ) as session:
        session.events.emit("run", "info", "run.started")

    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["name"] for event in events] == ["run.registry_registration_failed", "run.started"]
