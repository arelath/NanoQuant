"""Local run directories, leases, provenance, and atomic manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import cast

from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.domain.runs import LauncherProvenance, RunManifest, RunStatus
from nanoquant.infrastructure.io_utils import atomic_write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"run_{timestamp}_{token_hex(4)}"


def _git(path: Path, *arguments: str) -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(path), *arguments], check=True, capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        return None


def launcher_provenance(
    path: str | Path, experiment_number: int | None, arguments: tuple[str, ...] = ()
) -> LauncherProvenance:
    launcher = Path(path).resolve()
    content_hash = "sha256:" + hashlib.sha256(launcher.read_bytes()).hexdigest()
    root_text = _git(launcher.parent, "rev-parse", "--show-toplevel")
    root = Path(root_text) if root_text else None
    relative = launcher.relative_to(root).as_posix() if root and launcher.is_relative_to(root) else None
    revision = _git(root, "rev-parse", "HEAD") if root else None
    return LauncherProvenance(
        "numbered_runfile" if experiment_number is not None else "python",
        experiment_number,
        relative,
        content_hash,
        revision,
        arguments,
    )


def validate_launcher_number(config: RunConfig, launcher_path: str | Path) -> None:
    expected = config.intent.experiment_number
    launcher = Path(launcher_path)
    match = re.match(r"^(\d{3})[-_]", launcher.name)
    actual = int(match.group(1)) if match else None
    if expected is None:
        if actual is not None and actual != 0:
            raise ValueError("EXP001 numbered launcher requires intent.experiment_number")
    elif actual != expected:
        raise ValueError(f"EXP001 launcher number {actual!r} does not match intent.experiment_number {expected}")
    canonical_prefix = None if expected is None else f"{expected:03d}-"
    if canonical_prefix is not None and config.intent.name.startswith(canonical_prefix):
        if launcher.stem != config.intent.name:
            raise ValueError(
                f"EXP002 launcher name {launcher.stem!r} does not match intent.name {config.intent.name!r}"
            )


class RunLease:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False
        self.taken_over_owner: dict[str, object] | None = None

    @staticmethod
    def _owner(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"RUN001 active lease is unreadable: {path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"RUN001 active lease has invalid owner metadata: {path}")
        return cast(dict[str, object], payload)

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = self._owner(self.path)
            hostname = str(owner.get("hostname", ""))
            raw_pid = owner.get("pid")
            pid = raw_pid if type(raw_pid) is int else -1
            if hostname == socket.gethostname() and not self._pid_is_alive(pid):
                try:
                    self.path.unlink()
                    descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except (FileNotFoundError, FileExistsError) as race:
                    raise RuntimeError(f"RUN001 active lease changed while taking ownership: {self.path}") from race
                self.taken_over_owner = owner
            else:
                raise RuntimeError(
                    f"RUN001 active lease exists: {self.path}; owner={json.dumps(owner, sort_keys=True)}"
                ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                {"pid": os.getpid(), "hostname": socket.gethostname(), "acquired_at": _now()},
                output,
                sort_keys=True,
            )
            output.flush()
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> RunLease:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class RunDirectory:
    def __init__(self, root: str | Path, run_id: str) -> None:
        self.root = Path(root) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("artifacts", "reports", "state"):
            (self.root / name).mkdir(exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    def lease(self) -> RunLease:
        return RunLease(self.root / ".active-lease.json")

    def write_manifest(self, manifest: RunManifest) -> None:
        atomic_write_json(self.manifest_path, to_dict(manifest))

    def read_manifest(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self.manifest_path.read_text(encoding="utf-8")))


def initial_manifest(
    config: RunConfig,
    launcher: LauncherProvenance,
    environment: dict[str, object],
    run_id: str | None = None,
    parent_run_id: str | None = None,
    forked_from_stage: str | None = None,
) -> RunManifest:
    return initial_manifest_from_resolved(
        config_hash(config),
        to_dict(config),
        launcher,
        environment,
        run_id=run_id,
        parent_run_id=parent_run_id,
        forked_from_stage=forked_from_stage,
    )


def initial_manifest_from_resolved(
    resolved_config_hash: str,
    resolved_config: dict[str, object],
    launcher: LauncherProvenance,
    environment: dict[str, object],
    run_id: str | None = None,
    parent_run_id: str | None = None,
    forked_from_stage: str | None = None,
) -> RunManifest:
    now = _now()
    return RunManifest(
        1,
        run_id or create_run_id(),
        RunStatus.CREATED,
        now,
        now,
        resolved_config_hash,
        resolved_config,
        launcher,
        environment,
        parent_run_id,
        forked_from_stage,
    )


def transition(
    manifest: RunManifest,
    status: RunStatus,
    *,
    artifacts: tuple[str, ...] | None = None,
    failure: dict[str, object] | None = None,
) -> RunManifest:
    allowed = {
        RunStatus.CREATED: {RunStatus.RUNNING, RunStatus.INTERRUPTED},
        RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED},
        RunStatus.INTERRUPTED: {RunStatus.RUNNING, RunStatus.FAILED},
        RunStatus.FAILED: set(),
        RunStatus.COMPLETED: set(),
    }
    if status not in allowed[manifest.status]:
        raise ValueError(f"invalid run transition {manifest.status.value} -> {status.value}")
    return replace(
        manifest,
        status=status,
        updated_at=_now(),
        artifacts=manifest.artifacts if artifacts is None else artifacts,
        failure=failure,
    )
