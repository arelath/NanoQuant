"""Local run directories, leases, provenance, and atomic manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import cast

from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.domain.runs import LauncherProvenance, RunManifest, RunStatus


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
    match = re.match(r"^(\d{3})[-_]", Path(launcher_path).name)
    actual = int(match.group(1)) if match else None
    if expected is None:
        if actual is not None and actual != 0:
            raise ValueError("EXP001 numbered launcher requires intent.experiment_number")
    elif actual != expected:
        raise ValueError(f"EXP001 launcher number {actual!r} does not match intent.experiment_number {expected}")


class RunLease:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"RUN001 active lease exists: {self.path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump({"pid": os.getpid(), "acquired_at": _now()}, output)
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
        payload = json.dumps(to_dict(manifest), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        descriptor, temporary = tempfile.mkstemp(prefix="manifest-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                output.write(payload + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

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
    now = _now()
    return RunManifest(
        1,
        run_id or create_run_id(),
        RunStatus.CREATED,
        now,
        now,
        config_hash(config),
        to_dict(config),
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
