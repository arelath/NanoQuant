"""Immutable external-run registration and manifest-backed discovery."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from nanoquant.config.codec import from_dict
from nanoquant.domain.runs import RunManifest, RunStatus
from nanoquant.infrastructure.io_utils import safe_replace


@dataclass(frozen=True, slots=True)
class RegistryRecord:
    schema_version: int
    run_id: str
    path: str
    created_at: str
    component: str
    experiment_number: int | None


@dataclass(frozen=True, slots=True)
class DiscoveredRun:
    run_id: str
    path: Path
    source: str
    integrity: str
    manifest: RunManifest | None
    registry: RegistryRecord | None = None

    @property
    def status(self) -> str:
        return "unknown" if self.manifest is None else self.manifest.status.value

    @property
    def created_at(self) -> str:
        if self.manifest is not None:
            return self.manifest.created_at
        return "" if self.registry is None else self.registry.created_at

    @property
    def experiment_number(self) -> int | None:
        if self.manifest is not None:
            value = _experiment_number(self.manifest)
            if value is not None:
                return value
        return None if self.registry is None else self.registry.experiment_number

    @property
    def component(self) -> str:
        if self.manifest is not None:
            value = self.manifest.resolved_config.get("component")
            if isinstance(value, str):
                return value
        return "managed" if self.registry is None else self.registry.component


class RegistryLock:
    def __init__(self, path: Path, *, timeout_seconds: float = 5.0, stale_seconds: float = 60.0) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds
        self.acquired = False
        self.taken_over_owner: dict[str, object] | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                try:
                    owner = json.loads(self.path.read_text(encoding="utf-8"))
                    age = max(0.0, time.time() - self.path.stat().st_mtime)
                except (OSError, json.JSONDecodeError):
                    owner = {"unreadable": True}
                    age = self.stale_seconds + 1
                if age > self.stale_seconds:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        continue
                    self.taken_over_owner = cast(dict[str, object], owner) if isinstance(owner, dict) else {}
                    continue
                if time.monotonic() - started >= self.timeout_seconds:
                    raise RuntimeError(
                        f"RUN002 registry lock is busy: {self.path}; owner={json.dumps(owner, sort_keys=True)}"
                    ) from exc
                time.sleep(0.05)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "acquired_at": time.time(),
                    },
                    output,
                    sort_keys=True,
                )
                output.flush()
            self.acquired = True
            return

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> RegistryLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def _metadata_root(run_root: str | Path) -> Path:
    return Path(run_root).resolve() / ".nanoquant"


def registry_path(run_root: str | Path) -> Path:
    return _metadata_root(run_root) / "registry.jsonl"


def _registry_lock(run_root: str | Path) -> RegistryLock:
    return RegistryLock(_metadata_root(run_root) / "registry.lock")


def _experiment_number(manifest: RunManifest) -> int | None:
    if manifest.launcher.experiment_number is not None:
        return manifest.launcher.experiment_number
    direct = manifest.resolved_config.get("experiment_number")
    if type(direct) is int:
        return direct
    intent = manifest.resolved_config.get("intent")
    if isinstance(intent, dict):
        value = intent.get("experiment_number")
        if type(value) is int:
            return value
    return None


def _component(manifest: RunManifest) -> str:
    value = manifest.resolved_config.get("component")
    return value if isinstance(value, str) else "managed"


def _path_text(run_root: Path, output: Path) -> str:
    try:
        return Path(os.path.relpath(output, run_root)).as_posix()
    except ValueError:
        return str(output)


def _record(run_root: Path, output: Path, manifest: RunManifest) -> RegistryRecord:
    return RegistryRecord(
        1,
        manifest.run_id,
        _path_text(run_root, output),
        manifest.created_at,
        _component(manifest),
        _experiment_number(manifest),
    )


def _decode_registry_line(line: str) -> RegistryRecord | None:
    try:
        payload = json.loads(line)
        if not isinstance(payload, dict):
            return None
        record = RegistryRecord(
            int(payload["schema_version"]),
            str(payload["run_id"]),
            str(payload["path"]),
            str(payload["created_at"]),
            str(payload["component"]),
            None if payload.get("experiment_number") is None else int(payload["experiment_number"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return record if record.schema_version == 1 else None


def read_registry(run_root: str | Path) -> tuple[RegistryRecord, ...]:
    path = registry_path(run_root)
    if not path.exists():
        return ()
    records: list[RegistryRecord] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            record = _decode_registry_line(line)
            if record is None:
                continue
            key = (record.run_id, record.path)
            if key not in seen:
                seen.add(key)
                records.append(record)
    return tuple(records)


def register_external_run(run_root: str | Path, output: str | Path, manifest: RunManifest) -> dict[str, object] | None:
    root = Path(run_root).resolve()
    target = Path(output).resolve()
    if target == root or target.is_relative_to(root):
        return None
    record = _record(root, target, manifest)
    with _registry_lock(root) as lock:
        existing = read_registry(root)
        if not any(item.run_id == record.run_id and item.path == record.path for item in existing):
            path = registry_path(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as destination:
                destination.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")
                destination.flush()
        return lock.taken_over_owner


def _load_manifest(path: Path) -> tuple[RunManifest | None, str]:
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None, "corrupt"
        return from_dict(RunManifest, payload, path="manifest"), "ok"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, "corrupt"


def _registered_run(root: Path, record: RegistryRecord) -> DiscoveredRun:
    candidate = Path(record.path)
    target = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    manifest, integrity = _load_manifest(target / "manifest.json")
    if manifest is not None and manifest.run_id != record.run_id:
        integrity = "mismatched"
        manifest = None
    return DiscoveredRun(record.run_id, target, "registry", integrity, manifest, record)


def discover_runs(run_root: str | Path) -> tuple[DiscoveredRun, ...]:
    root = Path(run_root).resolve()
    found: list[DiscoveredRun] = []
    if root.exists():
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if not child.is_dir() or child.name == ".nanoquant":
                continue
            manifest, integrity = _load_manifest(child / "manifest.json")
            run_id = child.name if manifest is None else manifest.run_id
            found.append(DiscoveredRun(run_id, child.resolve(), "scan", integrity, manifest))
    found.extend(_registered_run(root, record) for record in read_registry(root))
    unique: dict[tuple[str, Path], DiscoveredRun] = {}
    for item in found:
        unique.setdefault((item.run_id, item.path), item)
    return tuple(unique.values())


def inspect_run_path(path: str | Path) -> DiscoveredRun:
    target = Path(path).resolve()
    manifest, integrity = _load_manifest(target / "manifest.json")
    if manifest is None and integrity == "missing":
        integrity = "unmanaged"
    run_id = target.name if manifest is None else manifest.run_id
    return DiscoveredRun(run_id, target, "path", integrity, manifest)


def selectable_runs(run_root: str | Path, *, status: RunStatus | None = None) -> tuple[DiscoveredRun, ...]:
    return tuple(
        item
        for item in discover_runs(run_root)
        if item.integrity == "ok"
        and item.manifest is not None
        and (status is None or item.manifest.status is status)
    )


def select_run(run_root: str | Path, selector: str, *, status: RunStatus | None = None) -> DiscoveredRun:
    candidates = selectable_runs(run_root, status=status)
    if selector == "latest":
        matching = candidates
    elif selector.startswith("exp:"):
        try:
            experiment = int(selector[4:])
        except ValueError as exc:
            raise ValueError(f"invalid experiment selector: {selector!r}") from exc
        matching = tuple(item for item in candidates if item.experiment_number == experiment)
    else:
        matching = tuple(item for item in candidates if item.run_id == selector)
    if not matching:
        raise FileNotFoundError(f"no run matches {selector!r}")
    newest_key = max((item.created_at, item.run_id) for item in matching)
    newest = tuple(item for item in matching if (item.created_at, item.run_id) == newest_key)
    paths = {item.path for item in newest}
    if len(paths) != 1:
        rendered = ", ".join(str(path) for path in sorted(paths, key=str))
        raise ValueError(f"ambiguous run selector {selector!r}: {rendered}")
    return newest[0]


def _manifest_paths(root: Path) -> set[Path]:
    if (root / "manifest.json").is_file():
        return {root / "manifest.json"}
    if not root.exists():
        return set()
    return set(root.rglob("manifest.json"))


def rebuild_registry(run_root: str | Path, include_roots: tuple[Path, ...] = ()) -> int:
    root = Path(run_root).resolve()
    targets: set[Path] = set()
    for record in read_registry(root):
        discovered = _registered_run(root, record)
        if discovered.integrity == "ok":
            targets.add(discovered.path)
    for include_root in include_roots:
        targets.update(path.parent.resolve() for path in _manifest_paths(include_root.resolve()))

    records: dict[tuple[str, str], RegistryRecord] = {}
    for target in targets:
        if target == root or target.is_relative_to(root):
            continue
        manifest, integrity = _load_manifest(target / "manifest.json")
        if integrity != "ok" or manifest is None:
            continue
        record = _record(root, target, manifest)
        records[(record.run_id, record.path)] = record

    with _registry_lock(root):
        path = registry_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix="registry-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                for record in sorted(records.values(), key=lambda item: (item.created_at, item.run_id, item.path)):
                    output.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")
                output.flush()
            safe_replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return len(records)
