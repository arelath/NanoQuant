"""Shared durable file hashing and atomic replacement primitives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


def hash_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 hex digest using bounded reads."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_canonical_text_file(path: str | Path) -> str:
    """Hash text source with checkout-specific line endings normalized to LF."""

    content = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def safe_replace(
    source: str | Path,
    destination: str | Path,
    *,
    attempts: int = 5,
    suppress_errors: bool = False,
) -> bool:
    """Atomically replace a path, retrying transient Windows sharing violations."""

    if attempts <= 0:
        raise ValueError("replace attempts must be positive")
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return True
        except PermissionError:
            if attempt + 1 < attempts:
                time.sleep(0.01 * (2**attempt))
                continue
            if not suppress_errors:
                raise
        except OSError:
            if not suppress_errors:
                raise
        return False
    return False


@dataclass(slots=True)
class AtomicWorkspace:
    """Stage a directory tree beside its destination and publish it atomically.

    Existing destinations are rejected by default because replacing a non-empty
    directory is not atomic on every supported platform. Callers that implement
    a separately validated backup protocol may opt in to replacement.
    """

    destination: Path
    replace_existing: bool = False
    prefix: str | None = None
    _temporary: Path | None = None
    _published: bool = False

    def __enter__(self) -> Path:
        destination = self.destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not self.replace_existing:
            raise FileExistsError(f"atomic workspace destination already exists: {destination}")
        self.destination = destination
        self._temporary = Path(
            tempfile.mkdtemp(
                prefix=self.prefix or f".{destination.name}-",
                dir=destination.parent,
            )
        )
        return self._temporary

    def publish(self) -> Path:
        if self._temporary is None:
            raise RuntimeError("atomic workspace has not been entered")
        if self._published:
            raise RuntimeError("atomic workspace was already published")
        if self.destination.exists():
            if not self.replace_existing:
                raise FileExistsError(
                    f"atomic workspace destination already exists: {self.destination}"
                )
            backup = self.destination.with_name(f".{self.destination.name}.backup")
            if backup.exists():
                raise FileExistsError(f"atomic workspace backup already exists: {backup}")
            safe_replace(self.destination, backup)
            try:
                safe_replace(self._temporary, self.destination)
            except BaseException:
                safe_replace(backup, self.destination)
                raise
            shutil.rmtree(backup)
        else:
            safe_replace(self._temporary, self.destination)
        self._published = True
        return self.destination

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self._temporary is not None and self._temporary.exists():
            shutil.rmtree(self._temporary, ignore_errors=True)


@contextmanager
def atomic_workspace(
    destination: str | Path,
    *,
    replace_existing: bool = False,
    prefix: str | None = None,
) -> Iterator[Path]:
    """Yield a staging directory, publishing it on successful context exit."""

    transaction = AtomicWorkspace(Path(destination), replace_existing, prefix)
    with transaction as temporary:
        yield temporary
        transaction.publish()


def atomic_write_json(
    path: str | Path,
    payload: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
    suppress_replace_errors: bool = False,
) -> bool:
    """Write JSON durably beside its destination and atomically publish it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(
                payload,
                output,
                sort_keys=sort_keys,
                indent=indent,
                ensure_ascii=ensure_ascii,
                allow_nan=allow_nan,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        return safe_replace(
            temporary,
            destination,
            suppress_errors=suppress_replace_errors,
        )
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    suppress_replace_errors: bool = False,
) -> bool:
    """Write UTF-8 text durably beside its destination and atomically publish it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        return safe_replace(
            temporary,
            destination,
            suppress_errors=suppress_replace_errors,
        )
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def rewrite_linked_text(path: str | Path, text: str) -> None:
    """Durably rewrite a file without replacing its inode, preserving published hard links."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        atomic_write_text(destination, text)
        return
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
