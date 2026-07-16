"""Shared durable file hashing and atomic replacement primitives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def hash_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 hex digest using bounded reads."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
