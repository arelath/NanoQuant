"""Dependency-light atomic directory transactions for deployment artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_output_directory(
    destination: str | Path,
    *,
    prefix: str | None = None,
    replace_existing: bool = False,
) -> Iterator[Path]:
    """Publish a newly built directory atomically and always remove staging."""

    target = Path(destination).resolve()
    if target.exists() and not replace_existing:
        raise FileExistsError(f"atomic output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=prefix or f".{target.name}-", dir=target.parent)
    )
    backup: Path | None = None
    try:
        yield temporary
        if target.exists():
            backup = target.with_name(f".{target.name}-previous")
            if backup.exists():
                raise FileExistsError(f"atomic output backup already exists: {backup}")
            os.replace(target, backup)
            try:
                os.replace(temporary, target)
            except BaseException:
                os.replace(backup, target)
                raise
            shutil.rmtree(backup)
            backup = None
        else:
            os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)


__all__ = ["atomic_output_directory"]
