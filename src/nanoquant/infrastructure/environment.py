"""Allowlisted environment capture with defense-in-depth redaction."""

from __future__ import annotations

import os
import platform
import re
import sys
from importlib.metadata import distributions
from pathlib import Path

from dotenv import load_dotenv

ALLOWED_ENVIRONMENT = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TOKENIZERS_PARALLELISM",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    }
)
SECRET_PATTERN = re.compile(r"(?i)(token|secret|password|passwd|api[_-]?key|credential|authorization)")


def load_repository_dotenv(repository_root: str | Path) -> bool:
    """Load a repository-local ``.env`` with precedence over inherited values."""

    return load_dotenv(Path(repository_root).resolve() / ".env", override=True)


def redact(name: str, value: str) -> str:
    return "<redacted>" if SECRET_PATTERN.search(name) or SECRET_PATTERN.search(value) else value


def capture_environment(environ: dict[str, str] | None = None) -> dict[str, object]:
    source = os.environ if environ is None else environ
    selected = {name: redact(name, source[name]) for name in sorted(ALLOWED_ENVIRONMENT & source.keys())}
    package_map: dict[str, str] = {}
    for distribution in distributions():
        name = distribution.metadata["Name"]
        if name:
            package_map[name] = distribution.version
    packages = sorted(package_map.items())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(packages),
        "environment": selected,
    }
