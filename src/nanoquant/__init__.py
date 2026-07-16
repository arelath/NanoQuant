"""NanoQuant rewrite public package with lazy research-surface imports."""

from __future__ import annotations

import os
from typing import Any

__all__ = ["ModelConfig", "RunConfig"]
__version__ = "0.1.0"


def _configure_cuda_allocator() -> None:
    """Enable expandable CUDA segments unless the operator chose an explicit value."""

    name = "PYTORCH_CUDA_ALLOC_CONF"
    current = os.environ.get(name, "")
    entries = tuple(item.strip() for item in current.split(",") if item.strip())
    if any(item.split(":", 1)[0].strip() == "expandable_segments" for item in entries):
        return
    os.environ[name] = ",".join((*entries, "expandable_segments:True"))


_configure_cuda_allocator()


def __getattr__(name: str) -> Any:
    """Keep ``nanoquant.runtime`` importable without loading research configuration."""

    if name in __all__:
        from nanoquant.config import ModelConfig, RunConfig

        return {"ModelConfig": ModelConfig, "RunConfig": RunConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
