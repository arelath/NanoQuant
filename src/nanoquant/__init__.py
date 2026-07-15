"""NanoQuant rewrite public package with lazy research-surface imports."""

from __future__ import annotations

from typing import Any

__all__ = ["ModelConfig", "RunConfig"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Keep ``nanoquant.runtime`` importable without loading research configuration."""

    if name in __all__:
        from nanoquant.config import ModelConfig, RunConfig

        return {"ModelConfig": ModelConfig, "RunConfig": RunConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
