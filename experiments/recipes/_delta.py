"""Fail-closed construction of experiment-specific dataclass deltas."""

from __future__ import annotations

import json
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar, cast

from nanoquant.config.codec import config_hash, from_dict
from nanoquant.config.schema import ModelConfig, RunConfig

T = TypeVar("T")


def retained_config(path: Path, expected_hash: str) -> RunConfig:
    """Load a frozen resolved configuration and verify its recorded semantic identity."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("resolved_config_hash") != expected_hash:
        raise ValueError("retained configuration provenance mismatch")
    config = from_dict(RunConfig, payload["resolved_config"])
    if config_hash(config) != expected_hash:
        raise ValueError("retained configuration hash mismatch")
    return config


def config_delta(parent: T, /, **changes: Any) -> T:
    """Replace only fields whose values actually differ from the inherited parent."""

    if not is_dataclass(parent) or isinstance(parent, type):
        raise TypeError("config_delta parent must be a dataclass instance")
    redundant = tuple(name for name, value in changes.items() if getattr(parent, name) == value)
    if redundant:
        names = ", ".join(sorted(redundant))
        raise ValueError(f"experiment config repeats inherited value(s): {names}")
    return cast(T, replace(parent, **changes))


def run_config_defaults(model_source: str) -> RunConfig:
    """Construct the schema-default run baseline for a required model source."""

    return RunConfig(model=ModelConfig(source=model_source))


__all__ = ["config_delta", "run_config_defaults"]
