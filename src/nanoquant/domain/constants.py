"""Stable vocabulary shared by research workflows and persisted contracts."""

from __future__ import annotations

from enum import Enum


class BackendType(str, Enum):
    """Supported materialized linear execution representations."""

    DENSE = "dense"
    FACTORIZED = "factorized"


CONFIG_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "EVENT_SCHEMA_VERSION",
    "BackendType",
]
