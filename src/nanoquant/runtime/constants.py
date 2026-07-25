"""Stable deployment vocabulary kept independent of research packages."""

from __future__ import annotations

from enum import Enum


class PackedLayout(str, Enum):
    LLAMA_CPP_I32_LSB_V1 = "llama.cpp-i32-lsb-v1"


RUNTIME_ARTIFACT_SCHEMA_VERSION = 1


__all__ = ["RUNTIME_ARTIFACT_SCHEMA_VERSION", "PackedLayout"]
