"""Typed stable diagnostic codes and base exceptions."""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Stable machine-readable codes used by raised operational errors."""

    ACTIVATION_CORRUPTION = "ACT001"
    ARTIFACT_CORRUPTION = "ART001"
    CONFIG_SCHEMA = "CFG001"
    RESOURCE_ADMISSION = "RES001"
    RESOURCE_EXHAUSTED = "RES002"
    RESOURCE_FALLBACK = "RES003"
    RUN_LEASE = "RUN001"
    RUN_IDENTITY = "RUN002"
    SOURCE_UNSUPPORTED = "SRC001"
    VRAM_SHARED_LIMIT = "VRAM001"
    VRAM_METER_UNAVAILABLE = "VRAM002"


class NanoQuantError(Exception):
    """Base error that formats and exposes a typed diagnostic code."""

    code: ErrorCode

    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code = code
        prefix = f"{code.value} "
        detail = message[len(prefix) :] if message.startswith(prefix) else message
        super().__init__(prefix + detail)


def coded_message(code: ErrorCode, message: str) -> str:
    """Format a diagnostic once even when compatibility callers include it."""

    prefix = f"{code.value} "
    return message if message.startswith(prefix) else prefix + message


__all__ = ["ErrorCode", "NanoQuantError", "coded_message"]
