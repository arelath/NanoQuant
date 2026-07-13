"""Pure profiling protocol used by instrumented numerical code."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol


class PhaseRecorder(Protocol):
    """Minimal observation surface safe to pass through domain code."""

    def phase(self, name: str, /, **attributes: object) -> AbstractContextManager[None]: ...

    def add(self, counter: str, value: float, /, **attributes: object) -> None: ...

    def mark(self, name: str, /, **attributes: object) -> None: ...


class _NullPhase(AbstractContextManager[None]):
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


_NULL_PHASE = _NullPhase()


class _NullRecorder:
    __slots__ = ()

    def phase(self, name: str, /, **attributes: object) -> AbstractContextManager[None]:
        return _NULL_PHASE

    def add(self, counter: str, value: float, /, **attributes: object) -> None:
        return None

    def mark(self, name: str, /, **attributes: object) -> None:
        return None


NULL_RECORDER: PhaseRecorder = _NullRecorder()
