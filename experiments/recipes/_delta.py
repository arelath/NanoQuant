"""Fail-closed construction of experiment-specific dataclass deltas."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any, TypeVar, cast

T = TypeVar("T")


def config_delta(parent: T, /, **changes: Any) -> T:
    """Replace only fields whose values actually differ from the inherited parent."""

    if not is_dataclass(parent) or isinstance(parent, type):
        raise TypeError("config_delta parent must be a dataclass instance")
    redundant = tuple(name for name, value in changes.items() if getattr(parent, name) == value)
    if redundant:
        names = ", ".join(sorted(redundant))
        raise ValueError(f"experiment config repeats inherited value(s): {names}")
    return cast(T, replace(parent, **changes))


__all__ = ["config_delta"]
