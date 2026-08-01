"""Strict schema-aware configuration decoding and canonical serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass, replace
from difflib import get_close_matches
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints

import yaml
from pydantic import ConfigDict, TypeAdapter, ValidationError

from .schema import RunConfig

T = TypeVar("T")


class ConfigDecodeError(ValueError):
    """Configuration is invalid at a precise dotted path."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


def _configure_dataclass_tree(annotation: Any, seen: set[type[object]] | None = None) -> None:
    """Apply fail-closed Pydantic settings to plain dataclass contracts."""

    visited = set() if seen is None else seen
    origin = get_origin(annotation)
    if origin is not None:
        for argument in get_args(annotation):
            _configure_dataclass_tree(argument, visited)
        return
    if not isinstance(annotation, type) or not is_dataclass(annotation) or annotation in visited:
        return
    visited.add(annotation)
    cast(Any, annotation).__pydantic_config__ = ConfigDict(extra="forbid")
    for hint in get_type_hints(annotation).values():
        _configure_dataclass_tree(hint, visited)


@cache
def _adapter(annotation: Any) -> TypeAdapter[Any]:
    _configure_dataclass_tree(annotation)
    return TypeAdapter(annotation)


def _render_location(path: str, location: tuple[int | str, ...]) -> str:
    rendered = path
    for part in location:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered


def _annotation_at(annotation: Any, location: tuple[int | str, ...]) -> Any:
    current = annotation
    for part in location[:-1]:
        origin = get_origin(current)
        if origin is not None:
            arguments = tuple(item for item in get_args(current) if item is not type(None))
            current = arguments[0] if arguments else Any
        if isinstance(part, int):
            continue
        if isinstance(current, type) and is_dataclass(current):
            current = get_type_hints(current).get(part, Any)
    return current


def _decode(value: Any, annotation: Any, path: str) -> Any:
    try:
        payload = json.dumps(value, allow_nan=True)
        return _adapter(annotation).validate_json(payload, strict=True)
    except ValidationError as exc:
        error = exc.errors(include_url=False)[0]
        location = tuple(error["loc"])
        error_path = _render_location(path, location)
        message = str(error["msg"])
        if error["type"] == "unexpected_keyword_argument" and location:
            parent = _annotation_at(annotation, location)
            if isinstance(parent, type) and is_dataclass(parent):
                names = tuple(field.name for field in fields(parent))
                unknown = str(location[-1])
                suggestion = get_close_matches(unknown, names, n=1, cutoff=0.55)
                if suggestion:
                    message = f"unknown field; did you mean {suggestion[0]!r}?"
                else:
                    message = "unknown field"
        raise ConfigDecodeError(error_path, message) from exc
    except (TypeError, ValueError) as exc:
        raise ConfigDecodeError(path, str(exc)) from exc


def from_dict(cls: type[T], data: dict[str, Any], *, path: str = "config") -> T:
    """Decode *data* through a cached strict Pydantic dataclass adapter."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    return cast(T, _decode(data, cls, path))


def to_dict(value: Any) -> Any:
    """Convert dataclasses/enums/tuples to stable JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_dict(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(to_dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def semantic_hash(value: Any) -> str:
    """Return a stable SHA-256 identity for a canonical JSON value."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def config_hash(config: RunConfig) -> str:
    payload = to_dict(config)
    if not isinstance(payload, dict):
        raise TypeError("run config did not encode as an object")
    distillation = payload.get("distillation")
    if isinstance(distillation, dict) and distillation.get("loss") == "top_k":
        if distillation.get("maximum_batches_per_epoch") is None:
            distillation.pop("maximum_batches_per_epoch", None)
        # This coefficient is semantically inactive for the conditional-only
        # objective, independent of the configured numeric value.
        distillation.pop("tail_mass_weight", None)
    return semantic_hash(payload)


def load_config(path: str | Path) -> RunConfig:
    source = Path(path)
    if source.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ConfigDecodeError("config", f"unsupported recipe extension {source.suffix!r}")
    if not isinstance(raw, dict):
        raise ConfigDecodeError("config", "recipe root must be an object")
    return from_dict(RunConfig, raw)


def _apply_path(instance: Any, parts: list[str], value: Any, full_path: str) -> Any:
    if not is_dataclass(instance):
        raise ConfigDecodeError(full_path, "path traverses a scalar value")
    field_map = {field.name: field for field in fields(instance)}
    head = parts[0]
    if head not in field_map:
        suggestion = get_close_matches(head, field_map, n=1, cutoff=0.55)
        suffix = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ConfigDecodeError(full_path, f"unknown field {head!r}{suffix}")
    if len(parts) > 1:
        child = _apply_path(getattr(instance, head), parts[1:], value, full_path)
    else:
        annotation = get_type_hints(type(instance))[head]
        child = _decode(value, annotation, full_path)
    return replace(cast(Any, instance), **{head: child})


def apply_overrides(config: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    """Apply sparse dotted-path overrides using schema types and no CLI defaults."""
    result = config
    for path in sorted(overrides):
        if not path or any(not part for part in path.split(".")):
            raise ConfigDecodeError(path or "config", "invalid dotted path")
        result = _apply_path(result, path.split("."), overrides[path], path)
    return result


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ConfigDecodeError(text, "override must use PATH=VALUE")
    path, raw = text.split("=", 1)
    return path.strip(), yaml.safe_load(raw)
