"""Strict schema-aware configuration decoding and canonical serialization."""

from __future__ import annotations

import hashlib
import json
import types
from dataclasses import MISSING, fields, is_dataclass, replace
from difflib import get_close_matches
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml

from .schema import RunConfig

T = TypeVar("T")


class ConfigDecodeError(ValueError):
    """Configuration is invalid at a precise dotted path."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


def _decode(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors: list[str] = []
        for candidate in (arg for arg in args if arg is not type(None)):
            try:
                return _decode(value, candidate, path)
            except (ConfigDecodeError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise ConfigDecodeError(path, f"does not match any allowed type ({'; '.join(errors)})")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigDecodeError(path, "expected an array")
        element_type = args[0] if args else Any
        if len(args) > 1 and args[1] is not Ellipsis and len(args) != len(value):
            raise ConfigDecodeError(path, f"expected {len(args)} elements")
        return tuple(
            _decode(item, element_type if len(args) == 2 else args[index], f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError as exc:
            allowed = ", ".join(repr(member.value) for member in annotation)
            raise ConfigDecodeError(path, f"expected one of {allowed}; got {value!r}") from exc
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigDecodeError(path, "expected an object")
        return from_dict(annotation, value, path=path)
    if annotation is Any:
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ConfigDecodeError(path, "expected a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigDecodeError(path, "expected an integer")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise ConfigDecodeError(path, "expected a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigDecodeError(path, "expected a string")
        return value
    return value


def from_dict(cls: type[T], data: dict[str, Any], *, path: str = "config") -> T:
    """Decode *data* to a dataclass, rejecting every unknown field."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    field_map = {field.name: field for field in fields(cls)}
    unknown = sorted(set(data) - set(field_map))
    if unknown:
        name = unknown[0]
        suggestion = get_close_matches(name, field_map, n=1, cutoff=0.55)
        suffix = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ConfigDecodeError(f"{path}.{name}", f"unknown field{suffix}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, field in field_map.items():
        if name in data:
            kwargs[name] = _decode(data[name], hints[name], f"{path}.{name}")
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ConfigDecodeError(f"{path}.{name}", "required field is missing")
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigDecodeError(path, str(exc)) from exc


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


def config_hash(config: RunConfig) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


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
