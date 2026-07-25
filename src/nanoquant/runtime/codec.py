"""Dependency-light, type-directed decoding for runtime manifests."""

from __future__ import annotations

import types
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints

T = TypeVar("T")


class RuntimeDecodeError(ValueError):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path


def _decode(value: object, annotation: object, path: str) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if annotation is Any or annotation is object:
        return value
    if origin in (types.UnionType,):
        if type(None) in arguments and value is None:
            return None
        candidates = tuple(item for item in arguments if item is not type(None))
        errors: list[ValueError] = []
        for candidate in candidates:
            try:
                return _decode(value, candidate, path)
            except ValueError as error:
                errors.append(error)
        raise RuntimeDecodeError(path, "value does not match any allowed type") from errors[-1]
    if origin is tuple:
        if not isinstance(value, list):
            raise RuntimeDecodeError(path, "expected an array")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode(item, arguments[0], f"{path}[{index}]") for index, item in enumerate(value))
        if len(value) != len(arguments):
            raise RuntimeDecodeError(path, f"expected {len(arguments)} array items")
        return tuple(
            _decode(item, item_type, f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, arguments, strict=True))
        )
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise RuntimeDecodeError(path, "expected an object with string keys")
        payload = cast(dict[str, object], value)
        field_map = {field.name: field for field in fields(annotation)}
        unknown = sorted(set(payload) - set(field_map))
        if unknown:
            raise RuntimeDecodeError(path, f"unknown fields: {unknown}")
        hints = get_type_hints(annotation)
        decoded: dict[str, object] = {}
        for name, field in field_map.items():
            child_path = f"{path}.{name}"
            if name in payload:
                decoded[name] = _decode(payload[name], hints[name], child_path)
            elif field.default is MISSING and field.default_factory is MISSING:
                raise RuntimeDecodeError(child_path, "required field is missing")
        return annotation(**decoded)
    if annotation is bool:
        if not isinstance(value, bool):
            raise RuntimeDecodeError(path, "expected a boolean")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeDecodeError(path, "expected an integer")
        return value
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeDecodeError(path, "expected a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise RuntimeDecodeError(path, "expected a string")
        return value
    raise RuntimeDecodeError(path, f"unsupported runtime annotation: {annotation!r}")


def decode_dataclass(model_type: type[T], payload: object, *, path: str = "manifest") -> T:
    """Decode one JSON-compatible object into a runtime dataclass graph."""

    if not is_dataclass(model_type):
        raise TypeError("runtime manifest target must be a dataclass type")
    return cast(T, _decode(payload, model_type, path))


__all__ = ["RuntimeDecodeError", "decode_dataclass"]
