"""Schema-derived reference data used by CLI help and documentation tooling."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints

from .codec import to_dict
from .schema import RunConfig


def schema_reference(cls: type[Any] = RunConfig, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hints = get_type_hints(cls)
    for field in fields(cls):
        path = f"{prefix}.{field.name}" if prefix else field.name
        annotation = hints[field.name]
        target = annotation
        origin = get_origin(annotation)
        if origin is not None:
            candidates = [arg for arg in get_args(annotation) if isinstance(arg, type) and is_dataclass(arg)]
            target = candidates[0] if candidates else annotation
        if isinstance(target, type) and is_dataclass(target):
            rows.extend(schema_reference(target, path))
            continue
        if field.default is not MISSING:
            default = to_dict(field.default)
        elif field.default_factory is not MISSING:
            default = to_dict(field.default_factory())
        else:
            default = None
        rows.append(
            {
                "path": path,
                "type": str(annotation),
                "default": default,
                "required": field.default is MISSING and field.default_factory is MISSING,
            }
        )
    return rows
