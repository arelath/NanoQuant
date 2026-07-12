"""Stage-specific semantic keys and human-visible invalidation explanations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from nanoquant.config.codec import to_dict


@dataclass(frozen=True, slots=True)
class SemanticKey:
    stage: str
    producer: str
    digest: str
    semantic_inputs: dict[str, object]


@dataclass(frozen=True, slots=True)
class CacheExplanation:
    reusable: bool
    reason: str
    changed_paths: tuple[str, ...] = ()


def semantic_key(stage: str, producer_name: str, producer_version: str, request: object) -> SemanticKey:
    values = to_dict(request)
    if not isinstance(values, dict):
        values = {"value": values}
    producer = f"{producer_name}@{producer_version}"
    identity = {"stage": stage, "producer": producer, "inputs": values}
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return SemanticKey(stage, producer, "sha256:" + hashlib.sha256(payload.encode()).hexdigest(), values)


def explain_reuse(expected: SemanticKey, candidate: SemanticKey) -> CacheExplanation:
    if expected.stage != candidate.stage:
        return CacheExplanation(False, f"stage changed: {candidate.stage} -> {expected.stage}", ("stage",))
    if expected.producer != candidate.producer:
        return CacheExplanation(False, f"producer changed: {candidate.producer} -> {expected.producer}", ("producer",))
    changed = tuple(_diff(expected.semantic_inputs, candidate.semantic_inputs))
    if changed:
        return CacheExplanation(False, "semantic inputs changed", changed)
    if expected.digest != candidate.digest:
        return CacheExplanation(False, "semantic digest mismatch despite equal decoded inputs")
    return CacheExplanation(True, "producer and all semantic inputs match")


def _diff(left: Any, right: Any, path: str = "request") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        changed: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                changed.append(child)
            else:
                changed.extend(_diff(left[key], right[key], child))
        return changed
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        changed = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            changed.extend(_diff(left_item, right_item, f"{path}[{index}]"))
        return changed
    return [] if left == right else [path]
