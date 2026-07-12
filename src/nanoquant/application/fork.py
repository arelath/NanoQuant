"""Visible upstream-reuse/downstream-invalidation fork semantics."""

from __future__ import annotations

from dataclasses import dataclass

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import RunConfig


@dataclass(frozen=True, slots=True)
class StageReuseDecision:
    stage: str
    action: str
    reason: str


@dataclass(frozen=True, slots=True)
class ForkPlan:
    parent_run_id: str
    changed_paths: tuple[str, ...]
    fork_from_stage: str
    stages: tuple[StageReuseDecision, ...]


STAGES = ("resolve-source", "prepare-dataset", "calibrate", "plan", "quantize", "pack", "evaluate", "report")
OWNERS = {
    "model": "resolve-source",
    "dataset": "prepare-dataset",
    "calibration": "calibrate",
    "reproducibility": "calibrate",
    "allocation": "plan",
    "outliers": "plan",
    "factorization": "quantize",
    "block_tuning": "quantize",
    "runtime": "quantize",
    "distillation": "quantize",
    "packing": "pack",
    "evaluation": "evaluate",
    "intent": "report",
    "observability": "report",
    "output": "report",
}


def _diff(left: object, right: object, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_diff(left[key], right[key], child))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        result = []
        for index, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
            result.extend(_diff(lhs, rhs, f"{path}[{index}]"))
        return result
    return [] if left == right else [path]


def plan_fork(parent_run_id: str, parent: RunConfig, candidate: RunConfig) -> ForkPlan:
    changes = tuple(_diff(to_dict(parent), to_dict(candidate)))
    affected = []
    for path in changes:
        root = path.split(".", 1)[0].split("[", 1)[0]
        if root == "schema_version":
            affected.append("resolve-source")
        elif root in OWNERS:
            affected.append(OWNERS[root])
    earliest = min(affected, key=STAGES.index) if affected else "report"
    boundary = STAGES.index(earliest)
    decisions = tuple(
        StageReuseDecision(
            stage,
            "reuse" if index < boundary else "invalidate",
            "semantic inputs unchanged upstream" if index < boundary else f"invalidated by changes at {earliest}",
        )
        for index, stage in enumerate(STAGES)
    )
    return ForkPlan(parent_run_id, changes, earliest, decisions)
