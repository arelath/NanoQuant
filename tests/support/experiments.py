"""Load concrete numbered experiment definitions from their colocated launchers."""

from __future__ import annotations

import runpy
from functools import cache
from pathlib import Path
from typing import Any, cast

from recipes import ExperimentDefinition

from nanoquant.config.codec import to_dict


@cache
def load_experiment(number: int) -> ExperimentDefinition[Any]:
    matches = tuple(Path("experiments").glob(f"{number:03d}-*.py"))
    if len(matches) != 1:
        raise ValueError(f"expected one launcher for Experiment {number:03d}, found {len(matches)}")
    namespace = runpy.run_path(str(matches[0]))
    return cast(ExperimentDefinition[Any], namespace["EXPERIMENT"])


def config_diff_paths(left: object, right: object, prefix: str = "") -> set[str]:
    left_value = to_dict(left)
    right_value = to_dict(right)
    if isinstance(left_value, dict) and isinstance(right_value, dict):
        paths = set()
        for key in left_value.keys() | right_value.keys():
            path = f"{prefix}.{key}" if prefix else key
            paths.update(config_diff_paths(left_value.get(key), right_value.get(key), path))
        return paths
    return set() if left_value == right_value else {prefix}


__all__ = ["config_diff_paths", "load_experiment"]
