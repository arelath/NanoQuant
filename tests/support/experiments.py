"""Load concrete numbered experiment definitions from their colocated launchers."""

from __future__ import annotations

import runpy
from functools import cache
from pathlib import Path
from typing import Any, cast

from recipes import ExperimentDefinition


@cache
def load_experiment(number: int) -> ExperimentDefinition[Any]:
    matches = tuple(Path("experiments").glob(f"{number:03d}-*.py"))
    if len(matches) != 1:
        raise ValueError(f"expected one launcher for Experiment {number:03d}, found {len(matches)}")
    namespace = runpy.run_path(str(matches[0]))
    return cast(ExperimentDefinition[Any], namespace["EXPERIMENT"])


__all__ = ["load_experiment"]
