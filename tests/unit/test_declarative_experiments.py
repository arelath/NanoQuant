from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any, cast

import pytest
from recipes import experiment_to_dict, load_declarative_experiment
from recipes._experiment import ExperimentDefinition, ExperimentWorkflow

from nanoquant.config.codec import ConfigDecodeError


def _experiment_028() -> ExperimentDefinition[ExperimentWorkflow]:
    namespace = runpy.run_path(
        "experiments/028-compress-and-benchmark-qwen3-0-6b.py",
        run_name="declarative_experiment_028",
    )
    return cast(ExperimentDefinition[ExperimentWorkflow], namespace["EXPERIMENT"])


def test_declarative_experiment_round_trip_preserves_complete_definition(
    tmp_path: Path,
) -> None:
    expected = _experiment_028()
    path = tmp_path / f"{expected.identity.canonical_name}.json"
    path.write_text(json.dumps(experiment_to_dict(expected)), encoding="utf-8")

    assert load_declarative_experiment(path) == expected


def test_declarative_experiment_rejects_filename_identity_drift(tmp_path: Path) -> None:
    expected = _experiment_028()
    payload = cast(dict[str, Any], experiment_to_dict(expected))
    path = tmp_path / "029-wrong-name.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigDecodeError, match="filename"):
        load_declarative_experiment(path)
