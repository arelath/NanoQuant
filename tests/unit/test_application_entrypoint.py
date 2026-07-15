from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

from nanoquant.application import run_quantization_experiment
from nanoquant.application.service import ApplicationContext, QuantizeApplication
from nanoquant.config.schema import ModelConfig, RunConfig


class _Events:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, str, str, dict[str, object]]] = []

    def emit(self, component: str, severity: str, name: str, **fields: object) -> None:
        self.emitted.append((component, severity, name, fields))


def test_compatibility_application_rejects_a_missing_pipeline() -> None:
    events = _Events()
    context = ApplicationContext(artifacts=Any, events=events)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="RUN002.*run_quantization_experiment"):
        QuantizeApplication().run(RunConfig(ModelConfig("fixture")), context)

    assert events.emitted[-1] == (
        "quantize",
        "error",
        "pipeline.not_configured",
        {"code": "RUN002"},
    )


def test_new_experiment_entrypoint_delegates_to_resident_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[RunConfig, str]] = []

    def resident(config: RunConfig, *, launcher_path: str) -> int:
        observed.append((config, launcher_path))
        return 17

    monkeypatch.setattr("nanoquant.resident_workflow.run_resident_experiment", resident)
    config = RunConfig(ModelConfig("fixture"))

    assert run_quantization_experiment(config, launcher_path="experiments/020-new.py") == 17
    assert observed == [(config, "experiments/020-new.py")]


def test_new_experiment_template_uses_the_real_quantization_entrypoint() -> None:
    namespace = runpy.run_path(str(Path("experiments/000_experiment_template.py")))

    assert namespace["run_quantization_experiment"] is run_quantization_experiment
    assert "run_experiment" not in namespace
