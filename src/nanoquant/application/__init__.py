"""Application services orchestrate domain components through ports."""

from typing import Any

from nanoquant.config.schema import RunConfig


def run_experiment(config: RunConfig, *, launcher_path: str, **kwargs: Any) -> int:
    """Run the callback-based compatibility composition root.

    New quantization experiments should use :func:`run_quantization_experiment`.
    This entry point remains for foundation adapters that explicitly supply a
    pipeline callback.
    """
    from nanoquant.bootstrap import run_experiment as composed_run_experiment

    return composed_run_experiment(config, launcher_path=launcher_path, **kwargs)


def run_quantization_experiment(config: RunConfig, *, launcher_path: str) -> int:
    """Run one new experiment through the canonical resident workflow."""

    from nanoquant.resident_workflow import run_resident_experiment

    return run_resident_experiment(config, launcher_path=launcher_path)


__all__ = ["run_experiment", "run_quantization_experiment"]
