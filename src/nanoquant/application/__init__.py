"""Application services orchestrate domain components through ports."""

from typing import Any

from nanoquant.config.schema import RunConfig


def run_experiment(config: RunConfig, *, launcher_path: str, **kwargs: Any) -> int:
    """Public convenience wrapper around the lazily loaded composition root."""
    from nanoquant.bootstrap import run_experiment as composed_run_experiment

    return composed_run_experiment(config, launcher_path=launcher_path, **kwargs)


__all__ = ["run_experiment"]
