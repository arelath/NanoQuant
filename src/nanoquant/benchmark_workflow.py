"""Numbered-experiment composition for the shared packed-runtime benchmark."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)
from nanoquant.infrastructure.runs import launcher_provenance, validate_launcher_number
from nanoquant.runtime_benchmark import RuntimeBenchmarkRequest, run_runtime_benchmark


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkExperiment:
    request: RuntimeBenchmarkRequest
    result_path: Path
    resolve_model_from_config: bool = False


def _resolve(path: Path, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def resolve_runtime_benchmark_experiment(
    config: RunConfig,
    experiment: RuntimeBenchmarkExperiment,
    *,
    launcher_path: str | Path,
) -> RuntimeBenchmarkExperiment:
    launcher = Path(launcher_path).resolve()
    repository_root = launcher.parent.parent
    request = experiment.request
    if experiment.resolve_model_from_config:
        configured_model = Path(config.model.source)
        if configured_model.exists():
            model = configured_model.resolve()
        else:
            model = Path(
                snapshot_download(
                    repo_id=config.model.source,
                    revision=str(config.model.revision),
                )
            ).resolve()
    else:
        model = _resolve(request.model, repository_root)
    return RuntimeBenchmarkExperiment(
        replace(
            request,
            packed_artifact=_resolve(request.packed_artifact, repository_root),
            model=model,
            run_output=(
                None if request.run_output is None else _resolve(request.run_output, repository_root)
            ),
        ),
        _resolve(experiment.result_path, repository_root),
        experiment.resolve_model_from_config,
    )


def execute_runtime_benchmark_experiment(
    config: RunConfig,
    experiment: RuntimeBenchmarkExperiment,
    *,
    launcher_path: str | Path,
) -> dict[str, Any]:
    """Run a pinned benchmark and atomically publish result plus launcher identity."""

    raise_for_issues(validate(config, ValidationPhase.RESOLVED))
    validate_launcher_number(config, launcher_path)
    resolved = resolve_runtime_benchmark_experiment(
        config,
        experiment,
        launcher_path=launcher_path,
    )
    result = run_runtime_benchmark(resolved.request)
    experiment_number = config.intent.experiment_number
    if experiment_number is None:
        raise ValueError("runtime benchmark requires an experiment number")
    repository_root = Path(launcher_path).resolve().parent.parent
    publication_directory = repository_root / "Results" / f"{experiment_number:03d}"
    payload = {
        **result,
        "experiment": {
            "config_hash": config_hash(config),
            "resolved_config": to_dict(config),
            "launcher": to_dict(
                launcher_provenance(launcher_path, config.intent.experiment_number)
            ),
        },
        "publication": {
            "directory": str(publication_directory),
            "manifest": str(publication_directory / "publication.json"),
        },
    }
    atomic_write_json(resolved.result_path, payload)
    publish_experiment_artifacts(
        repository_root,
        experiment_number,
        (PublishableArtifact(resolved.result_path, PublishableArtifactKind.STATISTICS),),
    )
    return payload


def run_runtime_benchmark_experiment(
    config: RunConfig,
    experiment: RuntimeBenchmarkExperiment,
    *,
    launcher_path: str | Path,
) -> int:
    execute_runtime_benchmark_experiment(config, experiment, launcher_path=launcher_path)
    return 0


__all__ = [
    "RuntimeBenchmarkExperiment",
    "execute_runtime_benchmark_experiment",
    "resolve_runtime_benchmark_experiment",
    "run_runtime_benchmark_experiment",
]
