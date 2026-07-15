"""Numbered-experiment composition for shared quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.runs import launcher_provenance, validate_launcher_number
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation


@dataclass(frozen=True, slots=True)
class QualityEvaluationExperiment:
    request: QualityEvaluationRequest
    result_path: Path
    resolve_model_from_config: bool = False


def _resolve(path: Path, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def resolve_quality_evaluation_experiment(
    config: RunConfig,
    experiment: QualityEvaluationExperiment,
    *,
    launcher_path: str | Path,
) -> QualityEvaluationExperiment:
    launcher = Path(launcher_path).resolve()
    repository_root = launcher.parent.parent
    request = experiment.request
    if experiment.resolve_model_from_config:
        configured = Path(config.model.source)
        snapshot = (
            configured.resolve()
            if configured.exists()
            else Path(
                snapshot_download(
                    repo_id=config.model.source,
                    revision=str(config.model.revision),
                )
            ).resolve()
        )
    else:
        snapshot = _resolve(request.snapshot, repository_root)
    return QualityEvaluationExperiment(
        replace(
            request,
            snapshot=snapshot,
            source=config.model.source,
            revision=str(config.model.revision),
            run_output=_resolve(request.run_output, repository_root),
        ),
        _resolve(experiment.result_path, repository_root),
        experiment.resolve_model_from_config,
    )


def execute_quality_evaluation_experiment(
    config: RunConfig,
    experiment: QualityEvaluationExperiment,
    *,
    launcher_path: str | Path,
) -> dict[str, Any]:
    raise_for_issues(validate(config, ValidationPhase.RESOLVED))
    validate_launcher_number(config, launcher_path)
    resolved = resolve_quality_evaluation_experiment(
        config,
        experiment,
        launcher_path=launcher_path,
    )
    result = execute_quality_evaluation(resolved.request)
    payload = {
        **result,
        "experiment": {
            "config_hash": config_hash(config),
            "resolved_config": to_dict(config),
            "launcher": to_dict(
                launcher_provenance(launcher_path, config.intent.experiment_number)
            ),
        },
    }
    atomic_write_json(resolved.result_path, payload)
    return payload


def run_quality_evaluation_experiment(
    config: RunConfig,
    experiment: QualityEvaluationExperiment,
    *,
    launcher_path: str | Path,
) -> int:
    execute_quality_evaluation_experiment(config, experiment, launcher_path=launcher_path)
    return 0


__all__ = [
    "QualityEvaluationExperiment",
    "execute_quality_evaluation_experiment",
    "resolve_quality_evaluation_experiment",
    "run_quality_evaluation_experiment",
]
