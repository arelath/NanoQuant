"""Pure-data codecs for standardized experiment definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml

from nanoquant.compression_benchmark_workflow import CompressionBenchmarkExperiment
from nanoquant.compression_quality_workflow import CompressionQualityExperiment
from nanoquant.config.codec import ConfigDecodeError, from_dict, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.quality_evaluation_workflow import QualityEvaluationExperiment
from nanoquant.rank_expansion_experiment import RankExpansionExperiment

from ._experiment import (
    ExperimentDefinition,
    ExperimentIdentity,
    ExperimentLayout,
    ExperimentWorkflow,
)

DECLARATIVE_EXPERIMENT_SCHEMA_VERSION = 1


class ExperimentWorkflowKind(str, Enum):
    COMPRESSION_BENCHMARK = "compression_benchmark"
    COMPRESSION_QUALITY = "compression_quality"
    QUALITY_EVALUATION = "quality_evaluation"
    RANK_EXPANSION = "rank_expansion"


_WORKFLOW_TYPES: dict[ExperimentWorkflowKind, type[ExperimentWorkflow]] = {
    ExperimentWorkflowKind.COMPRESSION_BENCHMARK: CompressionBenchmarkExperiment,
    ExperimentWorkflowKind.COMPRESSION_QUALITY: CompressionQualityExperiment,
    ExperimentWorkflowKind.QUALITY_EVALUATION: QualityEvaluationExperiment,
    ExperimentWorkflowKind.RANK_EXPANSION: RankExpansionExperiment,
}


@dataclass(frozen=True, slots=True)
class DeclarativeExperiment:
    schema_version: int
    workflow_kind: ExperimentWorkflowKind
    identity: ExperimentIdentity
    config: RunConfig
    workflow: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != DECLARATIVE_EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported declarative experiment schema: {self.schema_version}"
            )

    def materialize(self) -> ExperimentDefinition[ExperimentWorkflow]:
        workflow_type = _WORKFLOW_TYPES[self.workflow_kind]
        workflow = from_dict(workflow_type, self.workflow, path="experiment.workflow")
        layout = ExperimentLayout(self.identity)
        return ExperimentDefinition(self.identity, self.config, workflow, layout)


def experiment_to_dict(
    definition: ExperimentDefinition[ExperimentWorkflow],
) -> dict[str, Any]:
    workflow = definition.workflow
    kind = next(
        (
            candidate
            for candidate, workflow_type in _WORKFLOW_TYPES.items()
            if isinstance(workflow, workflow_type)
        ),
        None,
    )
    if kind is None:
        raise TypeError(f"unsupported experiment workflow: {type(workflow).__name__}")
    return cast(
        dict[str, Any],
        to_dict(
            DeclarativeExperiment(
                DECLARATIVE_EXPERIMENT_SCHEMA_VERSION,
                kind,
                definition.identity,
                definition.config,
                cast(dict[str, Any], to_dict(workflow)),
            )
        ),
    )


def load_declarative_experiment(
    path: str | Path,
) -> ExperimentDefinition[ExperimentWorkflow]:
    source = Path(path)
    if source.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ConfigDecodeError(
            "experiment", f"unsupported definition extension {source.suffix!r}"
        )
    if not isinstance(payload, dict):
        raise ConfigDecodeError("experiment", "definition root must be an object")
    envelope = from_dict(DeclarativeExperiment, payload, path="experiment")
    definition = envelope.materialize()
    if source.stem != definition.identity.canonical_name:
        raise ConfigDecodeError(
            "experiment.identity",
            "definition filename must match the canonical experiment name",
        )
    return definition


__all__ = [
    "DECLARATIVE_EXPERIMENT_SCHEMA_VERSION",
    "DeclarativeExperiment",
    "ExperimentWorkflowKind",
    "experiment_to_dict",
    "load_declarative_experiment",
]
