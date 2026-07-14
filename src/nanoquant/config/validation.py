"""Stable, phased configuration validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from nanoquant.ports.event_sink import Severity

from .schema import ObjectiveKind, OutlierSelector, RunConfig


class ValidationPhase(str, Enum):
    PRE_RESOLUTION = "pre_resolution"
    RESOLVED = "resolved"
    PLANNED = "planned"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    severity: str = "error"


def validate(config: RunConfig, phase: ValidationPhase = ValidationPhase.PRE_RESOLUTION) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []

    def require(condition: bool, code: str, path: str, message: str) -> None:
        if not condition:
            issues.append(ValidationIssue(code, path, message))

    require(config.schema_version == 1, "CFG001", "schema_version", "only schema version 1 is supported")
    require(bool(config.model.source.strip()), "CFG002", "model.source", "model source must not be empty")
    require(config.model.sequence_length > 0, "CFG003", "model.sequence_length", "must be positive")
    require(config.calibration.sample_count >= 0, "CFG004", "calibration.sample_count", "must not be negative")
    require(0 <= config.calibration.shrinkage <= 1, "CFG005", "calibration.shrinkage", "must be in [0, 1]")
    require(config.allocation.target_bpw > 0, "CFG006", "allocation.target_bpw", "must be positive")
    require(config.allocation.bounds.multiple > 0, "CFG007", "allocation.bounds.multiple", "must be positive")
    require(
        config.allocation.bounds.floor_fraction_of_uniform <= config.allocation.bounds.ceiling_fraction_of_uniform,
        "CFG008",
        "allocation.bounds",
        "floor must not exceed ceiling",
    )
    require(0 <= config.outliers.fraction < 1, "CFG009", "outliers.fraction", "must be in [0, 1)")
    require(
        not (config.outliers.selector is OutlierSelector.NONE and config.outliers.fraction > 0),
        "CFG010",
        "outliers",
        "positive fraction requires an enabled selector",
    )
    require(
        config.runtime.block_forward_batch_size > 0, "CFG011", "runtime.block_forward_batch_size", "must be positive"
    )
    require(
        config.factorization.admm.outer_iterations > 0,
        "CFG012",
        "factorization.admm.outer_iterations",
        "must be positive",
    )
    require(config.profiling.cuda_sample_every > 0, "CFG015", "profiling.cuda_sample_every", "must be positive")
    require(
        config.profiling.raw_samples_per_phase > 0,
        "CFG016",
        "profiling.raw_samples_per_phase",
        "must be positive",
    )
    levels: dict[str, Severity] = {}
    for path, value in (
        ("observability.console_level", config.observability.console_level),
        ("observability.event_level", config.observability.event_level),
    ):
        try:
            levels[path] = Severity.parse(value)
        except ValueError:
            require(False, "OBS001", path, "must be one of debug, info, warning, error")
    console_level = levels.get("observability.console_level")
    event_level = levels.get("observability.event_level")
    if console_level is not None and event_level is not None:
        require(
            event_level.rank <= console_level.rank,
            "OBS002",
            "observability.event_level",
            "must be at least as verbose as observability.console_level",
        )
        require(
            not config.observability.record_admm_steps or event_level is Severity.DEBUG,
            "OBS003",
            "observability.record_admm_steps",
            "requires observability.event_level=debug",
        )
    resource_interval = config.observability.record_resource_interval_seconds
    require(
        math.isfinite(resource_interval),
        "OBS004",
        "observability.record_resource_interval_seconds",
        "must be finite; values at or below zero disable resource sampling",
    )
    if math.isfinite(resource_interval) and 0 < resource_interval < 1:
        issues.append(
            ValidationIssue(
                "OBS004",
                "observability.record_resource_interval_seconds",
                "intervals below one second may create excessive event volume",
                "warning",
            )
        )
    if config.calibration.objective.kind is ObjectiveKind.BLOCK_DIAGONAL:
        require(
            bool(config.calibration.objective.block_size and config.calibration.objective.block_size > 0),
            "CFG013",
            "calibration.objective.block_size",
            "is required for block-diagonal objectives",
        )
    if config.calibration.objective.kind is ObjectiveKind.LOW_RANK_DIAGONAL:
        require(
            bool(config.calibration.objective.low_rank and config.calibration.objective.low_rank > 0),
            "CFG014",
            "calibration.objective.low_rank",
            "is required for low-rank-diagonal objectives",
        )
    if phase in (ValidationPhase.RESOLVED, ValidationPhase.PLANNED):
        require(
            config.model.revision is not None, "RES001", "model.revision", "resolved config requires a pinned revision"
        )
        require(
            config.model.tokenizer_revision is not None,
            "RES002",
            "model.tokenizer_revision",
            "resolved config requires a pinned tokenizer revision",
        )
    return tuple(issues)


def raise_for_issues(issues: tuple[ValidationIssue, ...]) -> None:
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        rendered = "\n".join(f"{item.code} {item.path}: {item.message}" for item in errors)
        raise ValueError(rendered)
