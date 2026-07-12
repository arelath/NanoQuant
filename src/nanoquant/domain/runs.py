"""Run identity, lifecycle, lineage, and provenance contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class LauncherProvenance:
    kind: str
    experiment_number: int | None
    repository_relative_path: str | None
    content_hash: str
    revision: str | None
    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema_version: int
    run_id: str
    status: RunStatus
    created_at: str
    updated_at: str
    config_hash: str
    resolved_config: dict[str, object]
    launcher: LauncherProvenance
    environment: dict[str, object]
    parent_run_id: str | None = None
    forked_from_stage: str | None = None
    artifacts: tuple[str, ...] = ()
    failure: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ProgressCursor:
    stage: str
    block: int | None = None
    layer: str | None = None
    attempt: int | None = None


@dataclass(frozen=True, slots=True)
class BudgetState:
    planned_bits: int
    accepted_bits: int
    retry_bits_spent: int


@dataclass(frozen=True, slots=True)
class RunState:
    schema_version: int
    run_id: str
    status: RunStatus
    cursor: ProgressCursor | None
    budget: BudgetState
    committed_artifacts: tuple[str, ...]
    journal_sequence: int
