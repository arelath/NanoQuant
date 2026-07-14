"""Journaled progress, orphan-commit discovery, and resume compatibility validation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nanoquant.config.codec import to_dict
from nanoquant.domain.models import ArtifactTypes, QuantizationPlan
from nanoquant.domain.runs import BudgetState, ProgressCursor, RunState, RunStatus
from nanoquant.infrastructure.io_utils import atomic_write_json

from .artifacts import ArtifactCorruptionError, LocalArtifactStore
from .commits import CommitIdentity


@dataclass(frozen=True, slots=True)
class JournalRecord:
    sequence: int
    kind: str
    block: int
    layer: str | None
    artifact_id: str
    identity: CommitIdentity
    timestamp: str


@dataclass(frozen=True, slots=True)
class ResumeDiscovery:
    valid_records: tuple[JournalRecord, ...]
    orphan_records: tuple[JournalRecord, ...]
    first_incomplete: ProgressCursor | None


class ProgressJournal:
    def __init__(self, directory: str | Path, run_id: str, artifacts: LocalArtifactStore) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.artifacts = artifacts
        self.path = self.directory / "journal.jsonl"
        self.state_path = self.directory / "run-state.json"

    def _records(self) -> list[JournalRecord]:
        if not self.path.exists():
            return []
        records: list[JournalRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
                raw["identity"] = CommitIdentity(**raw["identity"])
                record = JournalRecord(**raw)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                break
            if record.sequence != len(records) + 1:
                break
            records.append(record)
        return records

    def append(
        self, kind: str, block: int, layer: str | None, artifact_id: str, identity: CommitIdentity
    ) -> JournalRecord:
        self.artifacts.validate(artifact_id)
        sequence = len(self._records()) + 1
        record = JournalRecord(
            sequence, kind, block, layer, artifact_id, identity, datetime.now(timezone.utc).isoformat()
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())
        return record

    def write_state(self, state: RunState) -> None:
        atomic_write_json(self.state_path, to_dict(state))

    def _valid_journal_records(self, identity: CommitIdentity) -> list[JournalRecord]:
        valid = []
        for record in self._records():
            if record.identity != identity:
                break
            try:
                self.artifacts.validate(record.artifact_id)
            except (ArtifactCorruptionError, OSError, ValueError):
                break
            valid.append(record)
        return valid

    def _orphan_records(self, identity: CommitIdentity, known: set[str]) -> list[JournalRecord]:
        found = []
        for descriptor_path in self.artifacts.root.glob("??/sha256-*/descriptor.json"):
            try:
                descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
                artifact_id = descriptor["artifact_id"]
                if artifact_id in known or descriptor["artifact_type"] not in {
                    ArtifactTypes.LAYER_RESULT,
                    ArtifactTypes.BLOCK_RESULT,
                }:
                    continue
                self.artifacts.validate(artifact_id)
                filename = (
                    "layer-result.json"
                    if descriptor["artifact_type"] == ArtifactTypes.LAYER_RESULT
                    else "block-result.json"
                )
                payload = json.loads((descriptor_path.parent / filename).read_text(encoding="utf-8"))
                if CommitIdentity(**payload["identity"]) != identity:
                    continue
                result = payload["result"] if descriptor["artifact_type"] == ArtifactTypes.LAYER_RESULT else payload
                block = int(
                    result["layer"]["block"]["index"]
                    if descriptor["artifact_type"] == ArtifactTypes.LAYER_RESULT
                    else result["block"]["index"]
                )
                layer = result["layer"]["path"] if descriptor["artifact_type"] == ArtifactTypes.LAYER_RESULT else None
                found.append(
                    JournalRecord(
                        0,
                        "layer" if layer else "block",
                        block,
                        layer,
                        artifact_id,
                        identity,
                        descriptor_path.stat().st_mtime_ns.__str__(),
                    )
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, ArtifactCorruptionError):
                continue
        return sorted(found, key=lambda record: (record.block, record.layer is None, record.layer or ""))

    def discover(self, plan: QuantizationPlan, identity: CommitIdentity) -> ResumeDiscovery:
        valid = self._valid_journal_records(identity)
        orphans = self._orphan_records(identity, {record.artifact_id for record in valid})
        all_records = [*valid, *orphans]
        completed_blocks = {record.block for record in all_records if record.kind == "block"}
        completed_layers = {(record.block, record.layer) for record in all_records if record.kind == "layer"}
        first = None
        for block in plan.blocks:
            if block.block.index in completed_blocks:
                continue
            for layer in block.layers:
                if (block.block.index, layer.layer.path) not in completed_layers:
                    first = ProgressCursor("quantize-blocks", block.block.index, layer.layer.path, 0)
                    break
            if first is None:
                first = ProgressCursor("quantize-blocks", block.block.index, None, None)
            break
        return ResumeDiscovery(tuple(valid), tuple(orphans), first)

    def state_from_discovery(self, discovery: ResumeDiscovery, budget: BudgetState) -> RunState:
        records = (*discovery.valid_records, *discovery.orphan_records)
        return RunState(
            1,
            self.run_id,
            RunStatus.RUNNING,
            discovery.first_incomplete,
            budget,
            tuple(record.artifact_id for record in records),
            len(discovery.valid_records),
        )
