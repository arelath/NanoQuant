"""Immutable task-input and evaluator-result cache persistence."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

from nanoquant.application.evaluation_cache import (
    EvaluationResultCacheIdentity,
    TaskInputCacheIdentity,
)
from nanoquant.config.codec import canonical_json, from_dict, to_dict
from nanoquant.domain.models import ArtifactRef, ArtifactTypes
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.io_utils import atomic_write_json

T = TypeVar("T")
Identity = TaskInputCacheIdentity | EvaluationResultCacheIdentity
CacheKind = Literal["task-input", "evaluation-result"]


@dataclass(frozen=True, slots=True)
class EvaluationCacheIndex:
    schema_version: int
    task_inputs: tuple[tuple[str, ArtifactRef], ...]
    evaluation_results: tuple[tuple[str, ArtifactRef], ...]


@dataclass(frozen=True, slots=True)
class CommittedEvaluationCacheEntry:
    reference: ArtifactRef
    identity: Identity
    semantic_key: str
    payload: object


@dataclass(frozen=True, slots=True)
class EvaluationCacheLookup:
    status: Literal["hit", "miss"]
    semantic_key: str
    reason: str
    entry: CommittedEvaluationCacheEntry | None


@dataclass(frozen=True, slots=True)
class CachedEvaluationRun(Generic[T]):
    result: T
    cache_hit: bool
    reference: ArtifactRef


class EvaluationCache:
    """A run-local index over immutable, content-addressed evaluation artifacts."""

    INDEX_NAME = "evaluation-cache.json"
    LOCK_NAME = ".evaluation-cache.lock"

    def __init__(self, run_output: str | Path, artifacts: LocalArtifactStore) -> None:
        self.run_output = Path(run_output)
        self.artifacts = artifacts
        self.run_output.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.run_output / self.INDEX_NAME

    def _index(self) -> EvaluationCacheIndex:
        if not self.index_path.exists():
            return EvaluationCacheIndex(1, (), ())
        index = from_dict(
            EvaluationCacheIndex,
            json.loads(self.index_path.read_text(encoding="utf-8")),
            path="evaluation_cache",
        )
        if index.schema_version != 1:
            raise ValueError("unsupported evaluation cache index schema")
        for label, values in (
            ("task-input", index.task_inputs),
            ("evaluation-result", index.evaluation_results),
        ):
            keys = [key for key, _reference in values]
            if keys != sorted(keys) or len(keys) != len(set(keys)):
                raise ValueError(f"evaluation cache {label} index is not sorted and unique")
        return index

    def _write_index(self, index: EvaluationCacheIndex) -> None:
        atomic_write_json(self.index_path, to_dict(index))

    @contextmanager
    def _index_lock(self) -> Iterator[None]:
        lock_path = self.run_output / self.LOCK_NAME
        deadline = time.monotonic() + 30.0
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    stale = time.time() - lock_path.stat().st_mtime > 300.0
                except FileNotFoundError:
                    continue
                if stale:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for the evaluation cache index lock") from None
                time.sleep(0.01)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.fsync(descriptor)
            yield
        finally:
            os.close(descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _mapping(index: EvaluationCacheIndex, kind: CacheKind) -> dict[str, ArtifactRef]:
        values = index.task_inputs if kind == "task-input" else index.evaluation_results
        return dict(values)

    @staticmethod
    def _artifact_type(kind: CacheKind) -> str:
        return ArtifactTypes.EVALUATION_TASK_INPUTS if kind == "task-input" else ArtifactTypes.EVALUATION_RESULT

    @staticmethod
    def _identity_type(kind: CacheKind) -> type[TaskInputCacheIdentity] | type[EvaluationResultCacheIdentity]:
        return TaskInputCacheIdentity if kind == "task-input" else EvaluationResultCacheIdentity

    def _load(self, reference: ArtifactRef, expected: Identity, kind: CacheKind) -> CommittedEvaluationCacheEntry:
        descriptor = self.artifacts.validate(reference.artifact_id)
        expected_type = self._artifact_type(kind)
        if (
            reference.artifact_type != expected_type
            or descriptor.artifact_type != expected_type
            or reference.schema_version != descriptor.schema_version
        ):
            raise ValueError(f"cache artifact is not an {expected_type}")
        root = self.artifacts.path_for(reference.artifact_id)
        payload = json.loads((root / "entry.json").read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported cached evaluation artifact schema")
        identity = from_dict(self._identity_type(kind), payload["identity"], path=f"{kind}.identity")
        semantic_key = str(payload["semantic_key"])
        if identity != expected or semantic_key != expected.semantic_key:
            raise ValueError("cached evaluation artifact identity does not match its index key")
        return CommittedEvaluationCacheEntry(reference, identity, semantic_key, payload["payload"])

    def _lookup(self, identity: Identity, kind: CacheKind) -> EvaluationCacheLookup:
        semantic_key = identity.semantic_key
        reference = self._mapping(self._index(), kind).get(semantic_key)
        if reference is None:
            return EvaluationCacheLookup(
                "miss",
                semantic_key,
                f"no {kind} artifact matches the complete semantic identity",
                None,
            )
        return EvaluationCacheLookup(
            "hit",
            semantic_key,
            f"reused exact {kind} semantic identity",
            self._load(reference, identity, kind),
        )

    def lookup_task_inputs(self, identity: TaskInputCacheIdentity) -> EvaluationCacheLookup:
        return self._lookup(identity, "task-input")

    def lookup_evaluation_result(self, identity: EvaluationResultCacheIdentity) -> EvaluationCacheLookup:
        return self._lookup(identity, "evaluation-result")

    def _publish(self, kind: CacheKind, semantic_key: str, reference: ArtifactRef) -> bool:
        with self._index_lock():
            index = self._index()
            mapping = self._mapping(index, kind)
            existing = mapping.get(semantic_key)
            if existing is not None:
                if existing != reference:
                    raise ValueError(
                        f"conflicting {kind} artifacts share one semantic identity: "
                        f"{existing.artifact_id} != {reference.artifact_id}"
                    )
                return False
            mapping[semantic_key] = reference
            ordered = tuple(sorted(mapping.items()))
            updated = (
                EvaluationCacheIndex(index.schema_version, ordered, index.evaluation_results)
                if kind == "task-input"
                else EvaluationCacheIndex(index.schema_version, index.task_inputs, ordered)
            )
            self._write_index(updated)
            return True

    def _commit(self, identity: Identity, payload: object, kind: CacheKind) -> CommittedEvaluationCacheEntry:
        # Canonicalization both validates JSON safety and ensures dictionary key
        # order cannot create a second immutable artifact for the same payload.
        encoded = canonical_json(
            {
                "schema_version": 1,
                "identity": identity,
                "semantic_key": identity.semantic_key,
                "payload": payload,
            }
        )
        artifact_type = self._artifact_type(kind)
        with self.artifacts.begin_write(artifact_type) as writer:
            (writer.path / "entry.json").write_text(encoded + "\n", encoding="utf-8")
            descriptor = writer.commit()
        reference = ArtifactRef(artifact_type, descriptor.artifact_id, descriptor.schema_version)
        self._publish(kind, identity.semantic_key, reference)
        return self._load(reference, identity, kind)

    def commit_task_inputs(
        self,
        identity: TaskInputCacheIdentity,
        payload: object,
    ) -> CommittedEvaluationCacheEntry:
        return self._commit(identity, payload, "task-input")

    def commit_evaluation_result(
        self,
        identity: EvaluationResultCacheIdentity,
        payload: object,
    ) -> CommittedEvaluationCacheEntry:
        return self._commit(identity, payload, "evaluation-result")


def evaluate_with_cache(
    cache: EvaluationCache,
    identity: EvaluationResultCacheIdentity,
    evaluate: Callable[[], T],
    decode: Callable[[object], T],
) -> CachedEvaluationRun[T]:
    lookup = cache.lookup_evaluation_result(identity)
    if lookup.entry is not None:
        return CachedEvaluationRun(decode(lookup.entry.payload), True, lookup.entry.reference)
    result = evaluate()
    committed = cache.commit_evaluation_result(identity, to_dict(result))
    return CachedEvaluationRun(result, False, committed.reference)
