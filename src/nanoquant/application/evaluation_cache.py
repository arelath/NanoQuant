"""Complete semantic identities for evaluation input and result caching."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from nanoquant.application.evaluation import EvaluationPartition, EvaluatorSpec
from nanoquant.config.codec import canonical_json
from nanoquant.domain.models import ArtifactRef


def _semantic_hash(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_hash(value: str, field: str) -> None:
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{field} must be a sha256 semantic hash")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 semantic hash") from exc


def _validate_parameters(parameters: tuple[tuple[str, object], ...], field: str) -> None:
    names = [name for name, _value in parameters]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"{field} must have unique non-empty names")
    # Fail at identity construction, rather than after an expensive evaluation,
    # when a parameter is not stable JSON data or contains a non-finite number.
    canonical_json(tuple(sorted(parameters, key=lambda item: item[0])))


@dataclass(frozen=True, slots=True)
class TaskInputCacheIdentity:
    schema_version: int
    evaluator_key: str
    task_name: str
    task_revision: str
    dataset_name: str
    dataset_revision: str
    dataset_content_hash: str
    split: str
    partition_name: str
    partition_version: str
    partition_content_hash: str
    sample_item_hashes: tuple[str, ...]
    tokenizer_name: str
    tokenizer_revision: str
    tokenizer_content_hash: str
    tokenizer_parameters: tuple[tuple[str, object], ...]
    prompt_template_revision: str | None
    prompt_template_hash: str | None
    few_shot_count: int
    few_shot_item_hashes: tuple[str, ...]
    selection_seed: int
    preprocessing_version: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported task-input cache identity schema")
        for value, field in (
            (self.evaluator_key, "evaluator key"),
            (self.dataset_content_hash, "dataset content hash"),
            (self.partition_content_hash, "partition content hash"),
            (self.tokenizer_content_hash, "tokenizer content hash"),
        ):
            _require_hash(value, field)
        if self.prompt_template_hash is not None:
            _require_hash(self.prompt_template_hash, "prompt template hash")
        required = (
            self.task_name,
            self.task_revision,
            self.dataset_name,
            self.dataset_revision,
            self.split,
            self.partition_name,
            self.partition_version,
            self.tokenizer_name,
            self.tokenizer_revision,
            self.preprocessing_version,
        )
        if any(not value for value in required):
            raise ValueError("task-input cache identity text fields must be non-empty")
        if not self.sample_item_hashes:
            raise ValueError("task-input cache identity requires ordered sample hashes")
        for value in self.sample_item_hashes:
            _require_hash(value, "sample item hash")
        if len(self.sample_item_hashes) != len(set(self.sample_item_hashes)):
            raise ValueError("task-input cache identity sample hashes must be unique")
        expected_partition_hash = _semantic_hash(self.sample_item_hashes)
        if self.partition_content_hash != expected_partition_hash:
            raise ValueError("partition content hash does not match the ordered sample hashes")
        if type(self.few_shot_count) is not int or self.few_shot_count < 0:
            raise ValueError("few-shot count must be a non-negative integer")
        if len(self.few_shot_item_hashes) != self.few_shot_count:
            raise ValueError("few-shot count must match the ordered few-shot item hashes")
        for value in self.few_shot_item_hashes:
            _require_hash(value, "few-shot item hash")
        if len(self.few_shot_item_hashes) != len(set(self.few_shot_item_hashes)):
            raise ValueError("few-shot item hashes must be unique")
        if (self.prompt_template_revision is None) != (self.prompt_template_hash is None):
            raise ValueError("prompt template revision and hash must either both be set or both be absent")
        if self.prompt_template_revision == "":
            raise ValueError("prompt template revision must be non-empty when provided")
        if type(self.selection_seed) is not int:
            raise ValueError("selection seed must be an integer")
        _validate_parameters(self.tokenizer_parameters, "tokenizer parameters")

    @classmethod
    def build(
        cls,
        specification: EvaluatorSpec,
        partition: EvaluationPartition,
        *,
        task_name: str,
        task_revision: str,
        dataset_name: str,
        dataset_revision: str,
        dataset_content_hash: str,
        split: str,
        tokenizer_name: str,
        tokenizer_revision: str,
        tokenizer_content_hash: str,
        tokenizer_parameters: tuple[tuple[str, object], ...],
        prompt_template_revision: str | None,
        prompt_template_hash: str | None,
        few_shot_count: int,
        few_shot_item_hashes: tuple[str, ...],
        selection_seed: int,
        preprocessing_version: str,
    ) -> TaskInputCacheIdentity:
        return cls(
            1,
            specification.semantic_key,
            task_name,
            task_revision,
            dataset_name,
            dataset_revision,
            dataset_content_hash,
            split,
            partition.name,
            partition.version,
            partition.content_hash,
            partition.item_hashes,
            tokenizer_name,
            tokenizer_revision,
            tokenizer_content_hash,
            tokenizer_parameters,
            prompt_template_revision,
            prompt_template_hash,
            few_shot_count,
            few_shot_item_hashes,
            selection_seed,
            preprocessing_version,
        )

    @property
    def semantic_key(self) -> str:
        values = {
            "schema_version": self.schema_version,
            "evaluator_key": self.evaluator_key,
            "task_name": self.task_name,
            "task_revision": self.task_revision,
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "dataset_content_hash": self.dataset_content_hash,
            "split": self.split,
            "partition_name": self.partition_name,
            "partition_version": self.partition_version,
            "partition_content_hash": self.partition_content_hash,
            "sample_item_hashes": self.sample_item_hashes,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_content_hash": self.tokenizer_content_hash,
            "tokenizer_parameters": tuple(sorted(self.tokenizer_parameters, key=lambda item: item[0])),
            "prompt_template_revision": self.prompt_template_revision,
            "prompt_template_hash": self.prompt_template_hash,
            "few_shot_count": self.few_shot_count,
            "few_shot_item_hashes": self.few_shot_item_hashes,
            "selection_seed": self.selection_seed,
            "preprocessing_version": self.preprocessing_version,
        }
        return _semantic_hash(values)


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeIdentity:
    schema_version: int
    backend_name: str
    backend_version: str
    mode: str
    parameters: tuple[tuple[str, object], ...] = ()
    environment_hash: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported evaluation runtime identity schema")
        if not self.backend_name or not self.backend_version or not self.mode:
            raise ValueError("evaluation runtime identity fields must be non-empty")
        if self.environment_hash is not None:
            _require_hash(self.environment_hash, "evaluation environment hash")
        _validate_parameters(self.parameters, "runtime parameters")

    @property
    def semantic_key(self) -> str:
        return _semantic_hash(
            (
                self.schema_version,
                self.backend_name,
                self.backend_version,
                self.mode,
                tuple(sorted(self.parameters, key=lambda item: item[0])),
                self.environment_hash,
            )
        )


@dataclass(frozen=True, slots=True)
class EvaluationResultCacheIdentity:
    schema_version: int
    model_artifact: ArtifactRef
    evaluator_key: str
    task_input_key: str
    runtime_key: str
    seed: int

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported evaluation-result cache identity schema")
        artifact_id = self.model_artifact.artifact_id
        valid_artifact_id = artifact_id.startswith("sha256-") and len(artifact_id) == 71
        if valid_artifact_id:
            try:
                int(artifact_id[7:], 16)
            except ValueError:
                valid_artifact_id = False
        if not self.model_artifact.artifact_type or not valid_artifact_id or self.model_artifact.schema_version <= 0:
            raise ValueError("evaluation-result cache identity requires a complete model artifact")
        for value, field in (
            (self.evaluator_key, "evaluator key"),
            (self.task_input_key, "task-input key"),
            (self.runtime_key, "runtime key"),
        ):
            _require_hash(value, field)
        if type(self.seed) is not int:
            raise ValueError("evaluation seed must be an integer")

    @classmethod
    def build(
        cls,
        model_artifact: ArtifactRef,
        specification: EvaluatorSpec,
        task_inputs: TaskInputCacheIdentity,
        runtime: EvaluationRuntimeIdentity,
        *,
        seed: int,
    ) -> EvaluationResultCacheIdentity:
        if task_inputs.evaluator_key != specification.semantic_key:
            raise ValueError("task inputs were prepared for a different evaluator")
        return cls(
            1,
            model_artifact,
            specification.semantic_key,
            task_inputs.semantic_key,
            runtime.semantic_key,
            seed,
        )

    @property
    def semantic_key(self) -> str:
        return _semantic_hash(self)
