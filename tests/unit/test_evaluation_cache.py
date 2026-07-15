from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from nanoquant.application.evaluation import (
    CausalEvaluationResult,
    EvaluationPartition,
    EvaluatorSpec,
)
from nanoquant.application.evaluation_cache import (
    EvaluationResultCacheIdentity,
    EvaluationRuntimeIdentity,
    TaskInputCacheIdentity,
)
from nanoquant.config.codec import from_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.evaluation_cache import EvaluationCache, evaluate_with_cache


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _artifact(value: str) -> ArtifactRef:
    return ArtifactRef("packed-model", "sha256-" + hashlib.sha256(value.encode()).hexdigest(), 1)


def _identities() -> tuple[
    EvaluatorSpec,
    TaskInputCacheIdentity,
    EvaluationRuntimeIdentity,
    EvaluationResultCacheIdentity,
]:
    specification = EvaluatorSpec(
        "task-accuracy",
        "implementation-v3",
        "quick",
        (("normalization", "exact"), ("maximum_samples", 2)),
    )
    partition = EvaluationPartition.build("quick", "selection-v2", (("sample-a",), ("sample-b",)))
    task = TaskInputCacheIdentity.build(
        specification,
        partition,
        task_name="arc-easy",
        task_revision="task-v4",
        dataset_name="ai2_arc",
        dataset_revision="dataset-commit",
        dataset_content_hash=_hash("dataset"),
        split="validation",
        tokenizer_name="gemma-tokenizer",
        tokenizer_revision="dcc83ea8",
        tokenizer_content_hash=_hash("tokenizer"),
        tokenizer_parameters=(("add_bos", True), ("chat_template", "gemma")),
        prompt_template_revision="prompt-v5",
        prompt_template_hash=_hash("prompt"),
        few_shot_count=0,
        few_shot_item_hashes=(),
        selection_seed=17,
        preprocessing_version="preprocess-v2",
    )
    runtime = EvaluationRuntimeIdentity(
        1,
        "nanoquant-cuda",
        "packed-v1",
        "deterministic-logits",
        (("accumulation_dtype", "float32"), ("batch_size", 2)),
    )
    result = EvaluationResultCacheIdentity.build(
        _artifact("candidate"),
        specification,
        task,
        runtime,
        seed=23,
    )
    return specification, task, runtime, result


def test_complete_cache_identities_change_for_every_numerical_boundary() -> None:
    specification, task, runtime, result = _identities()
    reordered_spec = EvaluatorSpec(
        specification.name,
        specification.version,
        specification.tier,
        tuple(reversed(specification.parameters)),
    )
    reordered_task = replace(task, tokenizer_parameters=tuple(reversed(task.tokenizer_parameters)))
    reversed_partition = EvaluationPartition.build(
        "quick",
        "selection-v2",
        (("sample-b",), ("sample-a",)),
    )

    assert reordered_spec.semantic_key == specification.semantic_key
    assert reordered_task.semantic_key == task.semantic_key
    assert replace(task, task_revision="task-v5").semantic_key != task.semantic_key
    assert replace(task, dataset_content_hash=_hash("changed-dataset")).semantic_key != task.semantic_key
    assert replace(task, partition_version="selection-v3").semantic_key != task.semantic_key
    changed_selection = replace(
        task,
        partition_content_hash=reversed_partition.content_hash,
        sample_item_hashes=reversed_partition.item_hashes,
    )
    assert changed_selection.semantic_key != task.semantic_key
    assert replace(task, tokenizer_content_hash=_hash("changed-tokenizer")).semantic_key != task.semantic_key
    assert replace(task, tokenizer_parameters=(("add_bos", False),)).semantic_key != task.semantic_key
    assert replace(task, prompt_template_revision="prompt-v6").semantic_key != task.semantic_key
    assert replace(task, prompt_template_hash=_hash("changed-prompt")).semantic_key != task.semantic_key
    few_shot = replace(task, few_shot_count=1, few_shot_item_hashes=(_hash("demonstration"),))
    assert few_shot.semantic_key != task.semantic_key
    assert replace(task, selection_seed=18).semantic_key != task.semantic_key
    assert replace(task, preprocessing_version="preprocess-v3").semantic_key != task.semantic_key
    assert replace(runtime, mode="fast-math").semantic_key != runtime.semantic_key
    assert replace(runtime, parameters=(("batch_size", 1),)).semantic_key != runtime.semantic_key
    assert replace(runtime, environment_hash=_hash("other-host")).semantic_key != runtime.semantic_key
    assert replace(result, model_artifact=_artifact("other-model")).semantic_key != result.semantic_key
    fast_runtime_result = replace(result, runtime_key=replace(runtime, mode="fast-math").semantic_key)
    assert fast_runtime_result.semantic_key != result.semantic_key
    assert replace(result, seed=24).semantic_key != result.semantic_key


def test_result_identity_rejects_task_inputs_from_another_evaluator() -> None:
    specification, task, runtime, _result = _identities()
    changed = EvaluatorSpec(specification.name, "implementation-v4", specification.tier)

    with pytest.raises(ValueError, match="different evaluator"):
        EvaluationResultCacheIdentity.build(_artifact("candidate"), changed, task, runtime, seed=0)


def test_task_inputs_and_model_results_are_cached_independently_and_durably(tmp_path: Path) -> None:
    _specification, task, _runtime, result = _identities()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    cache = EvaluationCache(tmp_path / "run", artifacts)

    task_entry = cache.commit_task_inputs(task, {"token_ids": [[1, 2], [3, 4]], "labels": [0, 1]})
    result_entry = cache.commit_evaluation_result(result, {"accuracy": 0.5, "sample_count": 2})

    reopened = EvaluationCache(tmp_path / "run", LocalArtifactStore(tmp_path / "artifacts"))
    task_lookup = reopened.lookup_task_inputs(task)
    result_lookup = reopened.lookup_evaluation_result(result)
    other_model = replace(result, model_artifact=_artifact("other-model"))

    assert task_lookup.status == result_lookup.status == "hit"
    assert task_lookup.entry is not None and task_lookup.entry.reference == task_entry.reference
    assert result_lookup.entry is not None and result_lookup.entry.reference == result_entry.reference
    assert task_lookup.entry.payload["token_ids"] == [[1, 2], [3, 4]]  # type: ignore[index]
    assert result_lookup.entry.payload["accuracy"] == 0.5  # type: ignore[index]
    assert reopened.lookup_evaluation_result(other_model).status == "miss"
    assert "complete semantic identity" in reopened.lookup_evaluation_result(other_model).reason


def test_cache_rejects_conflicting_payloads_under_one_semantic_identity(tmp_path: Path) -> None:
    _specification, task, _runtime, _result = _identities()
    cache = EvaluationCache(tmp_path / "run", LocalArtifactStore(tmp_path / "artifacts"))
    first = cache.commit_task_inputs(task, {"tokens": [[1, 2]]})

    assert cache.commit_task_inputs(task, {"tokens": [[1, 2]]}).reference == first.reference
    with pytest.raises(ValueError, match="conflicting task-input artifacts"):
        cache.commit_task_inputs(task, {"tokens": [[9, 9]]})


def test_cached_evaluator_preserves_typed_result_and_skips_reexecution(tmp_path: Path) -> None:
    _specification, _task, _runtime, identity = _identities()
    cache = EvaluationCache(tmp_path / "run", LocalArtifactStore(tmp_path / "artifacts"))
    calls = 0

    def evaluate() -> CausalEvaluationResult:
        nonlocal calls
        calls += 1
        return CausalEvaluationResult(4.0, 2.0, 7.389056, 2, 1, 1)

    def decode(payload: object) -> CausalEvaluationResult:
        assert isinstance(payload, dict)
        return from_dict(CausalEvaluationResult, payload, path="cached_result")

    first = evaluate_with_cache(cache, identity, evaluate, decode)
    second = evaluate_with_cache(cache, identity, evaluate, decode)

    assert first.result == second.result
    assert not first.cache_hit and second.cache_hit
    assert first.reference == second.reference
    assert calls == 1


def test_cache_validates_artifact_before_reuse(tmp_path: Path) -> None:
    _specification, task, _runtime, _result = _identities()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    cache = EvaluationCache(tmp_path / "run", artifacts)
    entry = cache.commit_task_inputs(task, {"tokens": [[1, 2]]})
    path = artifacts.path_for(entry.reference.artifact_id) / "entry.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactCorruptionError):
        cache.lookup_task_inputs(task)


def test_concurrent_cache_publication_does_not_lose_index_entries(tmp_path: Path) -> None:
    _specification, task, _runtime, _result = _identities()
    identities = tuple(replace(task, selection_seed=index) for index in range(8))

    def publish(identity: TaskInputCacheIdentity) -> None:
        cache = EvaluationCache(tmp_path / "run", LocalArtifactStore(tmp_path / "artifacts"))
        cache.commit_task_inputs(identity, {"selection_seed": identity.selection_seed})

    with ThreadPoolExecutor(max_workers=4) as executor:
        tuple(executor.map(publish, identities))

    reopened = EvaluationCache(tmp_path / "run", LocalArtifactStore(tmp_path / "artifacts"))
    assert all(reopened.lookup_task_inputs(identity).status == "hit" for identity in identities)
