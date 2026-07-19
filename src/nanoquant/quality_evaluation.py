"""Pinned base-versus-frozen quality evaluation over WikiText and legacy tasks."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.application.evaluation import (
    CausalEvaluationRequest,
    evaluate_causal_nll,
    model_logits,
)
from nanoquant.application.task_evaluation import (
    MultipleChoiceEvaluationRequest,
    PreparedMultipleChoiceInputs,
    evaluate_multiple_choice,
    pinned_legacy_multiple_choice_tasks,
)
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.device_memory import SharedDeviceMemoryMonitor
from nanoquant.infrastructure.frozen_model_loader import LoadedFrozenModel, load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_task_evaluation import (
    hash_hf_tokenizer_snapshot,
    load_pinned_dataset_split,
    prepare_pinned_hf_multiple_choice_inputs,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.packed_model_loader import LoadedPackedModel, load_packed_model
from nanoquant.infrastructure.resource_usage import (
    peak_device_memory_bytes,
    peak_process_memory_bytes,
)
from nanoquant.infrastructure.streamed_language_model import BlockStreamedCausalLM

WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
DEFAULT_QUALITY_WIKITEXT_BATCH_SIZE = 8
DEFAULT_QUALITY_TASK_BATCH_SIZE = 4
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

QualityProgressCallback = Callable[[str, Mapping[str, object]], None]


def _emit_progress(
    progress: QualityProgressCallback | None,
    event: str,
    **fields: object,
) -> None:
    if progress is not None:
        progress(event, fields)


@dataclass(frozen=True, slots=True)
class QualityEvaluationRequest:
    snapshot: Path
    source: str
    revision: str
    run_output: Path
    device: str = "cuda:0"
    backend: str = "factorized"
    use_global_tuning: bool = True
    wikitext_samples: int = 16
    wikitext_sequence_length: int = 128
    wikitext_batch_size: int = DEFAULT_QUALITY_WIKITEXT_BATCH_SIZE
    task_names: tuple[str, ...] = ("piqa", "arc_easy", "boolq")
    task_limit: int = 25
    task_batch_size: int = DEFAULT_QUALITY_TASK_BATCH_SIZE
    local_files_only: bool = False
    maximum_wddm_shared_bytes: int | None = None
    packed_artifact: Path | None = None
    stream_base_model: bool = False

    def __post_init__(self) -> None:
        if not self.source or not self.revision:
            raise ValueError("quality evaluation model source and revision are required")
        if self.backend not in {"factorized", "dense"}:
            raise ValueError("quality evaluation backend is unsupported")
        if self.wikitext_samples <= 0 or self.wikitext_sequence_length < 2:
            raise ValueError("quality evaluation WikiText dimensions are invalid")
        if self.wikitext_batch_size <= 0 or self.task_limit <= 0 or self.task_batch_size <= 0:
            raise ValueError("quality evaluation batch sizes and task limit must be positive")
        if self.maximum_wddm_shared_bytes is not None and self.maximum_wddm_shared_bytes < 0:
            raise ValueError("quality evaluation shared-memory limit must be non-negative")
        supported = {task.task_name for task in pinned_legacy_multiple_choice_tasks()}
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("quality evaluation task names must be non-empty and unique")
        unknown = set(self.task_names) - supported
        if unknown:
            raise ValueError(f"quality evaluation tasks are unsupported: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class PreparedQualityInputs:
    wikitext_tokens: torch.Tensor
    wikitext_fingerprint: str
    bos_token_id: int
    pad_token_id: int
    tokenizer_hash: str
    tasks: tuple[PreparedMultipleChoiceInputs, ...]


def _checkpoint_dtype(snapshot: Path) -> torch.dtype:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(config.get("torch_dtype"), torch.float32)


def _wikitext_tokens(
    snapshot: Path,
    *,
    samples: int,
    sequence_length: int,
    local_files_only: bool,
    progress: QualityProgressCallback | None = None,
) -> tuple[torch.Tensor, str, int]:
    _emit_progress(progress, "wikitext_input_dataset_started", dataset=WIKITEXT_DATASET)
    dataset = load_pinned_dataset_split(
        WIKITEXT_DATASET,
        WIKITEXT_CONFIG,
        WIKITEXT_REVISION,
        "test",
        local_files_only=local_files_only,
    )
    _emit_progress(
        progress,
        "wikitext_input_dataset_completed",
        dataset=WIKITEXT_DATASET,
        rows=len(dataset),
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=False)
    payload = sequence_length - 1
    required = samples * payload
    # The protocol evaluates independent, bounded windows.  Ask the tokenizer
    # for exactly the prefix those windows consume instead of materializing the
    # entire WikiText test split as one apparent model input.  Explicit
    # truncation both preserves the historical token prefix and prevents
    # Transformers from warning that the full corpus exceeds model context.
    _emit_progress(progress, "wikitext_input_tokenization_started")
    encoded = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
        truncation=True,
        max_length=required,
    ).input_ids
    _emit_progress(progress, "wikitext_input_tokenization_completed", tokens=encoded.shape[1])
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("Gemma WikiText protocol requires a BOS token")
    if encoded.shape[1] < required:
        raise ValueError(f"WikiText token stream has {encoded.shape[1]} tokens; protocol requires {required}")
    rows = tuple(
        torch.cat(
            (
                torch.tensor([[bos_id]], dtype=encoded.dtype),
                encoded[:, index * payload : (index + 1) * payload],
            ),
            dim=1,
        )
        for index in range(samples)
    )
    return torch.cat(rows, dim=0), str(getattr(dataset, "_fingerprint", "unknown")), int(bos_id)


def prepare_quality_inputs(
    request: QualityEvaluationRequest,
    progress: QualityProgressCallback | None = None,
) -> PreparedQualityInputs:
    """Materialize the exact shared input partitions before claiming the GPU lease."""

    tokens, fingerprint, bos_id = _wikitext_tokens(
        request.snapshot,
        samples=request.wikitext_samples,
        sequence_length=request.wikitext_sequence_length,
        local_files_only=request.local_files_only,
        progress=progress,
    )
    _emit_progress(progress, "tokenizer_load_started")
    tokenizer = AutoTokenizer.from_pretrained(request.snapshot, local_files_only=False)
    _emit_progress(progress, "tokenizer_load_completed")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("quality evaluation tokenizer contains no pad token ID")
    _emit_progress(progress, "tokenizer_hash_started")
    tokenizer_hash = hash_hf_tokenizer_snapshot(request.snapshot)
    _emit_progress(progress, "tokenizer_hash_completed", tokenizer_hash=tokenizer_hash)
    by_name = {task.task_name: task for task in pinned_legacy_multiple_choice_tasks()}
    tasks = []
    for task_index, name in enumerate(request.task_names, start=1):
        _emit_progress(
            progress,
            "task_input_started",
            task=name,
            task_index=task_index,
            task_count=len(request.task_names),
        )
        prepared = prepare_pinned_hf_multiple_choice_inputs(
            by_name[name],
            tokenizer,
            tokenizer_name=request.source,
            tokenizer_revision=request.revision,
            tokenizer_content_hash=tokenizer_hash,
            maximum_samples=request.task_limit,
            local_files_only=request.local_files_only,
        )
        tasks.append(prepared)
        _emit_progress(
            progress,
            "task_input_completed",
            task=name,
            examples=len(prepared.examples),
        )
    return PreparedQualityInputs(
        tokens,
        fingerprint,
        bos_id,
        int(pad_token_id),
        tokenizer_hash,
        tuple(tasks),
    )


def _release_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_model(
    label: str,
    model: nn.Module,
    request: QualityEvaluationRequest,
    inputs: PreparedQualityInputs,
    monitor: SharedDeviceMemoryMonitor | None = None,
    progress: QualityProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    _emit_progress(progress, "model_evaluation_started", model=label)
    if request.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(request.device)
    cast(Any, model).config.use_cache = False
    model.eval()
    raw_logits = model_logits(model)

    def guarded_logits(tokens: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        logits = raw_logits(tokens, attention_mask)
        if monitor is not None:
            monitor.check()
        return logits

    wikitext_started = time.perf_counter()
    _emit_progress(progress, "wikitext_started", model=label)
    causal = evaluate_causal_nll(
        CausalEvaluationRequest(
            inputs.wikitext_tokens.to(request.device),
            max_length=request.wikitext_sequence_length,
            stride=request.wikitext_sequence_length,
            batch_size=request.wikitext_batch_size,
        ),
        guarded_logits,
    )
    wikitext_seconds = time.perf_counter() - wikitext_started
    _emit_progress(
        progress,
        "wikitext_completed",
        model=label,
        elapsed_seconds=wikitext_seconds,
        perplexity=causal.perplexity,
    )
    tasks = []
    for task_index, prepared in enumerate(inputs.tasks, start=1):
        task_started = time.perf_counter()
        _emit_progress(
            progress,
            "task_started",
            model=label,
            task=prepared.task.task_name,
            task_index=task_index,
            task_count=len(inputs.tasks),
        )
        result = evaluate_multiple_choice(
            MultipleChoiceEvaluationRequest(
                prepared.task,
                prepared.examples,
                batch_size=request.task_batch_size,
                maximum_samples=request.task_limit,
                pad_token_id=inputs.pad_token_id,
                device=request.device,
            ),
            guarded_logits,
        )
        tasks.append(
            {
                "task": to_dict(prepared.task),
                "task_input_identity": to_dict(prepared.cache_identity),
                "result": to_dict(result),
                "elapsed_seconds": time.perf_counter() - task_started,
            }
        )
        _emit_progress(
            progress,
            "task_completed",
            model=label,
            task=prepared.task.task_name,
            elapsed_seconds=tasks[-1]["elapsed_seconds"],
            metric=result.primary_metric,
            value=result.primary_value,
        )
    payload = {
        "label": label,
        "wikitext": to_dict(causal),
        "wikitext_elapsed_seconds": wikitext_seconds,
        "tasks": tasks,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_device_bytes": peak_device_memory_bytes(request.device),
        "peak_host_bytes": peak_process_memory_bytes(),
    }
    _emit_progress(
        progress,
        "model_evaluation_completed",
        model=label,
        elapsed_seconds=payload["elapsed_seconds"],
    )
    return payload


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"quality evaluation {field} is not numeric")
    return float(value)


def _comparison(base: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    base_ppl = _number(cast(dict[str, object], base["wikitext"])["perplexity"], "base perplexity")
    frozen_ppl = _number(
        cast(dict[str, object], frozen["wikitext"])["perplexity"],
        "frozen perplexity",
    )
    base_tasks = {
        str(cast(dict[str, object], item["result"])["task_name"]): cast(dict[str, object], item["result"])
        for item in cast(list[dict[str, object]], base["tasks"])
    }
    frozen_tasks = {
        str(cast(dict[str, object], item["result"])["task_name"]): cast(dict[str, object], item["result"])
        for item in cast(list[dict[str, object]], frozen["tasks"])
    }
    task_rows = []
    for name in base_tasks:
        baseline = _number(base_tasks[name]["primary_value"], f"{name} base metric")
        candidate = _number(frozen_tasks[name]["primary_value"], f"{name} frozen metric")
        task_rows.append(
            {
                "task_name": name,
                "metric": str(base_tasks[name]["primary_metric"]),
                "base": baseline,
                "frozen": candidate,
                "delta": candidate - baseline,
                "ratio": None if baseline == 0 else candidate / baseline,
            }
        )
    return {
        "wikitext": {
            "base_perplexity": base_ppl,
            "frozen_perplexity": frozen_ppl,
            "ratio": frozen_ppl / base_ppl,
            "relative_change": frozen_ppl / base_ppl - 1.0,
        },
        "tasks": task_rows,
    }


def execute_quality_evaluation(
    request: QualityEvaluationRequest,
    *,
    prepared: PreparedQualityInputs | None = None,
    progress: QualityProgressCallback | None = None,
) -> dict[str, Any]:
    """Evaluate the base and committed frozen models on identical pinned inputs."""

    _emit_progress(progress, "input_preparation_started", reused=prepared is not None)
    inputs = prepare_quality_inputs(request, progress) if prepared is None else prepared
    _emit_progress(progress, "input_preparation_completed", task_count=len(inputs.tasks))
    wall_started = time.perf_counter()
    _emit_progress(progress, "device_lease_started", device=request.device)
    with acquire_device_lease(request.device):
        _emit_progress(progress, "device_lease_acquired", device=request.device)
        monitor_context = (
            SharedDeviceMemoryMonitor(request.maximum_wddm_shared_bytes)
            if request.maximum_wddm_shared_bytes is not None
            else nullcontext(None)
        )
        with monitor_context as monitor:
            model_config = AutoConfig.from_pretrained(request.snapshot, local_files_only=False)
            model_type = str(getattr(model_config, "model_type", "")).lower()
            attention_implementation = "eager" if model_type.startswith("gemma") else "sdpa"
            base_load_started = time.perf_counter()
            _emit_progress(progress, "model_load_started", model="base")
            source_model = load_causal_language_model(
                request.snapshot,
                torch_dtype=_checkpoint_dtype(request.snapshot),
                attention_implementation=attention_implementation,
            ).to("cpu" if request.stream_base_model else request.device)
            base = (
                BlockStreamedCausalLM(
                    source_model,
                    adapter_for_config(cast(dict[str, object], model_config.to_dict())),
                    request.device,
                )
                if request.stream_base_model
                else source_model
            )
            if monitor is not None:
                monitor.check()
            base_load_seconds = time.perf_counter() - base_load_started
            _emit_progress(
                progress,
                "model_load_completed",
                model="base",
                elapsed_seconds=base_load_seconds,
            )
            try:
                base_result = _evaluate_model("base", base, request, inputs, monitor, progress)
                base_result["model_load_seconds"] = base_load_seconds
                base_result["execution"] = "block_streamed" if request.stream_base_model else "resident"
            finally:
                del base, source_model
                _release_device_memory()
            loaded: LoadedFrozenModel | LoadedPackedModel | None = None
            packed_descriptor_sha256 = None
            try:
                frozen_load_started = time.perf_counter()
                _emit_progress(progress, "model_load_started", model="frozen")
                loaded = (
                    load_frozen_run(
                        request.run_output,
                        request.snapshot,
                        source_name=request.source,
                        revision=request.revision,
                        device=request.device,
                        backend=request.backend,
                        use_global_tuning=request.use_global_tuning,
                    )
                    if request.packed_artifact is None
                    else load_packed_model(
                        request.packed_artifact,
                        request.run_output,
                        request.snapshot,
                        source_name=request.source,
                        revision=request.revision,
                        device=request.device,
                        backend=request.backend,
                        use_global_tuning=request.use_global_tuning,
                    )
                )
                if monitor is not None:
                    monitor.check()
                frozen_load_seconds = time.perf_counter() - frozen_load_started
                _emit_progress(
                    progress,
                    "model_load_completed",
                    model="frozen",
                    elapsed_seconds=frozen_load_seconds,
                )
                frozen_result = _evaluate_model(
                    "frozen",
                    loaded.model,
                    request,
                    inputs,
                    monitor,
                    progress,
                )
                frozen_result["model_load_seconds"] = frozen_load_seconds
                frozen_identity = to_dict(loaded.identity)
                global_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
                packed_descriptor_sha256 = (
                    loaded.packed_descriptor_sha256 if isinstance(loaded, LoadedPackedModel) else None
                )
            finally:
                if loaded is not None:
                    del loaded
                _release_device_memory()
    token_hash = "sha256:" + hashlib.sha256(
        inputs.wikitext_tokens.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "passed": all(
            math.isfinite(
                _number(cast(dict[str, object], case["wikitext"])["perplexity"], "perplexity")
            )
            for case in (base_result, frozen_result)
        ),
        "model": {
            "source": request.source,
            "revision": request.revision,
            "snapshot": str(request.snapshot.resolve()),
        },
        "candidate": {
            "run_output": str(request.run_output.resolve()),
            "commit_identity": frozen_identity,
            "global_tuning": global_tuning,
            "backend": request.backend,
            "packed_artifact": None
            if request.packed_artifact is None
            else str(request.packed_artifact.resolve()),
            "packed_descriptor_sha256": None
            if request.packed_artifact is None
            else packed_descriptor_sha256,
        },
        "protocol": {
            "wikitext_dataset": f"{WIKITEXT_DATASET}:{WIKITEXT_CONFIG}:test@{WIKITEXT_REVISION}",
            "wikitext_fingerprint": inputs.wikitext_fingerprint,
            "wikitext_samples": request.wikitext_samples,
            "wikitext_sequence_length": request.wikitext_sequence_length,
            "wikitext_batch_size": request.wikitext_batch_size,
            "wikitext_token_hash": token_hash,
            "task_names": request.task_names,
            "task_limit": request.task_limit,
            "task_batch_size": request.task_batch_size,
            "tokenizer_hash": inputs.tokenizer_hash,
            "base_execution": "block_streamed" if request.stream_base_model else "resident",
        },
        "results": {"base": base_result, "frozen": frozen_result},
        "comparison": _comparison(base_result, frozen_result),
        "wall_seconds": time.perf_counter() - wall_started,
        "resource_limits": {
            "maximum_wddm_shared_bytes": request.maximum_wddm_shared_bytes,
            "peak_wddm_shared_bytes": None if monitor is None else monitor.guard.peak_bytes,
        },
    }
    _emit_progress(progress, "quality_evaluation_completed", wall_seconds=payload["wall_seconds"])
    return payload


__all__ = [
    "DEFAULT_QUALITY_TASK_BATCH_SIZE",
    "DEFAULT_QUALITY_WIKITEXT_BATCH_SIZE",
    "PreparedQualityInputs",
    "QualityProgressCallback",
    "QualityEvaluationRequest",
    "execute_quality_evaluation",
    "prepare_quality_inputs",
]
