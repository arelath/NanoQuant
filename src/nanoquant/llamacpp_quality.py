"""Protocol-matched quality evaluation of the exported GGUF through llama.cpp."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoquant.application.task_evaluation import (
    MultipleChoiceEvaluationResult,
    MultipleChoiceExampleResult,
)
from nanoquant.config.codec import canonical_json, to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.resource_usage import (
    GpuProcessMemoryMonitor,
    process_memory_snapshot,
)
from nanoquant.infrastructure.subprocess_interop import (
    LlamaCppInterop,
    SubprocessInterop,
    SubprocessRequest,
)
from nanoquant.quality_evaluation import (
    PreparedQualityInputs,
    QualityEvaluationRequest,
    compare_quality_results,
)

LLAMACPP_QUALITY_SCHEMA_VERSION = 2
_INPUT_MAGIC = b"NQQL0001"
_OUTPUT_MAGIC = b"NQQO0001"


@dataclass(frozen=True, slots=True)
class LlamaCppQualityRequest:
    gguf: Path
    output: Path
    llama_cpp_root: Path
    device: str = "cuda"
    runner: Path | None = None
    gpu_layers: int = -1
    parallel: int = 4
    threads: int = 0
    batch_threads: int = 0

    def __post_init__(self) -> None:
        if self.parallel <= 0:
            raise ValueError("llama.cpp quality parallel sequence count must be positive")
        if self.threads < 0 or self.batch_threads < 0:
            raise ValueError("llama.cpp quality thread counts must be non-negative")
        if not self.device:
            raise ValueError("llama.cpp quality device is required")


@dataclass(frozen=True, slots=True)
class _Sequence:
    tokens: tuple[int, ...]
    score_start: int

    def __post_init__(self) -> None:
        if len(self.tokens) < 2 or self.score_start <= 0 or self.score_start >= len(self.tokens):
            raise ValueError("llama.cpp quality sequence is invalid")
        if any(token < 0 or token > 2**31 - 1 for token in self.tokens):
            raise ValueError("llama.cpp quality token ID is outside the int32 range")


@dataclass(frozen=True, slots=True)
class _Score:
    negative_log_likelihood: float
    token_count: int


@dataclass(frozen=True, slots=True)
class _RunnerResourceMetrics:
    peak_device_bytes: int | None
    peak_device_shared_bytes: int | None
    peak_host_bytes: int | None
    measurement: str


@dataclass(frozen=True, slots=True)
class _TaskCandidate:
    example_index: int
    choice_index: int
    normalization_length: int


def _resolve_runner(request: LlamaCppQualityRequest) -> Path:
    configured = request.runner
    if configured is None:
        environment = os.environ.get("NANOQUANT_LLAMA_CPP_QUALITY_RUNNER")
        configured = None if not environment else Path(environment)
    executable = ".exe" if os.name == "nt" else ""
    candidates = (
        configured,
        request.llama_cpp_root
        / "build"
        / "nanoquant-quality"
        / "Release"
        / f"nanoquant-llamacpp-quality{executable}",
        request.llama_cpp_root
        / "build"
        / "nanoquant-quality"
        / f"nanoquant-llamacpp-quality{executable}",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise FileNotFoundError(
        "NanoQuant llama.cpp quality runner is missing; run "
        "tools/build_llamacpp_quality.py or set NANOQUANT_LLAMA_CPP_QUALITY_RUNNER. "
        f"Searched: {rendered}"
    )


def _git_capture(root: Path) -> dict[str, object]:
    def capture(*arguments: str) -> str:
        result = SubprocessInterop().run(
            SubprocessRequest(("git", "-C", str(root), *arguments))
        )
        result.require_success("git provenance capture")
        return result.stdout.strip()

    return {
        "repository": capture("config", "--get", "remote.origin.url"),
        "commit": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current") or None,
        "dirty": bool(capture("status", "--porcelain")),
    }


def _runtime_files(root: Path, runner: Path) -> tuple[Path, ...]:
    names = (
        "llama.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
        "ggml-cuda.dll",
        "libllama.so",
        "libggml.so",
        "libggml-base.so",
        "libggml-cpu.so",
        "libggml-cuda.so",
        "libllama.dylib",
    )
    search_roots = (
        runner.parent,
        root / "build" / "bin",
        root / "build" / "bin" / "Release",
        root / "build" / "src",
        root / "build" / "src" / "Release",
        root / "build" / "ggml" / "src",
        root / "build" / "ggml" / "src" / "Release",
    )
    found = {runner.resolve()}
    for directory in search_roots:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                found.add(candidate.resolve())
    return tuple(sorted(found, key=str))


def _write_input(path: Path, sequences: tuple[_Sequence, ...]) -> str:
    digest = hashlib.sha256()

    def write(output: Any, value: bytes) -> None:
        output.write(value)
        digest.update(value)

    with path.open("wb") as output:
        write(output, _INPUT_MAGIC)
        write(output, struct.pack("<I", len(sequences)))
        for sequence in sequences:
            write(output, struct.pack("<II", len(sequence.tokens), sequence.score_start))
            write(output, struct.pack(f"<{len(sequence.tokens)}i", *sequence.tokens))
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest()


def _read_scores(path: Path, expected: int) -> tuple[_Score, ...]:
    with path.open("rb") as source:
        if source.read(8) != _OUTPUT_MAGIC:
            raise ValueError("llama.cpp quality output has an unsupported header")
        count_payload = source.read(4)
        if len(count_payload) != 4:
            raise ValueError("llama.cpp quality output is truncated")
        count = struct.unpack("<I", count_payload)[0]
        if count != expected:
            raise ValueError(f"llama.cpp quality score count differs: {count} != {expected}")
        scores = []
        for _index in range(count):
            payload = source.read(12)
            if len(payload) != 12:
                raise ValueError("llama.cpp quality score payload is truncated")
            nll, token_count = struct.unpack("<dI", payload)
            if not math.isfinite(nll) or nll < 0 or token_count <= 0:
                raise ValueError("llama.cpp quality score is invalid")
            scores.append(_Score(nll, token_count))
        if source.read(1):
            raise ValueError("llama.cpp quality output contains trailing bytes")
    return tuple(scores)


def _quality_sequences(
    quality: QualityEvaluationRequest,
    prepared: PreparedQualityInputs,
) -> tuple[tuple[_Sequence, ...], tuple[tuple[_TaskCandidate, ...], ...], tuple[int, ...]]:
    sequences = [
        _Sequence(tuple(int(token) for token in row.tolist()), 1)
        for row in prepared.wikitext_tokens
    ]
    task_candidates: list[tuple[_TaskCandidate, ...]] = []
    task_truncated: list[int] = []
    for task in prepared.tasks:
        candidates = []
        truncated = 0
        examples = task.examples[: quality.task_limit]
        for example_index, example in enumerate(examples):
            for choice_index, (context, continuation) in enumerate(
                zip(example.contexts, example.continuations, strict=True)
            ):
                if len(continuation) > task.task.maximum_length:
                    raise ValueError(
                        "multiple-choice continuation does not fit the task maximum length"
                    )
                retained = context[-(task.task.maximum_length + 1 - len(continuation)) :]
                truncated += int(len(retained) != len(context))
                sequence = (*retained, *continuation)
                sequences.append(_Sequence(sequence, len(retained)))
                normalization_length = (
                    example.normalization_lengths[choice_index]
                    if example.normalization_lengths
                    else len(continuation)
                )
                candidates.append(
                    _TaskCandidate(example_index, choice_index, normalization_length)
                )
        task_candidates.append(tuple(candidates))
        task_truncated.append(truncated)
    return tuple(sequences), tuple(task_candidates), tuple(task_truncated)


def _prediction(values: tuple[float, ...]) -> tuple[int, bool]:
    maximum = max(values)
    winners = tuple(index for index, value in enumerate(values) if value == maximum)
    return winners[0], len(winners) > 1


def _task_result(
    prepared: Any,
    candidates: tuple[_TaskCandidate, ...],
    scores: tuple[_Score, ...],
    truncated: int,
    maximum_samples: int,
) -> MultipleChoiceEvaluationResult:
    examples = prepared.examples[:maximum_samples]
    raw_scores = [[0.0] * len(example.contexts) for example in examples]
    mean_scores = [[0.0] * len(example.contexts) for example in examples]
    for candidate, score in zip(candidates, scores, strict=True):
        log_likelihood = -score.negative_log_likelihood
        raw_scores[candidate.example_index][candidate.choice_index] = log_likelihood
        mean_scores[candidate.example_index][candidate.choice_index] = (
            log_likelihood / candidate.normalization_length
        )
    results = []
    for example, raw, normalized in zip(examples, raw_scores, mean_scores, strict=True):
        raw_values = tuple(raw)
        normalized_values = tuple(normalized)
        raw_prediction, raw_tie = _prediction(raw_values)
        normalized_prediction, normalized_tie = _prediction(normalized_values)
        results.append(
            MultipleChoiceExampleResult(
                example.sample_id,
                example.correct_choice,
                raw_prediction,
                normalized_prediction,
                raw_values,
                normalized_values,
                raw_prediction == example.correct_choice,
                normalized_prediction == example.correct_choice,
                raw_tie,
                normalized_tie,
            )
        )
    result_tuple = tuple(results)
    raw_correct = sum(result.raw_correct for result in result_tuple)
    normalized_correct = sum(result.normalized_correct for result in result_tuple)
    accuracy = raw_correct / len(result_tuple)
    normalized_accuracy = normalized_correct / len(result_tuple)
    primary = (
        accuracy
        if prepared.task.primary_metric == "acc"
        else normalized_accuracy
    )
    return MultipleChoiceEvaluationResult(
        prepared.task.semantic_key,
        prepared.task.task_name,
        prepared.task.task_version,
        prepared.task.prompt_hash,
        len(result_tuple),
        raw_correct,
        normalized_correct,
        accuracy,
        normalized_accuracy,
        prepared.task.primary_metric,
        primary,
        truncated,
        sum(result.raw_tie or result.normalized_tie for result in result_tuple),
        result_tuple,
    )


def _run(
    request: LlamaCppQualityRequest,
    runner: Path,
    input_path: Path,
    output_path: Path,
) -> _RunnerResourceMetrics:
    command = (
        str(runner),
        "--model",
        str(request.gguf.resolve()),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--gpu-layers",
        str(request.gpu_layers),
        "--parallel",
        str(request.parallel),
        "--threads",
        str(request.threads),
        "--batch-threads",
        str(request.batch_threads),
    )
    interop = LlamaCppInterop(request.llama_cpp_root)
    print(
        "llama.cpp GGUF quality started: "
        f"model={request.gguf} parallel={request.parallel} gpu_layers={request.gpu_layers}",
        flush=True,
    )
    child = interop.start_streaming(
        interop.request(command, environment=interop.runtime_environment(runner)),
        on_stderr=lambda line: print(line, flush=True),
    )
    gpu_monitor = None
    try:
        gpu_monitor = GpuProcessMemoryMonitor(child.pid)
    except (OSError, AttributeError):
        pass
    peak_device_bytes = 0
    peak_device_shared_bytes = 0
    peak_host_bytes = 0

    def sample_resources() -> None:
        nonlocal peak_device_bytes, peak_device_shared_bytes, peak_host_bytes
        try:
            host = process_memory_snapshot(child.pid)
        except (OSError, FileNotFoundError, ProcessLookupError):
            host = None
        if host is not None:
            peak_host_bytes = max(peak_host_bytes, host.peak_working_set_bytes)
        gpu = None if gpu_monitor is None else gpu_monitor.sample()
        if gpu is not None:
            peak_device_bytes = max(peak_device_bytes, gpu.peak_dedicated_bytes)
            peak_device_shared_bytes = max(
                peak_device_shared_bytes,
                gpu.peak_shared_bytes,
            )

    while child.poll() is None:
        sample_resources()
        time.sleep(0.05)
    sample_resources()
    completed = child.wait()
    if completed.returncode != 0:
        detail = "\n".join(completed.stderr.splitlines()[-20:])
        raise RuntimeError(
            f"llama.cpp GGUF quality runner failed with exit code {completed.returncode}:\n{detail}"
        )
    return _RunnerResourceMetrics(
        peak_device_bytes or None,
        peak_device_shared_bytes or None,
        peak_host_bytes or None,
        "windows-pdh-child-process" if gpu_monitor is not None else "child-process-host-only",
    )


def execute_llamacpp_quality_evaluation(
    request: LlamaCppQualityRequest,
    quality: QualityEvaluationRequest,
    prepared: PreparedQualityInputs,
    base_result: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate an exported GGUF on the same token-level protocol as PyTorch."""

    gguf = request.gguf.resolve()
    root = request.llama_cpp_root.resolve()
    if not gguf.is_file():
        raise FileNotFoundError(f"llama.cpp quality GGUF is missing: {gguf}")
    if not (root / ".git").is_dir():
        raise FileNotFoundError(f"llama.cpp quality repository is missing: {root}")
    runner = _resolve_runner(request)
    sequences, task_candidates, task_truncated = _quality_sequences(quality, prepared)
    request.output.parent.mkdir(parents=True, exist_ok=True)
    input_descriptor, input_name = tempfile.mkstemp(
        prefix=".llamacpp-quality-input-",
        suffix=".bin",
        dir=request.output.parent,
    )
    os.close(input_descriptor)
    output_descriptor, output_name = tempfile.mkstemp(
        prefix=".llamacpp-quality-output-",
        suffix=".bin",
        dir=request.output.parent,
    )
    os.close(output_descriptor)
    input_path = Path(input_name)
    output_path = Path(output_name)
    try:
        input_sha256 = _write_input(input_path, sequences)
        runtime_files = _runtime_files(root, runner)
        git = _git_capture(root)
        file_records = tuple(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in runtime_files
        )
        runtime: dict[str, Any] = {
            "git": git,
            "files": file_records,
        }
        runtime_sha256 = hashlib.sha256(
            canonical_json(
                {
                    "git": git,
                    "files": tuple(
                        {
                            "name": Path(str(record["path"])).name,
                            "bytes": record["bytes"],
                            "sha256": record["sha256"],
                        }
                        for record in file_records
                    ),
                }
            ).encode("utf-8")
        ).hexdigest()
        protocol_sha256 = hashlib.sha256(
            canonical_json(
                {
                    "protocol": protocol,
                    "tasks": tuple(
                        {
                            "task": to_dict(task.task),
                            "cache_identity": to_dict(task.cache_identity),
                            "examples": tuple(
                                {
                                    "sample_id": example.sample_id,
                                    "correct_choice": example.correct_choice,
                                    "normalization_lengths": example.normalization_lengths,
                                }
                                for example in task.examples[: quality.task_limit]
                            ),
                        }
                        for task in prepared.tasks
                    ),
                }
            ).encode("utf-8")
        ).hexdigest()
        base_result_sha256 = hashlib.sha256(
            canonical_json(base_result).encode("utf-8")
        ).hexdigest()
        identity = {
            "schema_version": LLAMACPP_QUALITY_SCHEMA_VERSION,
            "gguf_sha256": hash_file(gguf),
            "gguf_bytes": gguf.stat().st_size,
            "input_sha256": input_sha256,
            "runner_sha256": hash_file(runner),
            "llama_cpp_commit": git["commit"],
            "runtime_sha256": runtime_sha256,
            "protocol_sha256": protocol_sha256,
            "base_result_sha256": base_result_sha256,
            "gpu_layers": request.gpu_layers,
            "parallel": request.parallel,
            "threads": request.threads,
            "batch_threads": request.batch_threads,
            "scoring": "causal-target-log-likelihood-f32-logits-f64-host-logsumexp-v1",
        }
        if request.output.is_file():
            try:
                existing = json.loads(request.output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if (
                isinstance(existing, dict)
                and existing.get("passed") is True
                and existing.get("identity") == identity
            ):
                existing["reused"] = True
                print(
                    f"Reusing completed llama.cpp GGUF quality result: {request.output}",
                    flush=True,
                )
                return existing

        started = time.perf_counter()
        with acquire_device_lease(request.device):
            measured = _run(request, runner, input_path, output_path)
        if measured is None:
            measured = _RunnerResourceMetrics(None, None, None, "unavailable")
        scores = _read_scores(output_path, len(sequences))
        wikitext_count = prepared.wikitext_tokens.shape[0]
        wikitext_scores = scores[:wikitext_count]
        total_nll = sum(score.negative_log_likelihood for score in wikitext_scores)
        total_tokens = sum(score.token_count for score in wikitext_scores)
        mean_nll = total_nll / total_tokens
        offset = wikitext_count
        tasks = []
        for prepared_task, candidates, truncated in zip(
            prepared.tasks, task_candidates, task_truncated, strict=True
        ):
            selected_scores = scores[offset : offset + len(candidates)]
            offset += len(candidates)
            result = _task_result(
                prepared_task,
                candidates,
                selected_scores,
                truncated,
                quality.task_limit,
            )
            tasks.append(
                {
                    "task": to_dict(prepared_task.task),
                    "task_input_identity": to_dict(prepared_task.cache_identity),
                    "result": to_dict(result),
                }
            )
        elapsed = time.perf_counter() - started
        candidate_result: dict[str, Any] = {
            "label": "gguf",
            "wikitext": {
                "total_negative_log_likelihood": total_nll,
                "mean_negative_log_likelihood": mean_nll,
                "perplexity": math.exp(mean_nll),
                "token_count": total_tokens,
                "window_count": wikitext_count,
                "sample_count": wikitext_count,
            },
            "tasks": tasks,
            "elapsed_seconds": elapsed,
            "peak_device_bytes": measured.peak_device_bytes,
            "peak_device_shared_bytes": measured.peak_device_shared_bytes,
            "peak_host_bytes": measured.peak_host_bytes,
            "memory_measurement": measured.measurement,
            "execution": "llama.cpp",
        }
        payload = {
            "schema_version": LLAMACPP_QUALITY_SCHEMA_VERSION,
            "passed": True,
            "identity": identity,
            "gguf": {
                "path": str(gguf),
                "bytes": gguf.stat().st_size,
                "sha256": identity["gguf_sha256"],
            },
            "model": {
                "source": quality.source,
                "revision": quality.revision,
                "snapshot": str(quality.snapshot.resolve()),
            },
            "runtime": runtime,
            "protocol": protocol,
            "results": {"base": base_result, "gguf": candidate_result},
            "comparison": compare_quality_results(base_result, candidate_result),
            "wall_seconds": elapsed,
            "reused": False,
        }
        atomic_write_json(request.output, payload)
        print(
            "llama.cpp GGUF quality completed: "
            f"perplexity={candidate_result['wikitext']['perplexity']:.6f} "
            f"elapsed_seconds={elapsed:.2f}",
            flush=True,
        )
        return payload
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def render_llamacpp_quality_markdown(payload: dict[str, Any]) -> str:
    """Render a compact deployment-quality section for the combined report."""

    comparison = payload["comparison"]
    wikitext = comparison["wikitext"]
    gguf_result = payload["results"]["gguf"]
    peak_device = gguf_result.get("peak_device_bytes")
    peak_shared = gguf_result.get("peak_device_shared_bytes")
    peak_host = gguf_result.get("peak_host_bytes")

    def bytes_or_unavailable(value: object) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("llama.cpp quality memory measurement is not numeric")
        return f"{int(value):,}"

    lines = [
        "## GGUF deployment quality (llama.cpp)",
        "",
        (
            f"- GGUF: `{payload['gguf']['path']}` "
            f"({int(payload['gguf']['bytes']):,} bytes; `sha256:{payload['gguf']['sha256']}`)"
        ),
        f"- llama.cpp commit: `{payload['runtime']['git']['commit']}`",
        (
            "- Scoring: identical prepared token IDs and target positions; "
            "llama.cpp F32 logits with host F64 log-sum-exp"
        ),
        f"- Wall time: {float(payload['wall_seconds']):.2f} seconds",
        f"- Runtime memory measurement: `{gguf_result.get('memory_measurement', 'unavailable')}`",
        "",
        "| Packed GGUF runtime resource | Peak bytes |",
        "| --- | ---: |",
        f"| Dedicated GPU memory | {bytes_or_unavailable(peak_device)} |",
        f"| Shared GPU memory | {bytes_or_unavailable(peak_shared)} |",
        f"| Host working set | {bytes_or_unavailable(peak_host)} |",
        "",
        "| Benchmark | Metric | BF16 | GGUF | Delta | Ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    base_ppl = float(wikitext["base_perplexity"])
    gguf_ppl = float(wikitext["frozen_perplexity"])
    lines.append(
        f"| WikiText-2 | perplexity ↓ | {base_ppl:.6f} | {gguf_ppl:.6f} | "
        f"{gguf_ppl - base_ppl:+.6f} | {gguf_ppl / base_ppl:.4f}x |"
    )
    for item in comparison["tasks"]:
        baseline = float(item["base"])
        candidate = float(item["frozen"])
        ratio = item["ratio"]
        ratio_text = "n/a" if ratio is None else f"{float(ratio):.4f}x"
        lines.append(
            f"| {item['task_name']} | {item['metric']} ↑ | {baseline:.4f} | "
            f"{candidate:.4f} | {candidate - baseline:+.4f} | {ratio_text} |"
        )
    return "\n".join((*lines, ""))


__all__ = [
    "LLAMACPP_QUALITY_SCHEMA_VERSION",
    "LlamaCppQualityRequest",
    "execute_llamacpp_quality_evaluation",
    "render_llamacpp_quality_markdown",
]
