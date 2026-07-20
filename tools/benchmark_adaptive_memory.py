"""Build the requested real-model adaptive-memory admission matrix.

The default is metadata-only. ``--probe-model-load`` additionally materializes
each selected placement under the cross-process CUDA lease and releases it
before proceeding to the next model.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import _paths  # noqa: F401
import torch
from recipes.base_compression import (
    BASE_COMPRESSION_TEMPLATE,
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE,
)

from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.config.codec import to_dict
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ExecutorKind,
    MemoryPolicyConfig,
    MemoryPolicyMode,
    MemoryPolicyProfile,
    RunConfig,
)
from nanoquant.domain.resources import (
    ResolvedMemoryPlan,
    select_fastest_observed_batch,
    throughput_batch_candidates,
)
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.device_memory import sample_device_memory
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.resource_planning import build_resident_memory_plan
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.resident_quantization import (
    _forward_metadata_to_device,
    _run_block_batched,
    _run_prefix_batched,
)


@dataclass(frozen=True, slots=True)
class ModelCase:
    source: str
    revision: str
    cache_directory: str
    template: RunConfig
    retained_evidence: Path
    architecture_proxy: bool = False


CASES = (
    ModelCase(
        "google/gemma-3-270m-it",
        "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
        "models--google--gemma-3-270m-it",
        BASE_COMPRESSION_TEMPLATE,
        Path("evidence/016/016-compress-and-benchmark-gemma-3-270m-it"),
        # The retained run uses the Unsloth mirror of the same architecture.
        True,
    ),
    ModelCase(
        "google/gemma-3-1b-it",
        "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "models--google--gemma-3-1b-it",
        BASE_COMPRESSION_TEMPLATE,
        Path("evidence/017/017-compress-and-benchmark-gemma-3-1b-it"),
    ),
    ModelCase(
        "meta-llama/Llama-3.2-1B-Instruct",
        "9213176726f574b556790deb65791e0c5aa438b6",
        "models--meta-llama--Llama-3.2-1B-Instruct",
        LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE,
        Path("evidence/019/019-compress-and-benchmark-llama-3-2-1b-instruct"),
    ),
    ModelCase(
        "google/gemma-3-4b-it",
        "093f9f388b31de276ce2de164bdc2081324b9767",
        "models--google--gemma-3-4b-it",
        GEMMA_3_4B_COMPRESSION_TEMPLATE,
        Path("evidence/018/018-compress-and-benchmark-gemma-3-4b-it"),
    ),
)

_THROUGHPUT_PROBE_REPETITIONS = 5


def _snapshot(cache_root: Path, case: ModelCase) -> Path:
    snapshot = cache_root / case.cache_directory / "snapshots" / case.revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"requested local snapshot is unavailable: {snapshot}")
    return snapshot.resolve()


def _adaptive_config(case: ModelCase, profile: MemoryPolicyProfile) -> RunConfig:
    template = case.template
    return replace(
        template,
        model=replace(
            template.model,
            source=case.source,
            revision=case.revision,
            tokenizer_source=case.source,
            tokenizer_revision=case.revision,
        ),
        runtime=replace(
            template.runtime,
            executor=ExecutorKind.AUTO,
            memory_policy=MemoryPolicyConfig(mode=MemoryPolicyMode.ADAPTIVE, profile=profile),
            activations=replace(template.runtime.activations, gpu_cache=ActivationGpuCacheMode.AUTO),
            on_cuda_oom=("reduce_batch_size", "move_activations_down_one_tier", "fail"),
        ),
        # Current CPU-offload execution cannot stream the model-level teacher.
        # Disabling KD does not change the compression-stage memory benchmark.
        distillation=replace(template.distillation, enabled=False),
    )


def _retained_observation(repository: Path, case: ModelCase) -> dict[str, object]:
    evidence = repository / case.retained_evidence
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    maxima = {
        "cuda_peak_allocated_bytes": 0,
        "cuda_peak_reserved_bytes": 0,
        "wddm_peak_dedicated_bytes": 0,
        "wddm_peak_shared_bytes": 0,
        "host_peak_working_set_bytes": 0,
        "host_peak_private_bytes": 0,
    }
    completed_blocks: set[int] = set()
    for line in (evidence / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        fields = event.get("fields", {})
        maxima["cuda_peak_allocated_bytes"] = max(
            maxima["cuda_peak_allocated_bytes"],
            int(fields.get("cuda.peak_allocated_bytes", 0) or 0),
            int(fields.get("cuda.window_peak_allocated_bytes", 0) or 0),
        )
        maxima["cuda_peak_reserved_bytes"] = max(
            maxima["cuda_peak_reserved_bytes"],
            int(fields.get("cuda.peak_reserved_bytes", 0) or 0),
            int(fields.get("cuda.window_peak_reserved_bytes", 0) or 0),
        )
        maxima["wddm_peak_dedicated_bytes"] = max(
            maxima["wddm_peak_dedicated_bytes"], int(fields.get("wddm.peak_dedicated_bytes", 0) or 0)
        )
        maxima["wddm_peak_shared_bytes"] = max(
            maxima["wddm_peak_shared_bytes"], int(fields.get("wddm.peak_shared_bytes", 0) or 0)
        )
        maxima["host_peak_working_set_bytes"] = max(
            maxima["host_peak_working_set_bytes"], int(fields.get("host.peak_working_set_bytes", 0) or 0)
        )
        maxima["host_peak_private_bytes"] = max(
            maxima["host_peak_private_bytes"], int(fields.get("host.peak_private_bytes", 0) or 0)
        )
        if event.get("name") == "block.completed" and isinstance(fields.get("block"), int):
            completed_blocks.add(fields["block"])
    return {
        "path": case.retained_evidence.as_posix(),
        "architecture_proxy": case.architecture_proxy,
        "status": manifest["status"],
        "completed_blocks": len(completed_blocks),
        **maxima,
    }


def _summary(plan: ResolvedMemoryPlan) -> dict[str, object]:
    predicted_stage_peak = max(stage.predicted_gpu_bytes for stage in plan.stages)
    cache_bytes = max(0, plan.peak_gpu_bytes - predicted_stage_peak)
    admitted_peak = max(
        stage.predicted_gpu_bytes + stage.uncertainty_bytes for stage in plan.stages
    ) + cache_bytes
    capacity = min(stage.gpu_capacity_bytes for stage in plan.stages)
    return {
        "executor": plan.executor,
        "activation_gpu_cache": plan.activation_gpu_cache,
        "predicted_peak_gpu_bytes": plan.peak_gpu_bytes,
        "admitted_peak_gpu_bytes": admitted_peak,
        "safe_gpu_capacity_bytes": capacity,
        "admission_headroom_bytes": capacity - admitted_peak,
        "predicted_peak_host_bytes": plan.peak_host_bytes,
        "predicted_peak_pinned_host_bytes": plan.peak_pinned_host_bytes,
        "predicted_peak_temporary_disk_bytes": plan.peak_temporary_disk_bytes,
        "stage_batches": {stage.stage: stage.batch_size for stage in plan.stages},
        "warnings": list(plan.warnings),
    }


def _block_elements(block: object) -> int:
    source_tensors = getattr(block, "source_tensors", ())
    total = 0
    for tensor in source_tensors:
        elements = 1
        for dimension in tensor.spec.shape:
            elements *= dimension
        total += elements
    return total


def _vocabulary_size(config: dict[str, object]) -> int:
    nested = config.get("text_config")
    source = nested if isinstance(nested, dict) else config
    value = source.get("vocab_size", 32_000)
    return int(value) if isinstance(value, (int, float)) else 32_000


def _probe_model_load(
    case: ModelCase,
    snapshot: Path,
    plan: ResolvedMemoryPlan,
    *,
    probe_block_forward: bool,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"passed": False, "error": "CUDA is unavailable"}
    stage = plan.stage("model_load")
    device = plan.envelope.device
    source = SafetensorsModelSource(snapshot, source=case.source, revision=case.revision, verify_hashes=False)
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    inventory = adapter.model_inventory(source)
    model = None
    loaded_model = None
    block = None
    tokens = None
    initial_inputs = None
    target_block = None
    metadata = None
    output = None
    decoder_layers = None
    capture = None
    forward_probe = None
    tuning_probe = None
    started = time.perf_counter()
    try:
        with acquire_device_lease(device):
            before = sample_device_memory()
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                model = load_causal_language_model(
                    snapshot,
                    torch_dtype=torch.bfloat16,
                    attention_implementation=adapter.attention_implementation,
                    local_files_only=True,
                )
                loaded_model = model
                if plan.executor == ExecutorKind.RESIDENT.value:
                    model.to(device)
                else:
                    largest_block = max(inventory.blocks, key=_block_elements)
                    block = adapter.load_block(source, largest_block.block, device)
                torch.cuda.synchronize(device)
                observed = sample_device_memory()
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
                if probe_block_forward:
                    forward_stage = plan.stage("block_forward")
                    batch_size = forward_stage.batch_size
                    sequence_length = case.template.model.sequence_length
                    model_device = device if plan.executor == ExecutorKind.RESIDENT.value else "cpu"
                    generator = torch.Generator(device=model_device).manual_seed(1729)
                    benchmark_samples = max(64, batch_size)
                    tokens = torch.randint(
                        0,
                        _vocabulary_size(checkpoint.config),
                        (benchmark_samples, sequence_length),
                        generator=generator,
                        device=model_device,
                    )
                    decoder_layers = adapter.get_decoder_layers(loaded_model)
                    capture = capture_prefix_invocations(
                        decoder_layers[0],
                        (lambda: adapter.run_decoder_forward(loaded_model, tokens[:1]),),
                    )[0]
                    initial_inputs = _run_prefix_batched(adapter, loaded_model, tokens, batch_size, "cpu")
                    largest_block = max(inventory.blocks, key=_block_elements)
                    target_block = (
                        decoder_layers[largest_block.block.index]
                        if plan.executor == ExecutorKind.RESIDENT.value
                        else block
                    )
                    if target_block is None:
                        raise RuntimeError("CPU-offload block probe has no loaded block")
                    metadata = _forward_metadata_to_device(capture.keyword, device)
                    legacy_batch_size = case.template.runtime.block_forward_batch_size
                    # Warm the kernels before comparing the former fixed batch
                    # with the adaptive batch over the same 64-sample workload.
                    output = _run_block_batched(
                        adapter,
                        target_block,
                        initial_inputs,
                        metadata,
                        legacy_batch_size,
                        "cpu",
                    )
                    torch.cuda.synchronize(device)
                    output = None
                    candidate_observations: list[tuple[int, float]] = []
                    candidate_peaks: dict[int, tuple[int, int]] = {}
                    for candidate in throughput_batch_candidates(batch_size, legacy_batch_size):
                        timings: list[float] = []
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)
                        for _ in range(_THROUGHPUT_PROBE_REPETITIONS):
                            candidate_started = time.perf_counter()
                            output = _run_block_batched(
                                adapter,
                                target_block,
                                initial_inputs,
                                metadata,
                                candidate,
                                "cpu",
                            )
                            torch.cuda.synchronize(device)
                            timings.append(time.perf_counter() - candidate_started)
                            del output
                        candidate_observations.append((candidate, statistics.median(timings)))
                        candidate_peaks[candidate] = (
                            int(torch.cuda.max_memory_allocated(device)),
                            int(torch.cuda.max_memory_reserved(device)),
                        )
                    selected_batch_size = select_fastest_observed_batch(
                        tuple(candidate_observations),
                        baseline_batch=legacy_batch_size,
                    )
                    candidate_seconds = dict(candidate_observations)
                    legacy_wall_seconds = candidate_seconds[legacy_batch_size]
                    selected_wall_seconds = candidate_seconds[selected_batch_size]
                    legacy_peak_allocated, legacy_peak_reserved = candidate_peaks[legacy_batch_size]
                    forward_peak_allocated, forward_peak_reserved = candidate_peaks[batch_size]
                    forward_admitted = forward_stage.predicted_gpu_bytes + forward_stage.uncertainty_bytes
                    forward_probe = {
                        "passed": (
                            forward_peak_allocated <= forward_admitted
                            and forward_peak_reserved <= forward_stage.gpu_capacity_bytes
                        ),
                        "batch_size": batch_size,
                        "selected_batch_size": selected_batch_size,
                        "wall_seconds": selected_wall_seconds,
                        "benchmark_samples": benchmark_samples,
                        "legacy_batch_size": legacy_batch_size,
                        "legacy_wall_seconds": legacy_wall_seconds,
                        "legacy_peak_allocated_bytes": legacy_peak_allocated,
                        "legacy_peak_reserved_bytes": legacy_peak_reserved,
                        "speedup": legacy_wall_seconds / selected_wall_seconds,
                        "candidate_observations": [
                            {"batch_size": candidate, "seconds": seconds}
                            for candidate, seconds in candidate_observations
                        ],
                        "peak_allocated_bytes": forward_peak_allocated,
                        "peak_reserved_bytes": forward_peak_reserved,
                        "admitted_bytes": forward_admitted,
                        "capacity_bytes": forward_stage.gpu_capacity_bytes,
                        "output_bytes": initial_inputs.numel() * initial_inputs.element_size(),
                    }
                    tuning_stage = plan.stage("tuning")
                    tuning_maximum = tuning_stage.batch_size
                    tuning_baseline = min(
                        tuning_maximum,
                        case.template.block_tuning.microbatch_size or tuning_maximum,
                    )
                    tuning_samples = min(initial_inputs.shape[0], tuning_maximum)
                    original_requires_grad = {
                        id(parameter): parameter.requires_grad for parameter in target_block.parameters()
                    }
                    for parameter in target_block.parameters():
                        parameter.requires_grad_(True)
                    target_values = _run_block_batched(
                        adapter,
                        target_block,
                        initial_inputs[:tuning_samples],
                        metadata,
                        tuning_baseline,
                        "cpu",
                    ).detach()
                    tuning_observations: list[tuple[int, float]] = []
                    tuning_peaks: dict[int, tuple[int, int]] = {}
                    try:
                        def benchmark_tuning_candidate(candidate: int) -> float:
                            target_block.zero_grad(set_to_none=True)
                            candidate_started = time.perf_counter()
                            for start in range(0, tuning_samples, candidate):
                                stop = min(start + candidate, tuning_samples)
                                input_batch = initial_inputs[start:stop].to(device)
                                target_batch = target_values[start:stop].to(device)
                                prediction = adapter.run_block(target_block, input_batch, **metadata)
                                loss = (prediction.float() - target_batch.float()).square().mean()
                                loss.backward()
                                del input_batch, target_batch, prediction, loss
                            torch.cuda.synchronize(device)
                            return time.perf_counter() - candidate_started

                        benchmark_tuning_candidate(tuning_baseline)
                        for candidate in throughput_batch_candidates(tuning_maximum, tuning_baseline):
                            timings = []
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats(device)
                            for _ in range(_THROUGHPUT_PROBE_REPETITIONS):
                                timings.append(benchmark_tuning_candidate(candidate))
                            tuning_observations.append((candidate, statistics.median(timings)))
                            tuning_peaks[candidate] = (
                                int(torch.cuda.max_memory_allocated(device)),
                                int(torch.cuda.max_memory_reserved(device)),
                            )
                        selected_tuning = select_fastest_observed_batch(
                            tuple(tuning_observations),
                            baseline_batch=tuning_baseline,
                        )
                        tuning_allocated, tuning_reserved = tuning_peaks[tuning_maximum]
                        tuning_admitted = tuning_stage.predicted_gpu_bytes + tuning_stage.uncertainty_bytes
                        tuning_seconds = dict(tuning_observations)
                        tuning_probe = {
                            "passed": (
                                tuning_allocated <= tuning_admitted
                                and tuning_reserved <= tuning_stage.gpu_capacity_bytes
                            ),
                            "maximum_safe_batch_size": tuning_maximum,
                            "baseline_batch_size": tuning_baseline,
                            "selected_batch_size": selected_tuning,
                            "peak_allocated_bytes": tuning_allocated,
                            "peak_reserved_bytes": tuning_reserved,
                            "admitted_bytes": tuning_admitted,
                            "capacity_bytes": tuning_stage.gpu_capacity_bytes,
                            "baseline_wall_seconds": tuning_seconds[tuning_baseline],
                            "selected_wall_seconds": tuning_seconds[selected_tuning],
                            "speedup": tuning_seconds[tuning_baseline] / tuning_seconds[selected_tuning],
                            "candidate_observations": [
                                {"batch_size": candidate, "seconds": seconds}
                                for candidate, seconds in tuning_observations
                            ],
                        }
                    finally:
                        target_block.zero_grad(set_to_none=True)
                        for parameter in target_block.parameters():
                            parameter.requires_grad_(original_requires_grad[id(parameter)])
                        target_values = None
            finally:
                metadata = None
                target_block = None
                initial_inputs = None
                tokens = None
                capture = None
                decoder_layers = None
                block = None
                loaded_model = None
                model = None
                gc.collect()
                torch.cuda.empty_cache()
    except BaseException as exc:
        failure = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "wall_seconds": time.perf_counter() - started,
        }
        del exc
        with acquire_device_lease(device):
            gc.collect()
            torch.cuda.empty_cache()
        return failure
    admitted = stage.predicted_gpu_bytes + stage.uncertainty_bytes
    return {
        "passed": peak_allocated <= admitted and peak_reserved <= stage.gpu_capacity_bytes,
        "executor": plan.executor,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "admitted_bytes": admitted,
        "capacity_bytes": stage.gpu_capacity_bytes,
        "working_set_delta_bytes": int(observed.get("host.working_set_bytes", 0))
        - int(before.get("host.working_set_bytes", 0)),
        "observation": observed,
        "block_forward_probe": forward_probe,
        "tuning_probe": tuning_probe,
    }


def build_matrix(
    repository: Path,
    cache_root: Path,
    profile: MemoryPolicyProfile,
    *,
    probe_model_load: bool = False,
    probe_block_forward: bool = False,
) -> dict[str, object]:
    # Exercise the host's largest ordinary temporary workspace rather than
    # coupling a memory benchmark to the repository drive's current free space.
    scratch = Path(tempfile.gettempdir()) / "nanoquant-adaptive-memory-matrix"
    scratch.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    plans: list[ResolvedMemoryPlan] = []
    snapshots: list[Path] = []
    for case in CASES:
        config = _adaptive_config(case, profile)
        snapshot = _snapshot(cache_root, case)
        retain_completed = config.evaluation.inline_quality
        plan = build_resident_memory_plan(
            config,
            snapshot,
            scratch,
            (config.calibration.sample_count, config.model.sequence_length),
            retain_completed_blocks=retain_completed,
        )
        row = {
            "model": case.source,
            "revision": case.revision,
            "snapshot": str(snapshot),
            "retain_completed_blocks": retain_completed,
            "plan": to_dict(plan),
            "summary": _summary(plan),
            "retained_observation": _retained_observation(repository, case),
        }
        plans.append(plan)
        snapshots.append(snapshot)
        rows.append(row)
    if probe_model_load:
        for case, snapshot, plan, row in zip(CASES, snapshots, plans, rows, strict=True):
            row["model_load_probe"] = _probe_model_load(
                case,
                snapshot,
                plan,
                probe_block_forward=probe_block_forward,
            )
    return {
        "schema_version": 1,
        "kind": "adaptive-memory-real-model-matrix",
        "profile": profile.value,
        "planning_directory": str(scratch.resolve()),
        "model_load_probed": probe_model_load,
        "block_forward_probed": probe_block_forward,
        "models": rows,
    }


def _gib(value: object) -> str:
    return f"{int(value) / 2**30:.2f}"


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Adaptive Memory Real-Model Matrix",
        "",
        f"Profile: `{payload['profile']}`",
        "",
        f"Planning workspace: `{payload['planning_directory']}`",
        "",
        "This metadata preflight is compared with retained real execution events. The 270M observation is an "
        "architecture-equivalent Unsloth mirror; the other three observations use the exact requested source and "
        "revision. Interrupted runs remain valid bounded memory evidence, not completion or quality evidence.",
        "",
        "| Model | Real blocks | Real peak reserved GiB | Executor | Safe forward max | Safe tuning max | Cache | "
        "Admitted / capacity GiB |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for row_value in payload["models"]:
        row = row_value if isinstance(row_value, dict) else {}
        summary = row.get("summary", {})
        observation = row.get("retained_observation", {})
        batches = summary.get("stage_batches", {})
        lines.append(
            f"| `{row['model']}` | {observation['completed_blocks']} | "
            f"{_gib(observation['cuda_peak_reserved_bytes'])} | `{summary['executor']}` | "
            f"{batches['block_forward']} | {batches['tuning']} | `{summary['activation_gpu_cache']}` | "
            f"{_gib(summary['admitted_peak_gpu_bytes'])} / {_gib(summary['safe_gpu_capacity_bytes'])} |"
        )
    lines.extend(
        [
            "",
            "The admitted peak includes the stage prediction, profile uncertainty, and selected activation cache. "
            "The capacity already excludes the configured CUDA reserve. Full plan contracts, envelopes, warnings, "
            "and retained host/WDDM counters are in the adjacent JSON file.",
            "",
        ]
    )
    if payload.get("model_load_probed"):
        lines.extend(
            [
                "## Leased model-load probes",
                "",
                "| Model | Placement | Allocated / admitted GiB | Reserved / capacity GiB | Result | Seconds |",
                "| --- | --- | ---: | ---: | --- | ---: |",
            ]
        )
        for row_value in payload["models"]:
            row = row_value if isinstance(row_value, dict) else {}
            probe = row.get("model_load_probe", {})
            lines.append(
                f"| `{row['model']}` | `{probe.get('executor', 'unknown')}` | "
                f"{_gib(probe.get('peak_allocated_bytes', 0))} / {_gib(probe.get('admitted_bytes', 0))} | "
                f"{_gib(probe.get('peak_reserved_bytes', 0))} / {_gib(probe.get('capacity_bytes', 0))} | "
                f"{'pass' if probe.get('passed') else 'FAIL'} | {float(probe.get('wall_seconds', 0)):.1f} |"
            )
        lines.append("")
    if payload.get("block_forward_probed"):
        lines.extend(
            [
                "## Leased block-forward probes",
                "",
                "Each timing processes the same 64-sample, full-sequence workload after a warm-up pass.",
                "",
                "| Model | Fixed -> selected batch (safe max) | Max allocated / admitted GiB | "
                "Max reserved / capacity GiB | "
                "Result | Fixed / adaptive seconds | Speedup |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row_value in payload["models"]:
            row = row_value if isinstance(row_value, dict) else {}
            load_probe = row.get("model_load_probe", {})
            probe = load_probe.get("block_forward_probe", {})
            lines.append(
                f"| `{row['model']}` | {probe.get('legacy_batch_size', 0)} -> "
                f"{probe.get('selected_batch_size', 0)} ({probe.get('batch_size', 0)}) | "
                f"{_gib(probe.get('peak_allocated_bytes', 0))} / {_gib(probe.get('admitted_bytes', 0))} | "
                f"{_gib(probe.get('peak_reserved_bytes', 0))} / {_gib(probe.get('capacity_bytes', 0))} | "
                f"{'pass' if probe.get('passed') else 'FAIL'} | "
                f"{float(probe.get('legacy_wall_seconds', 0)):.2f} / "
                f"{float(probe.get('wall_seconds', 0)):.2f} | "
                f"{float(probe.get('speedup', 0)):.2f}x |"
            )
        lines.append("")
        lines.extend(
            [
                "## Leased tuning forward/backward probes",
                "",
                "Each timing executes five medianed forward/backward passes over one logical tuning batch; "
                "no optimizer step changes model weights.",
                "",
                "| Model | Fixed -> selected microbatch (safe max) | Max allocated / admitted GiB | "
                "Max reserved / capacity GiB | Result | Fixed / adaptive seconds | Speedup |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row_value in payload["models"]:
            row = row_value if isinstance(row_value, dict) else {}
            load_probe = row.get("model_load_probe", {})
            probe = load_probe.get("tuning_probe", {})
            lines.append(
                f"| `{row['model']}` | {probe.get('baseline_batch_size', 0)} -> "
                f"{probe.get('selected_batch_size', 0)} ({probe.get('maximum_safe_batch_size', 0)}) | "
                f"{_gib(probe.get('peak_allocated_bytes', 0))} / {_gib(probe.get('admitted_bytes', 0))} | "
                f"{_gib(probe.get('peak_reserved_bytes', 0))} / {_gib(probe.get('capacity_bytes', 0))} | "
                f"{'pass' if probe.get('passed') else 'FAIL'} | "
                f"{float(probe.get('baseline_wall_seconds', 0)):.2f} / "
                f"{float(probe.get('selected_wall_seconds', 0)):.2f} | "
                f"{float(probe.get('speedup', 0)):.2f}x |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence path")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in MemoryPolicyProfile),
        default=MemoryPolicyProfile.BALANCED.value,
    )
    parser.add_argument("--probe-model-load", action="store_true")
    parser.add_argument("--probe-block-forward", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    payload = build_matrix(
        repository,
        args.cache_root.resolve(),
        MemoryPolicyProfile(args.profile),
        probe_model_load=args.probe_model_load or args.probe_block_forward,
        probe_block_forward=args.probe_block_forward,
    )
    output = args.output if args.output.is_absolute() else repository / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    print(output)
    probes = [row.get("model_load_probe") for row in payload["models"] if isinstance(row, dict)]
    load_failed = any(isinstance(probe, dict) and not probe.get("passed") for probe in probes)
    forward_failed = any(
        isinstance(probe, dict)
        and isinstance(probe.get("block_forward_probe"), dict)
        and not probe["block_forward_probe"].get("passed")
        for probe in probes
    )
    tuning_failed = any(
        isinstance(probe, dict)
        and isinstance(probe.get("tuning_probe"), dict)
        and not probe["tuning_probe"].get("passed")
        for probe in probes
    )
    return 1 if load_failed or forward_failed or tuning_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
