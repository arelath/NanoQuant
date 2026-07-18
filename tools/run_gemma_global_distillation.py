"""Run the pinned Gemma 3 1B top-k model-level KD protocol on a complete frozen run."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import _paths  # noqa: F401
from recipes import BASE_COMPRESSION_TEMPLATE
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.global_distillation import run_global_topk_distillation
from nanoquant.infrastructure.hf_calibration_dataset import load_or_prepare_calibration
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    distillation_request_from_config,
)

MODEL_REVISION = str(BASE_COMPRESSION_TEMPLATE.model.revision)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--maximum-tokens-per-batch", type=int, default=512)
    parser.add_argument("--vocabulary-chunk-size", type=int, default=8192)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--block-snapshot-samples", type=int, default=4)
    parser.add_argument("--block-snapshot-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--profile",
        choices=(ProfilingLevel.OFF.value, ProfilingLevel.MACRO.value, ProfilingLevel.MICRO.value),
        default=ProfilingLevel.MACRO.value,
    )
    parser.add_argument("--profile-cuda-timing", action="store_true")
    parser.add_argument("--profile-cuda-sample-every", type=int, default=16)
    parser.add_argument("--profile-memory-counters", action="store_true")
    parser.add_argument(
        "--replace-global-tuning",
        action="store_true",
        help="Start from immutable pre-KD commits and atomically replace the active tuned result.",
    )
    parser.add_argument(
        "--interrupt-after-epoch-commits",
        type=int,
        help="Exit cleanly after this many new durable epoch checkpoints; resume with the same command.",
    )
    parser.add_argument(
        "--initial-cooldown-seconds",
        type=float,
        default=0.0,
        help="Acquire the CUDA lease, then idle before loading the frozen model.",
    )
    parser.add_argument(
        "--epoch-cooldown-seconds",
        type=float,
        default=0.0,
        help="Idle after each non-final durable epoch checkpoint while retaining the CUDA lease.",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("distillation sample count must be positive")
    calibration = load_or_prepare_calibration(
        args.snapshot,
        args.run_output,
        sample_count=args.samples,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    base = BASE_COMPRESSION_TEMPLATE
    config = replace(
        base,
        intent=replace(base.intent, experiment_number=None, name=args.run_output.name),
        calibration=replace(base.calibration, sample_count=args.samples),
        distillation=replace(
            base.distillation,
            enabled=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            top_k=args.top_k,
            vocabulary_chunk_size=args.vocabulary_chunk_size,
            token_chunk_size=args.token_chunk_size,
            maximum_tokens_per_batch=args.maximum_tokens_per_batch,
            gradient_checkpointing=True,
            weight_decay=0.0,
        ),
        runtime=replace(base.runtime, compute_device=args.device),
        profiling=ProfilingConfig(
            level=ProfilingLevel(args.profile),
            cuda_timing=args.profile_cuda_timing,
            cuda_sample_every=args.profile_cuda_sample_every,
            memory_counters=args.profile_memory_counters,
            emit_span_events=False,
        ),
        observability=replace(
            base.observability,
            block_snapshot_samples=args.block_snapshot_samples,
            block_snapshot_tokens=args.block_snapshot_tokens,
        ),
    )
    inputs = ResolvedResidentInputs(
        snapshot=args.snapshot,
        output=args.run_output,
        registry_root=args.run_output.parent,
        token_ids=calibration.input_ids,
        quality_token_ids=None,
        launcher_path=Path(__file__),
        pad_token_id=tokenizer.pad_token_id,
    )
    options = ResidentExecutionOptions(
        replace_existing_global_tuning=args.replace_global_tuning,
        interrupt_after_distillation_epoch_commits=args.interrupt_after_epoch_commits,
        distillation_initial_cooldown_seconds=args.initial_cooldown_seconds,
        distillation_epoch_cooldown_seconds=args.epoch_cooldown_seconds,
    )
    request = distillation_request_from_config(config, inputs, options)
    try:
        result = run_global_topk_distillation(request)
    except InterruptedError as exc:
        print(json.dumps({"status": "interrupted", "reason": str(exc)}, indent=2))
        return
    print(
        json.dumps(
            {
                "artifact": result.reference.artifact_id,
                "epoch_losses": result.metrics.epoch_losses,
                "steps_completed": result.metrics.steps_completed,
                "selected_parameter_count": result.metrics.selected_parameter_count,
                "teacher_cache_bytes": result.metrics.teacher_cache_bytes,
                "wall_seconds": result.result.wall_seconds,
                "peak_gpu_bytes": result.result.peak_gpu_bytes,
                "peak_host_bytes": result.result.peak_host_bytes,
                "block_snapshot_protocol_hash": result.result.block_snapshot_protocol_hash,
                "block_metrics": [
                    {
                        "block": item.block.index,
                        "final_frozen_pre_kd": item.final_frozen_pre_kd,
                        "final_post_kd": item.final_post_kd,
                        "absolute_delta": item.post_kd_vs_pre_kd.absolute_delta,
                        "relative_delta": item.post_kd_vs_pre_kd.relative_delta,
                    }
                    for item in result.result.block_metrics
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
