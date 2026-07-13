"""Run the pinned Experiment 018 top-k model-level KD protocol on a complete frozen run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from nanoquant.application.distillation import TopKDistillationConfig
from nanoquant.domain.models import ArtifactRef
from nanoquant.global_distillation import GlobalDistillationRequest, run_global_topk_distillation
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
CALIBRATION_ARTIFACT = "sha256-ad1f609729f86db7598eed5c703c55aacbb9cb024cab816ca7b300d574b7a4c8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("evidence/m3/experiment018-calibration"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--maximum-tokens-per-batch", type=int, default=512)
    parser.add_argument("--vocabulary-chunk-size", type=int, default=8192)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
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
    args = parser.parse_args()

    calibration = load_pinned_calibration(
        args.calibration,
        ArtifactRef("calibration-dataset-manifest", CALIBRATION_ARTIFACT, 1),
    )
    if args.samples <= 0 or args.samples > calibration.input_ids.shape[0]:
        raise ValueError("distillation sample count is outside the pinned calibration dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    request = GlobalDistillationRequest(
        args.run_output,
        args.snapshot,
        "google/gemma-3-1b-it",
        MODEL_REVISION,
        calibration.input_ids[: args.samples],
        TopKDistillationConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            top_k=args.top_k,
            vocabulary_chunk_size=args.vocabulary_chunk_size,
            token_chunk_size=args.token_chunk_size,
            maximum_tokens_per_batch=args.maximum_tokens_per_batch,
            gradient_checkpointing=True,
            # The legacy Optimi AdamW invocation leaves weight decay at its
            # zero default. Keep this explicit in the pinned protocol.
            weight_decay=0.0,
            seed=0,
        ),
        device=args.device,
        pad_token_id=tokenizer.pad_token_id,
        replace_existing_global_tuning=args.replace_global_tuning,
        interrupt_after_epoch_commits=args.interrupt_after_epoch_commits,
    )
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
