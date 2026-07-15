"""Evaluate a base or committed frozen model with the retained WikiText-2 protocol."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanoquant.application.evaluation import CausalEvaluationRequest, evaluate_causal_nll, model_logits
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.profiling import profiled_run
from nanoquant.infrastructure.resource_usage import peak_device_memory_bytes, peak_process_memory_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation window batch size; values above 1 are faster but approximate for parity.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("factorized", "dense"), default="factorized")
    parser.add_argument(
        "--dataset-arrow",
        type=Path,
        help="Previously materialized WikiText test Arrow file; bypasses dataset-builder startup.",
    )
    parser.add_argument(
        "--ignore-global-tuning",
        action="store_true",
        help="Evaluate immutable pre-KD block commits even when a global-tuning artifact is active.",
    )
    parser.add_argument("--evaluate-base", action="store_true")
    parser.add_argument(
        "--profile",
        choices=(ProfilingLevel.OFF.value, ProfilingLevel.MACRO.value, ProfilingLevel.MICRO.value),
        default=ProfilingLevel.MACRO.value,
    )
    parser.add_argument("--profile-cuda-timing", action="store_true")
    parser.add_argument("--profile-cuda-sample-every", type=int, default=16)
    parser.add_argument("--profile-memory-counters", action="store_true")
    return parser


def _checkpoint_dtype(snapshot: Path) -> torch.dtype:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(config.get("torch_dtype"), torch.float32)


def _protocol_tokens(
    snapshot: Path,
    samples: int,
    sequence_length: int,
    dataset_arrow: Path | None = None,
) -> tuple[torch.Tensor, str, int]:
    if samples <= 0 or sequence_length < 2:
        raise ValueError("samples must be positive and sequence length must be at least two")
    dataset = (
        Dataset.from_file(str(dataset_arrow))
        if dataset_arrow is not None
        else load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("Gemma WikiText protocol requires a BOS token")
    payload = sequence_length - 1
    required = samples * payload
    if encoded.shape[1] < required:
        raise ValueError(f"WikiText token stream has {encoded.shape[1]} tokens; protocol requires {required}")
    rows = []
    for index in range(samples):
        chunk = encoded[:, index * payload : (index + 1) * payload]
        rows.append(torch.cat((torch.tensor([[bos_id]], dtype=chunk.dtype), chunk), dim=1))
    tokens = torch.cat(rows, dim=0)
    fingerprint = str(getattr(dataset, "_fingerprint", "unknown"))
    return tokens, fingerprint, int(bos_id)


def _evaluate(
    model: nn.Module,
    tokens: torch.Tensor,
    device: str,
    batch_size: int,
    recorder: PhaseRecorder = NULL_RECORDER,
) -> dict[str, object]:
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with recorder.phase("tokens_to_device"):
        request = CausalEvaluationRequest(
            tokens.to(device),
            max_length=tokens.shape[1],
            stride=tokens.shape[1],
            batch_size=batch_size,
        )
    with recorder.phase("causal_nll"):
        result = evaluate_causal_nll(request, model_logits(model))
        recorder.add("evaluation.tokens", result.token_count)
        recorder.add("evaluation.windows", result.window_count)
    return {
        "total_negative_log_likelihood": result.total_negative_log_likelihood,
        "mean_negative_log_likelihood": result.mean_negative_log_likelihood,
        "perplexity": result.perplexity,
        "token_count": result.token_count,
        "window_count": result.window_count,
        "sample_count": result.sample_count,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_device_bytes": peak_device_memory_bytes(device),
        "peak_host_bytes": peak_process_memory_bytes(),
    }


def _run(args: argparse.Namespace, recorder: PhaseRecorder) -> None:
    with recorder.phase("dataset"):
        tokens, dataset_fingerprint, bos_id = _protocol_tokens(
            args.snapshot,
            args.samples,
            args.sequence_length,
            args.dataset_arrow,
        )
    results: dict[str, object] = {}
    frozen_global_tuning: str | None = None
    if args.evaluate_base:
        with recorder.phase("base_load"):
            base = cast(
                nn.Module,
                AutoModelForCausalLM.from_pretrained(
                    args.snapshot,
                    local_files_only=True,
                    torch_dtype=_checkpoint_dtype(args.snapshot),
                ),
            ).to(args.device)
            base.eval()
            cast(Any, base).config.use_cache = False
        with recorder.phase("base_evaluate"):
            results["base"] = _evaluate(base, tokens, args.device, args.batch_size, recorder)
        with recorder.phase("base_release"):
            del base
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    if args.run_output is not None:
        with recorder.phase("frozen_load"):
            loaded = load_frozen_run(
                args.run_output,
                args.snapshot,
                source_name="google/gemma-3-1b-it",
                revision=args.revision,
                device=args.device,
                backend=args.backend,
                use_global_tuning=not args.ignore_global_tuning,
                recorder=recorder,
            )
        frozen_global_tuning = None if loaded.global_tuning is None else loaded.global_tuning.artifact_id
        with recorder.phase("frozen_evaluate"):
            results["frozen"] = _evaluate(loaded.model, tokens, args.device, args.batch_size, recorder)
        with recorder.phase("frozen_release"):
            del loaded
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "model": "google/gemma-3-1b-it",
        "revision": args.revision,
        "snapshot": str(args.snapshot.resolve()),
        "run_output": None if args.run_output is None else str(args.run_output.resolve()),
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_arrow": None if args.dataset_arrow is None else str(args.dataset_arrow.resolve()),
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "backend": args.backend,
        "global_tuning_artifact": frozen_global_tuning,
        "bos_token_id": bos_id,
        "token_hash": "sha256:" + hashlib.sha256(tokens.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        "results": results,
    }
    with recorder.phase("report"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, indent=2))


def main() -> None:
    args = _parser().parse_args()

    def execute() -> None:
        profiling = ProfilingConfig(
            level=ProfilingLevel(args.profile),
            cuda_timing=args.profile_cuda_timing,
            cuda_sample_every=args.profile_cuda_sample_every,
            memory_counters=args.profile_memory_counters,
        )
        with profiled_run(profiling, args.output.parent, None, run_id="wikitext-evaluation") as recorder:
            with recorder.phase("run"):
                _run(args, recorder)

    if args.device.startswith("cuda"):
        with acquire_device_lease(args.device):
            execute()
    else:
        execute()


if __name__ == "__main__":
    main()
