"""Evaluate exact chunked-prefill parity on a self-contained runtime bundle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from nanoquant.application.long_context_evaluation import (
    LongContextCase,
    LongContextEvaluationRequest,
    evaluate_long_context,
)
from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.runtime_long_context import (
    gemma3_hybrid_long_context_protocol,
    make_runtime_long_context_generator,
)
from nanoquant.runtime import (
    CudaPackedBackend,
    GenerationRequest,
    SamplingConfig,
    TransformersGenerationModel,
    batch_prompts,
    generate,
    hybrid_cache_factory,
    load_transformers_runtime,
    open_runtime_bundle,
)


def _token_hash(tokens: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _build_prompt_tokens(
    tokenizer: Any,
    prompt_unit: str,
    minimum_tokens: int,
) -> tuple[int, ...]:
    repetitions = 1
    while True:
        prompt = " ".join(prompt_unit for _ in range(repetitions))
        token_ids = tuple(
            int(token)
            for token in tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if len(token_ids) >= minimum_tokens:
            return token_ids
        repetitions = max(repetitions + 1, math.ceil(repetitions * minimum_tokens / len(token_ids)))


def _eos_token_ids(value: int | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)) and value:
        return tuple(int(item) for item in value)
    raise ValueError("runtime bundle model contains no EOS token ID")


@torch.inference_mode()
def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("packed long-context evaluation requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    started = time.perf_counter()
    bundle = open_runtime_bundle(args.bundle, verify_hashes=True)
    tokenizer = bundle.load_tokenizer()
    prompt_tokens = _build_prompt_tokens(tokenizer, args.prompt_unit, args.minimum_prompt_tokens)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("runtime bundle tokenizer has no pad token ID")

    with wait_for_device_lease(str(device), args.wait_for_device_seconds):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        loaded = load_transformers_runtime(
            bundle,
            CudaPackedBackend(),
            device=device,
            input_dtype=args.input_dtype,
            batch_size=1,
            prefill_tokens=len(prompt_tokens),
        )
        model = loaded.model
        protocol = gemma3_hybrid_long_context_protocol(
            model.config,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        total_tokens = len(prompt_tokens) + args.max_new_tokens
        if total_tokens > protocol.maximum_context_length:
            raise ValueError(
                f"long-context request requires {total_tokens} tokens; "
                f"model limit is {protocol.maximum_context_length}"
            )
        if len(prompt_tokens) <= protocol.sliding_window:
            raise ValueError(
                f"prompt has {len(prompt_tokens)} tokens and does not cross the "
                f"{protocol.sliding_window}-token sliding window"
            )
        if len(prompt_tokens) <= protocol.prefill_chunk_size:
            raise ValueError("long-context prompt must exercise multiple prefill chunks")

        eos_token_ids = (
            (int(model.config.vocab_size),)
            if args.ignore_eos
            else _eos_token_ids(model.config.eos_token_id)
        )
        shell = TransformersGenerationModel(model, hybrid_cache_factory(model.config))
        input_ids, attention_mask = batch_prompts(
            (prompt_tokens,),
            pad_token_id=pad_token_id,
            device=device,
        )
        reference_request = GenerationRequest(
            input_ids,
            attention_mask,
            args.max_new_tokens,
            eos_token_ids,
            pad_token_id,
            sampling=SamplingConfig(mode="greedy"),
            stopping_check_interval=args.stopping_check_interval,
            prefill_chunk_size=args.reference_prefill_chunk_size,
        )
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        reference_started = time.perf_counter()
        reference = generate(reference_request, shell)
        torch.cuda.synchronize(device)
        reference_seconds = time.perf_counter() - reference_started
        reference_peak = torch.cuda.max_memory_allocated(device)
        reference_length = reference.lengths[0]
        expected_tokens = tuple(
            int(token) for token in reference.token_ids[0, :reference_length].tolist()
        )
        expected_stop = reference.stop_reasons[0]

        del reference, reference_request, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()
        case = LongContextCase(
            "packed-gemma-sliding-window-rollover",
            "1",
            prompt_tokens,
            expected_tokens,
            expected_stop,
        )
        runtime_generate = make_runtime_long_context_generator(
            shell,
            device=device,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
            stopping_check_interval=args.stopping_check_interval,
            profile_device_memory=True,
        )
        candidate_started = time.perf_counter()
        evaluation = evaluate_long_context(
            LongContextEvaluationRequest(protocol, (case,)),
            runtime_generate,
        )
        torch.cuda.synchronize(device)
        candidate_seconds = time.perf_counter() - candidate_started

    descriptor = bundle.root / "nanoquant-runtime-bundle.json"
    return {
        "schema_version": 1,
        "bundle": str(bundle.root),
        "bundle_descriptor_sha256": hash_file(descriptor),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "input_dtype": args.input_dtype,
        "prompt_unit": args.prompt_unit,
        "prompt_token_count": len(prompt_tokens),
        "prompt_token_sha256": _token_hash(prompt_tokens),
        "expected_token_ids": list(expected_tokens),
        "expected_token_sha256": _token_hash(expected_tokens),
        "expected_text": tokenizer.decode(expected_tokens),
        "expected_stop_reason": expected_stop,
        "reference": {
            "prefill_chunk_size": args.reference_prefill_chunk_size,
            "prefill_forward_count": (
                1
                if args.reference_prefill_chunk_size is None
                else math.ceil(len(prompt_tokens) / args.reference_prefill_chunk_size)
            ),
            "decode_forward_count": max(0, len(expected_tokens) - 1),
            "maximum_cache_length": total_tokens,
            "seconds": reference_seconds,
            "peak_device_bytes": reference_peak,
        },
        "protocol": asdict(protocol),
        "evaluation": asdict(evaluation),
        "candidate_seconds": candidate_seconds,
        "replaced_linear_count": loaded.replaced_linear_count,
        "prefill_fallback_count": loaded.plans.prefill.plan.fallback_count,
        "decode_fallback_count": loaded.plans.decode.plan.fallback_count,
        "fused_rms_norm_count": loaded.fused_rms_norm_count,
        "fused_decode_rope_count": loaded.fused_decode_rope_count,
        "fused_decode_attention_count": loaded.fused_decode_attention_count,
        "grouped_decode_qkv_count": loaded.grouped_decode_qkv_count,
        "grouped_decode_mlp_count": loaded.grouped_decode_mlp_count,
        "short_sliding_mask_count": loaded.short_sliding_mask_count,
        "native_bfloat16_tied_projection_count": (
            loaded.native_bfloat16_tied_projection_count
        ),
        "wall_seconds": time.perf_counter() - started,
        "passed": evaluation.passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--input-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float32",
    )
    parser.add_argument("--minimum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument(
        "--reference-prefill-chunk-size",
        type=int,
        help="bound the oracle prefill independently; the default uses one monolithic forward",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--stopping-check-interval", type=int, default=1)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prompt-unit",
        default="Explain one practical benefit of model compression.",
    )
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.minimum_prompt_tokens,
        args.prefill_chunk_size,
        args.max_new_tokens,
        args.stopping_check_interval,
    ) <= 0:
        parser.error("prompt, chunk, generation, and stopping lengths must be positive")
    if args.reference_prefill_chunk_size is not None and args.reference_prefill_chunk_size <= 0:
        parser.error("reference prefill chunk size must be positive when configured")
    result = _evaluate(args)
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
