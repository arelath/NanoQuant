"""Validate self-contained packed-bundle loading and deterministic generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
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


def _eos_token_ids(value: int | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)) and value:
        return tuple(int(item) for item in value)
    raise ValueError("runtime bundle model contains no EOS token ID")


def _token_hash(tokens: torch.Tensor, lengths: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    host = tokens.detach().cpu()
    for row, length in enumerate(lengths):
        digest.update(int(length).to_bytes(8, "little"))
        digest.update(host[row, :length].contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def _validate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("runtime bundle CUDA validation requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    started = time.perf_counter()
    bundle = open_runtime_bundle(args.bundle, verify_hashes=True)
    tokenizer = bundle.load_tokenizer()
    prompt_ids = tuple(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for prompt in args.prompt
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("runtime bundle tokenizer has no pad token ID")
    input_ids, attention_mask = batch_prompts(prompt_ids, pad_token_id=pad_token_id)
    batch_size, prompt_width = input_ids.shape

    with wait_for_device_lease(str(device), args.wait_for_device_seconds):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        loaded = load_transformers_runtime(
            bundle,
            CudaPackedBackend(),
            device=device,
            input_dtype=args.input_dtype,
            batch_size=batch_size,
            prefill_tokens=prompt_width,
        )
        model = loaded.model
        request = GenerationRequest(
            input_ids.to(device),
            attention_mask.to(device),
            args.max_new_tokens,
            (
                (int(model.config.vocab_size),)
                if args.ignore_eos
                else _eos_token_ids(model.config.eos_token_id)
            ),
            pad_token_id,
            sampling=SamplingConfig(mode="greedy"),
            stopping_check_interval=args.stopping_check_interval,
        )
        created_caches: list[object] = []
        cache_factory = hybrid_cache_factory(model.config)

        def capture_cache(
            batch: int,
            length: int,
            target: torch.device,
            dtype: torch.dtype,
        ) -> object:
            cache = cache_factory(batch, length, target, dtype)
            created_caches.append(cache)
            return cache

        shell = TransformersGenerationModel(model, capture_cache)
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        first = generate(request, shell)
        torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_started
        first_allocated = torch.cuda.memory_allocated(device)
        second = generate(request, shell)
        torch.cuda.synchronize(device)
        second_allocated = torch.cuda.memory_allocated(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)

    if not torch.equal(first.token_ids, second.token_ids):
        raise ValueError("runtime bundle deterministic token replay differs")
    if first.lengths != second.lengths or first.stop_reasons != second.stop_reasons:
        raise ValueError("runtime bundle deterministic stopping replay differs")
    generated_ids = [
        first.token_ids[row, :length].detach().cpu().tolist()
        for row, length in enumerate(first.lengths)
    ]
    generated_text = [tokenizer.decode(tokens) for tokens in generated_ids]
    reference_prefix: dict[str, Any] | None = None
    if args.reference_output is not None:
        reference = json.loads(args.reference_output.read_text(encoding="utf-8"))
        expected_prompt = reference.get("prompt")
        expected_text = reference.get("generated")
        expected_count = reference.get("generated_token_count")
        if (
            expected_prompt != args.prompt[0]
            or not isinstance(expected_text, str)
            or not isinstance(expected_count, int)
            or expected_count <= 0
        ):
            raise ValueError("runtime bundle reference output is invalid")
        actual_prefix = tokenizer.decode(generated_ids[0][:expected_count])
        if actual_prefix != expected_text:
            raise ValueError(
                f"runtime bundle reference prefix differs: {actual_prefix!r} != {expected_text!r}"
            )
        reference_prefix = {
            "path": str(args.reference_output.resolve()),
            "token_count": expected_count,
            "text": expected_text,
            "exact": True,
        }
    descriptor = bundle.root / "nanoquant-runtime-bundle.json"
    return {
        "schema_version": 1,
        "bundle": str(bundle.root),
        "bundle_descriptor_sha256": hash_file(descriptor),
        "bundle_member_count": len(bundle.manifest.members),
        "bundle_member_bytes": bundle.manifest.total_member_bytes,
        "shell_tensor_count": len(bundle.manifest.shell_tensors),
        "excluded_linear_count": len(bundle.manifest.excluded_linear_modules),
        "packed_layer_count": bundle.packed.manifest.layer_count,
        "device": torch.cuda.get_device_name(device),
        "input_dtype": args.input_dtype,
        "batch_size": batch_size,
        "prompt_lengths": [len(item) for item in prompt_ids],
        "prompt_width": prompt_width,
        "max_new_tokens": args.max_new_tokens,
        "replaced_linear_count": loaded.replaced_linear_count,
        "fused_rms_norm_count": loaded.fused_rms_norm_count,
        "fused_decode_rope_count": loaded.fused_decode_rope_count,
        "fused_decode_attention_count": loaded.fused_decode_attention_count,
        "grouped_decode_qkv_count": loaded.grouped_decode_qkv_count,
        "short_sliding_mask_count": loaded.short_sliding_mask_count,
        "native_bfloat16_tied_projection_count": (
            loaded.native_bfloat16_tied_projection_count
        ),
        "fast_sliding_update_count": sum(
            int(getattr(cache, "nanoquant_fast_sliding_update_count", 0))
            for cache in created_caches
        ),
        "prefill_fallback_count": loaded.plans.prefill.plan.fallback_count,
        "decode_fallback_count": loaded.plans.decode.plan.fallback_count,
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "generated_token_sha256": _token_hash(first.token_ids, first.lengths),
        "reference_prefix": reference_prefix,
        "stop_reasons": list(first.stop_reasons),
        "maximum_cache_length": first.maximum_cache_length,
        "generation_seconds": generation_seconds,
        "first_allocated_bytes": first_allocated,
        "second_allocated_bytes": second_allocated,
        "peak_allocated_bytes": peak_allocated,
        "deterministic_replay": True,
        "wall_seconds": time.perf_counter() - started,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stopping-check-interval", type=int, default=8)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.prompt:
        args.prompt = ["Write a short paragraph about quantization."]
    if args.max_new_tokens <= 0 or args.stopping_check_interval <= 0:
        parser.error("generation lengths and stopping interval must be positive")
    result = _validate(args)
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
