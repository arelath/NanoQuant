"""Historical short-decode comparison across base, logical, and packed models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.frozen_model_loader import LoadedFrozenModel, load_frozen_run
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes
from nanoquant.runtime import (
    CudaPackedBackend,
    LoadedTransformersRuntime,
    execution_workload,
    load_transformers_runtime,
    summarize_benchmark,
)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_DEFAULT_PROMPT = "Explain why compact language models are useful for local inference."


class _Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def __call__(self, text: str, *, return_tensors: str, padding: bool) -> Any: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class LegacyShortDecodeCase:
    """One retained row from legacy Experiment 002's aggregate CSV."""

    name: str
    throughput_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("legacy short-decode case name must be non-empty")
        if not math.isfinite(self.throughput_per_second) or self.throughput_per_second <= 0:
            raise ValueError("legacy short-decode throughput must be finite and positive")
        if self.peak_allocated_bytes <= 0 or self.peak_reserved_bytes <= 0:
            raise ValueError("legacy short-decode memory peaks must be positive")


@dataclass(frozen=True, slots=True)
class ShortDecodeBenchmarkRequest:
    """Pinned three-case protocol for the legacy Experiment 002 workload."""

    snapshot: Path
    run_output: Path
    runtime_bundle: Path
    source: str
    revision: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    backend: str = "factorized"
    use_global_tuning: bool = True
    prompt: str = _DEFAULT_PROMPT
    prompt_tokens: int = 32
    max_new_tokens: int = 32
    warmups: int = 1
    repetitions: int = 3
    seed: int = 0
    top_k: int = 32
    temperature: float = 0.8
    wait_for_device_seconds: float = 0.0
    legacy_cases: tuple[LegacyShortDecodeCase, ...] = ()
    legacy_summary_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source or not self.revision:
            raise ValueError("short-decode source and revision must be non-empty")
        if self.dtype not in _DTYPES:
            raise ValueError("short-decode dtype is unsupported")
        if self.backend not in {"dense", "factorized"}:
            raise ValueError("short-decode frozen backend is unsupported")
        if not self.prompt:
            raise ValueError("short-decode prompt must be non-empty")
        if self.prompt_tokens <= 0 or self.max_new_tokens <= 1:
            raise ValueError("short-decode prompt/output lengths are invalid")
        if self.warmups < 0 or self.repetitions <= 0:
            raise ValueError("short-decode warmups must be non-negative and repetitions positive")
        if self.seed < 0 or self.top_k <= 0:
            raise ValueError("short-decode seed/top-k values are invalid")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("short-decode temperature must be finite and positive")
        if self.wait_for_device_seconds < 0:
            raise ValueError("short-decode device wait must not be negative")
        names = tuple(case.name for case in self.legacy_cases)
        if len(names) != len(set(names)):
            raise ValueError("legacy short-decode case names must be unique")
        if self.legacy_summary_sha256 is not None and (
            len(self.legacy_summary_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.legacy_summary_sha256)
        ):
            raise ValueError("legacy short-decode summary hash must be lowercase SHA-256")


def _fill_token_id(tokenizer: _Tokenizer) -> int:
    excluded = {
        value
        for value in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if value is not None
    }
    for text in (" hello", " the", " a", ".", "0", "1", "and", "I", "to"):
        for token_id in reversed(tokenizer.encode(text, add_special_tokens=False)):
            if token_id not in excluded:
                return int(token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    return 0


def force_prompt_tokens(tokenizer: _Tokenizer, prompt: str, target_tokens: int) -> torch.Tensor:
    """Reproduce the legacy truncate-or-non-special-fill prompt policy."""

    if target_tokens <= 0:
        raise ValueError("target prompt token count must be positive")
    encoded = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = cast(torch.Tensor, encoded["input_ids"])
    attention_mask = cast(
        torch.Tensor,
        encoded.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long)),
    )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape != attention_mask.shape:
        raise ValueError("short-decode tokenizer returned an invalid prompt batch")
    actual = int(attention_mask.to(dtype=torch.long).sum().item())
    if actual <= 0:
        raise ValueError("short-decode tokenizer returned an empty prompt")
    tokens = input_ids[:, : min(actual, target_tokens)].to(dtype=torch.long)
    if tokens.shape[1] < target_tokens:
        filler = torch.full(
            (1, target_tokens - tokens.shape[1]),
            _fill_token_id(tokenizer),
            dtype=torch.long,
        )
        tokens = torch.cat((tokens, filler), dim=1)
    return tokens


def _sample(logits: torch.Tensor, *, temperature: float, top_k: int) -> torch.Tensor:
    values = logits[:, -1] / max(temperature, 1e-5)
    keep = min(top_k, values.shape[-1])
    threshold = torch.topk(values, keep, dim=-1).values[:, -1, None]
    values = torch.where(values < threshold, -torch.inf, values)
    probabilities = torch.softmax(values, dim=-1)
    exponential = torch.empty_like(probabilities).exponential_(1)
    return torch.argmax(probabilities / exponential, dim=-1, keepdim=True).long()


def _new_static_cache(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    maximum_length: int,
) -> object:
    from transformers import StaticCache

    cache_type = cast(Any, StaticCache)
    return cache_type(
        cast(Any, model).config,
        max_batch_size=1,
        max_cache_len=maximum_length,
        device=device,
        dtype=dtype,
    )


@torch.inference_mode()
def _decode_trial(
    model: nn.Module,
    prompt_tokens: torch.Tensor,
    cache: object,
    request: ShortDecodeBenchmarkRequest,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    reset = getattr(cache, "reset", None)
    if callable(reset):
        reset()
    input_ids = prompt_tokens.to(device)
    cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
    with execution_workload("prefill"):
        output = cast(Any, model)(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.long),
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
    current = _sample(output.logits, temperature=request.temperature, top_k=request.top_k)
    generated = torch.empty(
        (1, request.max_new_tokens),
        dtype=torch.long,
        device=device,
    )
    generated[:, 0] = current[:, 0]
    decode_position = torch.tensor((input_ids.shape[1],), device=device, dtype=torch.long)
    full_attention_mask = torch.ones(
        (1, input_ids.shape[1] + request.max_new_tokens),
        dtype=torch.long,
        device=device,
    )

    # This intentionally preserves Experiment 002's historical boundary: prefill
    # is excluded at the Python level but was not synchronized before this timer.
    started = time.perf_counter()
    for output_index in range(1, request.max_new_tokens):
        with execution_workload("decode"):
            output = cast(Any, model)(
                input_ids=current,
                attention_mask=full_attention_mask[:, : input_ids.shape[1] + output_index],
                position_ids=decode_position.unsqueeze(0),
                past_key_values=cache,
                cache_position=decode_position,
                use_cache=True,
                return_dict=True,
            )
        current = _sample(output.logits, temperature=request.temperature, top_k=request.top_k)
        generated[:, output_index] = current[:, 0]
        decode_position += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, generated.cpu()


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _case(
    name: str,
    model: nn.Module,
    prompt_tokens: torch.Tensor,
    request: ShortDecodeBenchmarkRequest,
    device: torch.device,
    *,
    implementation: dict[str, Any],
) -> dict[str, Any]:
    dtype = _DTYPES[request.dtype]
    cast(Any, model).config.use_cache = True
    model.eval()
    torch.manual_seed(request.seed)
    torch.cuda.manual_seed_all(request.seed)
    cache = _new_static_cache(
        model,
        device=device,
        dtype=dtype,
        maximum_length=request.prompt_tokens + request.max_new_tokens,
    )
    for _ in range(request.warmups):
        _decode_trial(model, prompt_tokens, cache, request, device)

    samples = []
    peaks_allocated = []
    peaks_reserved = []
    baselines_allocated = []
    output_hashes = []
    for _ in range(request.repetitions):
        torch.cuda.synchronize(device)
        baselines_allocated.append(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        elapsed, output = _decode_trial(model, prompt_tokens, cache, request, device)
        samples.append(elapsed)
        peaks_allocated.append(torch.cuda.max_memory_allocated(device))
        peaks_reserved.append(torch.cuda.max_memory_reserved(device))
        output_hashes.append(_tensor_hash(output))
    units = request.max_new_tokens - 1
    distribution = summarize_benchmark(
        samples,
        unit_name="decode_tokens",
        units_per_sample=units,
    )
    return {
        "name": name,
        "implementation": implementation,
        "timing": distribution.as_dict(),
        "aggregate_throughput_per_second": request.repetitions * units / sum(samples),
        "baseline_allocated_bytes": max(baselines_allocated),
        "peak_allocated_bytes": max(peaks_allocated),
        "peak_reserved_bytes": max(peaks_reserved),
        "incremental_peak_allocated_bytes": max(peaks_allocated) - min(baselines_allocated),
        "output_token_sha256": output_hashes,
    }


def _release_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _base_model(request: ShortDecodeBenchmarkRequest, device: torch.device) -> nn.Module:
    model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(
            request.snapshot,
            local_files_only=True,
            torch_dtype=_DTYPES[request.dtype],
            attn_implementation="eager",
        ),
    )
    return model.to(device)


def _current_comparison(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = float(cases[0]["aggregate_throughput_per_second"])
    baseline_memory = int(cases[0]["peak_allocated_bytes"])
    return [
        {
            "name": str(case["name"]),
            "speedup_vs_base": float(case["aggregate_throughput_per_second"]) / baseline,
            "peak_allocated_ratio_vs_base": int(case["peak_allocated_bytes"]) / baseline_memory,
        }
        for case in cases
    ]


@torch.inference_mode()
def execute_short_decode_benchmark(request: ShortDecodeBenchmarkRequest) -> dict[str, Any]:
    """Run three non-overlapping model cases and return auditable raw samples."""

    device = torch.device(request.device)
    if device.type != "cuda":
        raise ValueError("short-decode benchmark requires CUDA")
    tokenizer = cast(_Tokenizer, AutoTokenizer.from_pretrained(request.snapshot, local_files_only=True))
    prompt_tokens = force_prompt_tokens(tokenizer, request.prompt, request.prompt_tokens)
    fill_token_id = _fill_token_id(tokenizer)
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    loaded_frozen: LoadedFrozenModel | None = None
    loaded_packed: LoadedTransformersRuntime | None = None
    with wait_for_device_lease(request.device, request.wait_for_device_seconds):
        base: nn.Module | None = None
        try:
            base = _base_model(request, device)
            cases.append(
                _case(
                    "base-transformers",
                    base,
                    prompt_tokens,
                    request,
                    device,
                    implementation={"kind": "source", "attention": "eager"},
                )
            )
        finally:
            if base is not None:
                del base
            _release_device_memory()

        try:
            loaded_frozen = load_frozen_run(
                request.run_output,
                request.snapshot,
                source_name=request.source,
                revision=request.revision,
                device=request.device,
                backend=request.backend,
                use_global_tuning=request.use_global_tuning,
            )
            cases.append(
                _case(
                    "frozen-factorized-reference",
                    loaded_frozen.model,
                    prompt_tokens,
                    request,
                    device,
                    implementation={
                        "kind": "logical-frozen-reference",
                        "backend": request.backend,
                        "commit_identity": {
                            "config_hash": loaded_frozen.identity.config_hash,
                            "model_hash": loaded_frozen.identity.model_hash,
                            "plan_hash": loaded_frozen.identity.plan_hash,
                        },
                    },
                )
            )
        finally:
            if loaded_frozen is not None:
                del loaded_frozen
                loaded_frozen = None
            _release_device_memory()

        try:
            loaded_packed = load_transformers_runtime(
                request.runtime_bundle,
                CudaPackedBackend(),
                device=device,
                input_dtype=request.dtype,
                batch_size=1,
                prefill_tokens=request.prompt_tokens,
            )
            cases.append(
                _case(
                    "packed-production",
                    loaded_packed.model,
                    prompt_tokens,
                    request,
                    device,
                    implementation={
                        "kind": "immutable-packed-runtime",
                        "replaced_linear_count": loaded_packed.replaced_linear_count,
                        "prefill_fallback_count": loaded_packed.plans.prefill.plan.fallback_count,
                        "decode_fallback_count": loaded_packed.plans.decode.plan.fallback_count,
                        "prefill_backend": loaded_packed.plans.prefill.plan.layers[0].backend_name,
                        "decode_backend": loaded_packed.plans.decode.plan.layers[0].backend_name,
                    },
                )
            )
        finally:
            if loaded_packed is not None:
                del loaded_packed
                loaded_packed = None
            _release_device_memory()

    properties = torch.cuda.get_device_properties(device)
    legacy = [
        {
            "name": case.name,
            "aggregate_throughput_per_second": case.throughput_per_second,
            "peak_allocated_bytes": case.peak_allocated_bytes,
            "peak_reserved_bytes": case.peak_reserved_bytes,
        }
        for case in request.legacy_cases
    ]
    return {
        "schema_version": 1,
        "passed": len(cases) == 3
        and int(cases[-1]["implementation"]["replaced_linear_count"]) > 0
        and int(cases[-1]["implementation"]["prefill_fallback_count"]) == 0
        and int(cases[-1]["implementation"]["decode_fallback_count"]) == 0,
        "model": {
            "source": request.source,
            "revision": request.revision,
            "snapshot": str(request.snapshot.resolve()),
        },
        "candidate": {
            "run_output": str(request.run_output.resolve()),
            "runtime_bundle": str(request.runtime_bundle.resolve()),
            "runtime_bundle_descriptor_sha256": hash_file(
                request.runtime_bundle / "nanoquant-runtime-bundle.json"
            ),
        },
        "protocol": {
            "prompt": request.prompt,
            "prompt_tokens": request.prompt_tokens,
            "prompt_token_sha256": _tensor_hash(prompt_tokens),
            "fill_token_id": fill_token_id,
            "max_new_tokens": request.max_new_tokens,
            "timed_decode_tokens": request.max_new_tokens - 1,
            "warmups": request.warmups,
            "repetitions": request.repetitions,
            "seed": request.seed,
            "top_k": request.top_k,
            "temperature": request.temperature,
            "dtype": request.dtype,
            "timing_boundary": (
                "legacy-compatible wall timer after unsynchronized prefill; "
                "includes token-2-through-token-N forward and sampling work"
            ),
            "case_lifetime": "sequential-load-measure-release; no model overlap",
        },
        "cases": cases,
        "comparison": _current_comparison(cases),
        "legacy_reference": {
            "summary_sha256": request.legacy_summary_sha256,
            "cases": legacy,
            "disposition": (
                "historical results use the legacy smoke checkpoint and mutable eager/GEMV paths; "
                "current cases use the validated v28 candidate and immutable packed runtime"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": properties.total_memory,
        },
        "peak_host_bytes": peak_process_memory_bytes(),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--runtime-bundle", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bfloat16")
    parser.add_argument("--backend", choices=("dense", "factorized"), default="factorized")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    request = ShortDecodeBenchmarkRequest(
        snapshot=args.snapshot,
        run_output=args.run_output,
        runtime_bundle=args.runtime_bundle,
        source=args.source,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        prompt=args.prompt,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        warmups=args.warmups,
        repetitions=args.repetitions,
        seed=args.seed,
        top_k=args.top_k,
        temperature=args.temperature,
        wait_for_device_seconds=args.wait_for_device_seconds,
    )
    result = execute_short_decode_benchmark(request)
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


__all__ = [
    "LegacyShortDecodeCase",
    "ShortDecodeBenchmarkRequest",
    "execute_short_decode_benchmark",
    "force_prompt_tokens",
]


if __name__ == "__main__":
    main()
