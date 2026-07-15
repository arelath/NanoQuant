"""Benchmark packed NanoQuant kernel, layer, block, prefill, decode, and generation scopes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes
from nanoquant.infrastructure.runtime_export import load_frozen_run_auxiliary
from nanoquant.runtime import (
    BenchmarkDistribution,
    CudaPackedBackend,
    GenerationRequest,
    PreparedLinear,
    SamplingConfig,
    TransformersGenerationModel,
    WorkloadSpec,
    batch_prompts,
    benchmark_wall,
    bind_fused_decode_rope,
    bind_native_bfloat16_tied_projection,
    bind_prepared_linears,
    bind_prepared_rms_norms,
    bind_short_sliding_masks,
    execution_workload,
    generate,
    hybrid_cache_factory,
    open_packed_artifact,
    plan_execution_workloads,
    prepare_execution_workloads,
    summarize_benchmark,
    transformers_decoder_module_paths,
)
from nanoquant.runtime.backend import WorkloadKind

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_SUITES = ("kernel", "layer", "block", "prefill", "decode", "end-to-end")


class _BlockEventCapture:
    def __init__(self, block: nn.Module) -> None:
        self._starts: list[torch.cuda.Event] = []
        self._ends: list[torch.cuda.Event] = []
        self.enabled = False
        self._before = block.register_forward_pre_hook(self._pre)
        self._after = block.register_forward_hook(self._post)

    def _pre(self, _module: nn.Module, _inputs: tuple[object, ...]) -> None:
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._starts.append(event)

    def _post(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        _output: object,
    ) -> None:
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._ends.append(event)

    def close(self) -> None:
        self._before.remove()
        self._after.remove()

    def measured_seconds(self, *, warmups: int, repetitions: int) -> tuple[float, ...]:
        expected = warmups + repetitions
        if len(self._starts) != expected or len(self._ends) != expected:
            raise ValueError(
                f"selected transformer block executed {len(self._starts)} times, expected {expected}"
            )
        return tuple(
            start.elapsed_time(end) / 1000.0
            for start, end in zip(
                self._starts[warmups:],
                self._ends[warmups:],
                strict=True,
            )
        )


def _cuda_benchmark(
    operation: Callable[[object | None], object],
    *,
    setup: Callable[[], object] | None,
    warmups: int,
    repetitions: int,
    device: torch.device,
    unit_name: str,
    units_per_sample: int,
) -> BenchmarkDistribution:
    for _ in range(warmups):
        state = None if setup is None else setup()
        operation(state)
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repetitions):
        state = None if setup is None else setup()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation(state)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
        del result, state
    return summarize_benchmark(
        samples,
        unit_name=unit_name,
        units_per_sample=units_per_sample,
    )


def _case(
    suite: str,
    name: str,
    distribution: BenchmarkDistribution,
    *,
    warmups: int,
    repetitions: int,
    baseline_allocated: int,
    peak_allocated: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "suite": suite,
        "name": name,
        "warmups": warmups,
        "repetitions": repetitions,
        "timing": distribution.as_dict(),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
        **metadata,
    }


def _benchmark_cuda_case(
    suite: str,
    name: str,
    operation: Callable[[object | None], object],
    *,
    setup: Callable[[], object] | None,
    args: argparse.Namespace,
    device: torch.device,
    unit_name: str,
    units_per_sample: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    distribution = _cuda_benchmark(
        operation,
        setup=setup,
        warmups=args.warmups,
        repetitions=args.repetitions,
        device=device,
        unit_name=unit_name,
        units_per_sample=units_per_sample,
    )
    peak = torch.cuda.max_memory_allocated(device)
    return _case(
        suite,
        name,
        distribution,
        warmups=args.warmups,
        repetitions=args.repetitions,
        baseline_allocated=baseline,
        peak_allocated=peak,
        metadata=metadata,
    )


def _token_hash(tokens: torch.Tensor, lengths: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    host = tokens.detach().cpu()
    for row, length in enumerate(lengths):
        digest.update(int(length).to_bytes(8, "little"))
        digest.update(host[row, :length].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _eos_token_ids(value: int | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)) and value:
        return tuple(int(item) for item in value)
    raise ValueError("model configuration contains no EOS token ID")


def _model_layers(model: nn.Module) -> nn.ModuleList:
    core = getattr(model, "model", None)
    layers = None if core is None else getattr(core, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("runtime benchmark model shell exposes no transformer layer list")
    return layers


def _apply_auxiliary(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, str | None]:
    if args.run_output is None:
        return 0, None
    auxiliary = load_frozen_run_auxiliary(
        args.run_output,
        args.expected_blocks,
        use_global_tuning=True,
        fresh_validation=True,
    )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in auxiliary.parameters:
            if name not in parameters:
                raise ValueError(f"frozen auxiliary parameter is absent from shell: {name}")
            target = parameters[name]
            if target.shape != value.shape:
                raise ValueError(f"frozen auxiliary parameter shape differs: {name}")
            target.copy_(value.to(device=device, dtype=target.dtype))
    del parameters
    return (
        len(auxiliary.parameters),
        None if auxiliary.global_tuning is None else auxiliary.global_tuning.artifact_id,
    )


@torch.inference_mode()
def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("packed runtime benchmarks require a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _DTYPES[args.input_dtype]
    selected_suites = set(_SUITES if "all" in args.suite else args.suite)
    artifact = open_packed_artifact(args.packed_artifact, verify_hashes=True)
    entries = tuple(entry for block in artifact.manifest.blocks for entry in block.layers)
    states = {entry.spec.name: artifact.load_layer(entry.spec.name) for entry in entries}
    specs = tuple(entry.spec for entry in entries)
    layer_name = args.layer or entries[0].spec.name
    layer_indices = {entry.spec.name: index for index, entry in enumerate(entries)}
    if layer_name not in layer_indices:
        raise ValueError(f"benchmark layer is absent from the packed artifact: {layer_name}")
    layer_index = layer_indices[layer_name]
    if not 0 <= args.block_index < len(artifact.manifest.blocks):
        raise ValueError("benchmark block index is outside the packed artifact")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt_ids = (
        tuple(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for prompt in args.prompt
        )
        if args.chat_template
        else tuple(tokenizer.encode(prompt, add_special_tokens=True) for prompt in args.prompt)
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("model tokenizer contains no pad token ID")
    input_ids, attention_mask = batch_prompts(prompt_ids, pad_token_id=pad_token_id)
    batch_size, prompt_width = input_ids.shape
    actual_prompt_tokens = sum(len(item) for item in prompt_ids)
    maximum_cache_length = prompt_width + args.max_new_tokens
    backend = CudaPackedBackend()
    plans = plan_execution_workloads(
        specs,
        prefill=WorkloadSpec(
            "prefill", "cuda", args.input_dtype, batch_size, prompt_width, deterministic=True
        ),
        decode=WorkloadSpec(
            "decode", "cuda", args.input_dtype, batch_size, 1, deterministic=True
        ),
        prefill_backends=(backend,),
        decode_backends=(backend,),
        strict=True,
    )

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    with wait_for_device_lease(str(device), args.wait_for_device_seconds):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).eval()
        model.to(device)
        native_bfloat16_tied_projection_count = (
            bind_native_bfloat16_tied_projection(model)
            if args.native_bfloat16_tied_projection
            and device.type == "cuda"
            and dtype == torch.float32
            else 0
        )
        auxiliary_count, global_tuning = _apply_auxiliary(model, args, device)
        prepared = prepare_execution_workloads(plans, states, (backend,), device)
        layer_names = tuple(entry.spec.name for entry in entries)
        replaced = bind_prepared_linears(
            model,
            prepared,
            transformers_decoder_module_paths(layer_names),
        )
        fused_rms_norm_count = bind_prepared_rms_norms(model) if args.fused_rms_norm else 0
        fused_decode_rope_count = bind_fused_decode_rope(model) if args.fused_decode_rope else 0
        short_sliding_mask_count = (
            bind_short_sliding_masks(model) if args.short_sliding_masks else 0
        )
        del states
        gc.collect()
        torch.cuda.empty_cache()

        device_input_ids = input_ids.to(device)
        device_attention_mask = attention_mask.to(device)
        prompt_positions = torch.arange(prompt_width, device=device).expand(batch_size, -1).clone()
        prompt_positions.masked_fill_(~device_attention_mask, 0)
        prompt_cache_positions = torch.arange(prompt_width, device=device)
        full_attention_mask = torch.zeros(
            (batch_size, maximum_cache_length), dtype=torch.bool, device=device
        )
        full_attention_mask[:, :prompt_width] = device_attention_mask
        cache_factory = hybrid_cache_factory(
            model.config,
            None if args.cache_dtype is None else _DTYPES[args.cache_dtype],
            fast_sliding_prefix=args.fast_sliding_cache,
            fused_cache_prefix=args.fused_cache_prefix,
        )
        last_cache: list[object | None] = [None]

        def tracked_cache_factory(
            current_batch_size: int,
            current_cache_length: int,
            current_device: torch.device,
            current_dtype: torch.dtype,
        ) -> object:
            cache = cache_factory(
                current_batch_size,
                current_cache_length,
                current_device,
                current_dtype,
            )
            last_cache[0] = cache
            return cache

        shell = TransformersGenerationModel(model, tracked_cache_factory)

        selected_spec = entries[layer_index].spec
        for workload, plan, token_count in (
            ("prefill", prepared.prefill, prompt_width),
            ("decode", prepared.decode, 1),
        ):
            value = torch.randn(
                (batch_size, token_count, selected_spec.in_features),
                device=device,
                dtype=dtype,
            )
            metadata = {
                "workload": workload,
                "layer": layer_name,
                "shape": list(value.shape),
                "backend": plan.dispatches[layer_index].plan.backend_name,
            }
            if "kernel" in selected_suites:
                cases.append(
                    _benchmark_cuda_case(
                        "kernel",
                        f"{workload}:{layer_name}",
                        lambda _state, dispatch=plan.dispatches[layer_index], tensor=value: dispatch.linear(
                            tensor
                        ),
                        setup=None,
                        args=args,
                        device=device,
                        unit_name="token_slots",
                        units_per_sample=batch_size * token_count,
                        metadata=metadata,
                    )
                )
            if "layer" in selected_suites:
                module = PreparedLinear(prepared, layer_index)

                def run_layer(
                    _state: object | None,
                    *,
                    kind: WorkloadKind = workload,
                    tensor: torch.Tensor = value,
                    selected: PreparedLinear = module,
                ) -> torch.Tensor:
                    with execution_workload(kind):
                        return selected(tensor)

                cases.append(
                    _benchmark_cuda_case(
                        "layer",
                        f"{workload}:{layer_name}",
                        run_layer,
                        setup=None,
                        args=args,
                        device=device,
                        unit_name="token_slots",
                        units_per_sample=batch_size * token_count,
                        metadata=metadata,
                    )
                )

        def prefill(_state: object | None = None) -> object:
            return shell.forward_step(
                input_ids=device_input_ids,
                attention_mask=full_attention_mask[:, :prompt_width],
                position_ids=prompt_positions,
                cache_position=prompt_cache_positions,
                cache=None,
                max_cache_length=maximum_cache_length,
                workload="prefill",
                deterministic=True,
            )

        def decode_setup() -> object:
            step = prefill()
            logits = step.logits
            cache = step.cache
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            mask = full_attention_mask[:, : prompt_width + 1].clone()
            mask[:, -1] = True
            positions = torch.full(
                (batch_size, 1), prompt_width, dtype=torch.long, device=device
            )
            cache_position = torch.tensor((prompt_width,), dtype=torch.long, device=device)
            return (cache, next_tokens, mask, positions, cache_position)

        def decode(state: object | None) -> object:
            if not isinstance(state, tuple) or len(state) != 5:
                raise ValueError("decode benchmark setup returned invalid state")
            cache, tokens, mask, positions, cache_position = state
            return shell.forward_step(
                input_ids=tokens,
                attention_mask=mask,
                position_ids=positions,
                cache_position=cache_position,
                cache=cache,
                max_cache_length=maximum_cache_length,
                workload="decode",
                deterministic=True,
            )

        if "prefill" in selected_suites:
            cases.append(
                _benchmark_cuda_case(
                    "prefill",
                    "model-prefill",
                    prefill,
                    setup=None,
                    args=args,
                    device=device,
                    unit_name="prompt_tokens",
                    units_per_sample=actual_prompt_tokens,
                    metadata={"prompt_width": prompt_width, "batch_size": batch_size},
                )
            )
        if "decode" in selected_suites:
            cases.append(
                _benchmark_cuda_case(
                    "decode",
                    "model-single-token-decode",
                    decode,
                    setup=decode_setup,
                    args=args,
                    device=device,
                    unit_name="decoded_tokens",
                    units_per_sample=batch_size,
                    metadata={"batch_size": batch_size, "cache_length": prompt_width},
                )
            )

        if "block" in selected_suites:
            block_capture = _BlockEventCapture(_model_layers(model)[args.block_index])
            try:
                def measured_prefill(_state: object | None = None) -> object:
                    block_capture.enabled = True
                    try:
                        return prefill()
                    finally:
                        block_capture.enabled = False

                measured_prefill_case = _benchmark_cuda_case(
                    "block",
                    f"prefill:block-{args.block_index}:carrier",
                    measured_prefill,
                    setup=None,
                    args=args,
                    device=device,
                    unit_name="prompt_tokens",
                    units_per_sample=actual_prompt_tokens,
                    metadata={"instrumentation": "internal-block-events"},
                )
                block_distribution = summarize_benchmark(
                    block_capture.measured_seconds(
                        warmups=args.warmups, repetitions=args.repetitions
                    ),
                    unit_name="token_slots",
                    units_per_sample=batch_size * prompt_width,
                )
                cases.append(
                    _case(
                        "block",
                        f"prefill:block-{args.block_index}",
                        block_distribution,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                        baseline_allocated=measured_prefill_case["baseline_allocated_bytes"],
                        peak_allocated=measured_prefill_case["peak_allocated_bytes"],
                        metadata={
                            "workload": "prefill",
                            "block_index": args.block_index,
                            "batch_size": batch_size,
                            "token_count": prompt_width,
                            "instrumentation": "cuda-events-in-block-hooks",
                        },
                    )
                )

                block_capture.close()
                block_capture = _BlockEventCapture(_model_layers(model)[args.block_index])

                def measured_decode(state: object | None) -> object:
                    block_capture.enabled = True
                    try:
                        return decode(state)
                    finally:
                        block_capture.enabled = False

                measured_decode_case = _benchmark_cuda_case(
                    "block",
                    f"decode:block-{args.block_index}:carrier",
                    measured_decode,
                    setup=decode_setup,
                    args=args,
                    device=device,
                    unit_name="decoded_tokens",
                    units_per_sample=batch_size,
                    metadata={"instrumentation": "internal-block-events"},
                )
                block_distribution = summarize_benchmark(
                    block_capture.measured_seconds(
                        warmups=args.warmups, repetitions=args.repetitions
                    ),
                    unit_name="decoded_tokens",
                    units_per_sample=batch_size,
                )
                cases.append(
                    _case(
                        "block",
                        f"decode:block-{args.block_index}",
                        block_distribution,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                        baseline_allocated=measured_decode_case["baseline_allocated_bytes"],
                        peak_allocated=measured_decode_case["peak_allocated_bytes"],
                        metadata={
                            "workload": "decode",
                            "block_index": args.block_index,
                            "batch_size": batch_size,
                            "token_count": 1,
                            "instrumentation": "cuda-events-in-block-hooks",
                        },
                    )
                )
            finally:
                block_capture.close()

        generated_hashes: list[str] = []
        if "end-to-end" in selected_suites:
            eos = (
                (int(model.config.vocab_size),)
                if args.ignore_eos
                else _eos_token_ids(model.config.eos_token_id)
            )

            def request(max_new_tokens: int) -> GenerationRequest:
                return GenerationRequest(
                    device_input_ids,
                    device_attention_mask,
                    max_new_tokens,
                    eos,
                    pad_token_id,
                    sampling=SamplingConfig(mode="greedy"),
                    stopping_check_interval=args.stopping_check_interval,
                )

            def run_generation(generation_request: GenerationRequest) -> object:
                result = generate(generation_request, shell)
                generated_hashes.append(_token_hash(result.token_ids, result.lengths))
                return result

            for name, generation_request, units in (
                ("time-to-first-token", request(1), batch_size),
                (
                    "complete-generation",
                    request(args.max_new_tokens),
                    batch_size * args.max_new_tokens,
                ),
            ):
                generated_hashes.clear()
                torch.cuda.synchronize(device)
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                distribution = benchmark_wall(
                    lambda current=generation_request: run_generation(current),
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    synchronize=lambda: torch.cuda.synchronize(device),
                    unit_name="generated_tokens",
                    units_per_sample=units,
                )
                if len(set(generated_hashes)) != 1:
                    raise ValueError(f"deterministic generation output changed during {name}")
                peak = torch.cuda.max_memory_allocated(device)
                cases.append(
                    _case(
                        "end-to-end",
                        name,
                        distribution,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                        baseline_allocated=baseline,
                        peak_allocated=peak,
                        metadata={
                            "batch_size": batch_size,
                            "prompt_width": prompt_width,
                            "max_new_tokens": generation_request.max_new_tokens,
                            "token_output_sha256": generated_hashes[0],
                        },
                    )
                )

        torch.cuda.synchronize(device)
        allocated_after = torch.cuda.memory_allocated(device)

    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": 1,
        "passed": True,
        "artifact": {
            "path": str(artifact.root.resolve()),
            "descriptor_sha256": hash_file(artifact.root / "nanoquant-packed-model.json"),
            "layer_count": len(entries),
            "weight_bytes": artifact.manifest.weight_bytes,
        },
        "model": str(args.model.resolve()),
        "run_output": None if args.run_output is None else str(args.run_output.resolve()),
        "auxiliary_parameter_count": auxiliary_count,
        "global_tuning_artifact": global_tuning,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_runtime": torch.version.cuda,
            "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
            "device": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "cpu_count": os.cpu_count(),
        },
        "configuration": {
            "suites": sorted(selected_suites),
            "input_dtype": args.input_dtype,
            "cache_dtype": args.cache_dtype or args.input_dtype,
            "fused_rms_norm": args.fused_rms_norm,
            "fused_decode_rope": args.fused_decode_rope,
            "short_sliding_masks": args.short_sliding_masks,
            "fast_sliding_cache": args.fast_sliding_cache,
            "fused_cache_prefix": args.fused_cache_prefix,
            "native_bfloat16_tied_projection": args.native_bfloat16_tied_projection,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "prompt": args.prompt,
            "chat_template": args.chat_template,
            "prompt_lengths": [len(item) for item in prompt_ids],
            "prompt_width": prompt_width,
            "batch_size": batch_size,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": args.ignore_eos,
            "stopping_check_interval": args.stopping_check_interval,
            "layer": layer_name,
            "block_index": args.block_index,
        },
        "dispatch": {
            "replaced_linear_count": replaced,
            "fused_rms_norm_count": fused_rms_norm_count,
            "fused_decode_rope_count": fused_decode_rope_count,
            "short_sliding_mask_count": short_sliding_mask_count,
            "fused_cache_update_count": getattr(
                last_cache[0], "nanoquant_fused_cache_update_count", 0
            ),
            "native_bfloat16_tied_projection_count": (
                native_bfloat16_tied_projection_count
            ),
            "prefill_fallback_count": plans.prefill.fallback_count,
            "decode_fallback_count": plans.decode.fallback_count,
            "prefill_backend": plans.prefill.layers[0].backend_name,
            "decode_backend": plans.decode.layers[0].backend_name,
        },
        "cases": cases,
        "allocated_after_bytes": allocated_after,
        "peak_host_bytes": peak_process_memory_bytes(),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--expected-blocks", type=int, default=26)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--cache-dtype", choices=tuple(_DTYPES))
    parser.add_argument(
        "--suite",
        action="append",
        choices=("all", *_SUITES),
        default=[],
        help="Benchmark suite; repeat to select multiple suites (default: all).",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--fused-rms-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replace Gemma3 RMSNorms with the native fused F32 operation",
    )
    parser.add_argument(
        "--fused-decode-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replace pinned one-token Gemma3 RoPE with one Triton launch",
    )
    parser.add_argument(
        "--short-sliding-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="elide identity Gemma3 sliding masks while the context fits the window",
    )
    parser.add_argument(
        "--fast-sliding-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use direct prefix updates until a sliding KV cache reaches rollover",
    )
    parser.add_argument(
        "--fused-cache-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fuse F32-to-F16 prefix updates and F16-to-F32 attention views",
    )
    parser.add_argument(
        "--native-bfloat16-tied-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain the tied Gemma embedding/output table in native BF16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--stopping-check-interval", type=int, default=8)
    parser.add_argument("--layer")
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.suite:
        args.suite = ["all"]
    if not args.prompt:
        args.prompt = ["Write a short paragraph about quantization."]
    if args.warmups < 0 or args.repetitions <= 0:
        parser.error("warmups must be non-negative and repetitions must be positive")
    if args.max_new_tokens <= 0 or args.stopping_check_interval <= 0:
        parser.error("generation lengths and stopping interval must be positive")
    result = _benchmark(args)
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
