"""Profile one protocol-matched packed Gemma decode and account its nested CUDA time."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes
from nanoquant.runtime import (
    CudaPackedBackend,
    PreparedLinear,
    TransformersGenerationModel,
    batch_prompts,
    hybrid_cache_factory,
    load_transformers_runtime,
    open_runtime_bundle,
    profile_ratio_samples,
    sum_aligned_profile_samples,
    summarize_benchmark,
)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _distribution(values: Sequence[float]) -> dict[str, float]:
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("profile distributions require at least one sample")
    # BenchmarkDistribution already supplies stable interpolated percentiles. A
    # unit count of one makes its latency distribution the generic value summary.
    return summarize_benchmark(
        samples,
        unit_name="value",
        units_per_sample=1,
    ).latency_seconds


class _CudaModuleCapture:
    """Preallocate event pairs and record one execution per module and sample."""

    def __init__(self, modules: dict[str, nn.Module], repetitions: int) -> None:
        if repetitions <= 0:
            raise ValueError("profile repetitions must be positive")
        self._events = {
            name: tuple(
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(repetitions)
            )
            for name in modules
        }
        self._counts = {name: [0] * repetitions for name in modules}
        self._handles: list[Any] = []
        self._sample: int | None = None
        for name, module in modules.items():
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self._handles.append(module.register_forward_hook(self._post_hook(name)))

    def _pre_hook(self, name: str):  # type: ignore[no-untyped-def]
        def record(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
            sample = self._sample
            if sample is None:
                return
            if self._counts[name][sample] != 0:
                raise RuntimeError(f"profiled module executed more than once: {name}")
            self._events[name][sample][0].record()
            self._counts[name][sample] = 1

        return record

    def _post_hook(self, name: str):  # type: ignore[no-untyped-def]
        def record(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            _output: object,
        ) -> None:
            sample = self._sample
            if sample is None:
                return
            if self._counts[name][sample] != 1:
                raise RuntimeError(f"profiled module post-hook has no pre-hook: {name}")
            self._events[name][sample][1].record()
            self._counts[name][sample] = 2

        return record

    def begin(self, sample: int) -> None:
        if self._sample is not None:
            raise RuntimeError("a CUDA module profile sample is already active")
        self._sample = sample

    def end(self) -> None:
        if self._sample is None:
            raise RuntimeError("no CUDA module profile sample is active")
        self._sample = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def samples(self) -> dict[str, tuple[float, ...]]:
        result = {}
        for name, counts in self._counts.items():
            if all(count == 0 for count in counts):
                continue
            if any(count != 2 for count in counts):
                raise RuntimeError(f"profiled module did not execute exactly once per sample: {name}")
            result[name] = tuple(
                start.elapsed_time(end) / 1000.0 for start, end in self._events[name]
            )
        return result


def _module_inventory(model: nn.Module) -> tuple[dict[str, nn.Module], dict[str, tuple[str, ...]]]:
    core = getattr(model, "model", None)
    layers = None if core is None else getattr(core, "layers", None)
    if not isinstance(core, nn.Module) or not isinstance(layers, nn.ModuleList):
        raise ValueError("runtime profile model exposes no decoder layer list")

    modules: dict[str, nn.Module] = {"model-forward": model}
    top_level = []
    for child_name, child in core.named_children():
        if child is layers:
            continue
        label = f"core.{child_name}"
        modules[label] = child
        top_level.append(label)
    for child_name, child in model.named_children():
        if child is core:
            continue
        label = f"head.{child_name}"
        modules[label] = child
        top_level.append(label)

    blocks = []
    components = []
    component_roles: dict[str, list[str]] = defaultdict(list)
    for index, block in enumerate(layers):
        block_label = f"block.{index}"
        modules[block_label] = block
        blocks.append(block_label)
        for child_name, child in block.named_children():
            component_label = f"block.{index}.{child_name}"
            modules[component_label] = child
            components.append(component_label)
            component_roles[child_name].append(component_label)
    top_level.extend(blocks)

    linears = []
    roles: dict[str, list[str]] = defaultdict(list)
    for module_name, module in model.named_modules():
        if not isinstance(module, PreparedLinear):
            continue
        label = f"linear.{module_name}"
        modules[label] = module
        linears.append(label)
        roles[module_name.rsplit(".", 1)[-1]].append(label)
    if len(blocks) != 26 or len(linears) != 182:
        raise ValueError(
            f"pinned Gemma profile inventory differs: blocks={len(blocks)}, linears={len(linears)}"
        )
    groups = {
        "top_level": tuple(top_level),
        "blocks": tuple(blocks),
        "components": tuple(components),
        "linears": tuple(linears),
        **{
            f"component_role.{role}": tuple(names)
            for role, names in sorted(component_roles.items())
        },
        **{f"linear_role.{role}": tuple(names) for role, names in sorted(roles.items())},
    }
    return modules, groups


def _timing(samples: Sequence[float], unit_name: str = "decode_forwards") -> dict[str, object]:
    return summarize_benchmark(
        samples,
        unit_name=unit_name,
        units_per_sample=1,
    ).as_dict()


def _torch_event_record(event: object) -> dict[str, object]:
    return {
        "key": str(event.key),
        "count": int(event.count),
        "self_cpu_seconds": float(event.self_cpu_time_total) / 1_000_000.0,
        "cpu_total_seconds": float(event.cpu_time_total) / 1_000_000.0,
        "self_device_seconds": float(event.self_device_time_total) / 1_000_000.0,
        "device_total_seconds": float(event.device_time_total) / 1_000_000.0,
    }


def _summarize_torch_profile(trace: object, *, wall_seconds: float, output_token: int) -> dict[str, object]:
    events = tuple(trace.key_averages())
    cuda_events = tuple(
        event
        for event in events
        if str(event.device_type).rsplit(".", 1)[-1].lower() == "cuda"
        and float(event.self_device_time_total) > 0.0
    )
    cpu_events = tuple(
        event
        for event in events
        if str(event.device_type).rsplit(".", 1)[-1].lower() == "cpu"
        and float(event.self_cpu_time_total) > 0.0
    )
    cuda_sorted = sorted(
        cuda_events,
        key=lambda event: float(event.self_device_time_total),
        reverse=True,
    )
    cpu_sorted = sorted(
        cpu_events,
        key=lambda event: float(event.self_cpu_time_total),
        reverse=True,
    )
    launch_events = tuple(
        event
        for event in cpu_events
        if str(event.key).startswith(("cudaLaunchKernel", "cuLaunchKernelEx"))
    )
    aten_events = tuple(event for event in cpu_events if str(event.key).startswith("aten::"))
    nanoquant_events = tuple(
        event
        for event in cuda_events
        if str(event.key).startswith("_nanoquant_stage")
    )
    return {
        "wall_seconds": wall_seconds,
        "output_token_id": output_token,
        "cuda": {
            "kernel_launch_count": sum(int(event.count) for event in cuda_events),
            "unique_kernel_count": len(cuda_events),
            "kernel_self_seconds": sum(
                float(event.self_device_time_total) for event in cuda_events
            )
            / 1_000_000.0,
            "nanoquant_kernel_launch_count": sum(
                int(event.count) for event in nanoquant_events
            ),
            "nanoquant_kernel_self_seconds": sum(
                float(event.self_device_time_total) for event in nanoquant_events
            )
            / 1_000_000.0,
            "kernels": [_torch_event_record(event) for event in cuda_sorted],
        },
        "cpu": {
            "cuda_launch_api_count": sum(int(event.count) for event in launch_events),
            "cuda_launch_api_self_seconds": sum(
                float(event.self_cpu_time_total) for event in launch_events
            )
            / 1_000_000.0,
            "cuda_launch_api_events": [
                _torch_event_record(event)
                for event in sorted(launch_events, key=lambda event: str(event.key))
            ],
            "aten_operator_call_count": sum(int(event.count) for event in aten_events),
            "aten_operator_self_seconds": sum(
                float(event.self_cpu_time_total) for event in aten_events
            )
            / 1_000_000.0,
            "top_self_events": [_torch_event_record(event) for event in cpu_sorted[:100]],
        },
        "notes": {
            "timing_is_diagnostic": True,
            "reason": "Kineto tracing and synchronization add overhead; CUDA-event passes remain authoritative.",
            "cuda_kernel_self_time_is_non_nested": True,
        },
    }


@torch.inference_mode()
def _profile(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("runtime decode profiling requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    opened = open_runtime_bundle(args.bundle, verify_hashes=True)
    tokenizer = cast(Any, opened.load_tokenizer())
    prompt_ids = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if args.chat_template
        else tokenizer.encode(args.prompt, add_special_tokens=True)
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("runtime profile tokenizer contains no pad token ID")
    input_ids, attention_mask = batch_prompts((prompt_ids,), pad_token_id=int(pad_token_id))
    prompt_width = input_ids.shape[1]
    maximum_cache_length = prompt_width + args.max_new_tokens
    started = time.perf_counter()

    with wait_for_device_lease(str(device), args.wait_for_device_seconds):
        runtime = load_transformers_runtime(
            opened,
            CudaPackedBackend(),
            device=device,
            input_dtype=args.input_dtype,
            batch_size=1,
            prefill_tokens=prompt_width,
            fuse_rms_norm=args.fused_rms_norm,
            fuse_decode_rope=args.fused_decode_rope,
        )
        model = runtime.model
        shell = TransformersGenerationModel(
            model,
            hybrid_cache_factory(model.config, _DTYPES[args.cache_dtype]),
        )
        tokens = input_ids.to(device)
        prompt_mask = attention_mask.to(device)
        full_mask = torch.zeros((1, maximum_cache_length), dtype=torch.bool, device=device)
        full_mask[:, :prompt_width] = prompt_mask
        positions = torch.arange(prompt_width, dtype=torch.long, device=device).unsqueeze(0)
        cache_positions = torch.arange(prompt_width, dtype=torch.long, device=device)

        def prefill() -> tuple[object, torch.Tensor]:
            step = shell.forward_step(
                input_ids=tokens,
                attention_mask=full_mask[:, :prompt_width],
                position_ids=positions,
                cache_position=cache_positions,
                cache=None,
                max_cache_length=maximum_cache_length,
                workload="prefill",
                deterministic=True,
            )
            if step.cache is None:
                raise ValueError("runtime profile prefill returned no cache")
            next_token = torch.argmax(step.logits[:, -1, :], dim=-1).unsqueeze(1)
            return step.cache, next_token

        decode_mask = full_mask[:, : prompt_width + 1].clone()
        decode_mask[:, -1] = True
        decode_positions = torch.full((1, 1), prompt_width, dtype=torch.long, device=device)
        decode_cache_position = torch.tensor((prompt_width,), dtype=torch.long, device=device)

        def decode(cache: object, next_token: torch.Tensor):
            return shell.forward_step(
                input_ids=next_token,
                attention_mask=decode_mask,
                position_ids=decode_positions,
                cache_position=decode_cache_position,
                cache=cache,
                max_cache_length=maximum_cache_length,
                workload="decode",
                deterministic=True,
            )

        for _ in range(args.warmups):
            cache, next_token = prefill()
            decode(cache, next_token)
        torch.cuda.synchronize(device)

        modules, groups = _module_inventory(model)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

        def run_pass(region_names: Iterable[str]) -> tuple[
            dict[str, tuple[float, ...]], tuple[float, ...], tuple[int, ...]
        ]:
            selected = {name: modules[name] for name in dict.fromkeys(region_names)}
            capture = _CudaModuleCapture(selected, args.repetitions)
            wall_samples = []
            output_tokens = []
            try:
                for sample in range(args.repetitions):
                    cache, next_token = prefill()
                    torch.cuda.synchronize(device)
                    sample_started = time.perf_counter()
                    capture.begin(sample)
                    try:
                        step = decode(cache, next_token)
                    finally:
                        capture.end()
                    torch.cuda.synchronize(device)
                    wall_samples.append(time.perf_counter() - sample_started)
                    output_tokens.append(
                        int(torch.argmax(step.logits[:, -1, :], dim=-1).item())
                    )
            finally:
                capture.close()
            return capture.samples(), tuple(wall_samples), tuple(output_tokens)

        pass_regions = {
            "top_level": ("model-forward", *groups["top_level"]),
            "components": (
                "model-forward",
                *groups["blocks"],
                *groups["components"],
            ),
            "linears": ("model-forward", *groups["blocks"], *groups["linears"]),
        }
        raw_passes = {
            name: run_pass(regions) for name, regions in pass_regions.items()
        }

        kernel_profile = None
        if args.kernel_profile:
            from torch.profiler import ProfilerActivity, profile

            cache, next_token = prefill()
            torch.cuda.synchronize(device)
            kernel_started = time.perf_counter()
            with profile(
                activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as trace:
                kernel_step = decode(cache, next_token)
                torch.cuda.synchronize(device)
            kernel_wall = time.perf_counter() - kernel_started
            kernel_output_token = int(
                torch.argmax(kernel_step.logits[:, -1, :], dim=-1).item()
            )
            kernel_profile = _summarize_torch_profile(
                trace,
                wall_seconds=kernel_wall,
                output_token=kernel_output_token,
            )

        def pass_groups(
            region_samples: dict[str, tuple[float, ...]],
        ) -> dict[str, tuple[float, ...]]:
            return {
                name: sum_aligned_profile_samples(region_samples, regions)
                for name, regions in groups.items()
                if regions and all(region in region_samples for region in regions)
            }

        grouped_passes = {
            name: pass_groups(region_samples)
            for name, (region_samples, _wall, _tokens) in raw_passes.items()
        }
        top_regions, top_wall, _top_tokens = raw_passes["top_level"]
        component_regions, _component_wall, _component_tokens = raw_passes["components"]
        linear_regions, _linear_wall, _linear_tokens = raw_passes["linears"]
        top_groups = grouped_passes["top_level"]
        component_groups = grouped_passes["components"]
        linear_groups = grouped_passes["linears"]
        ratios = {
            "top_level_over_model": profile_ratio_samples(
                top_groups["top_level"], top_regions["model-forward"]
            ),
            "top_level_over_wall": profile_ratio_samples(top_groups["top_level"], top_wall),
            "model_over_wall": profile_ratio_samples(top_regions["model-forward"], top_wall),
            "components_over_blocks": profile_ratio_samples(
                component_groups["components"], component_groups["blocks"]
            ),
            "linears_over_blocks": profile_ratio_samples(
                linear_groups["linears"], linear_groups["blocks"]
            ),
            "linears_over_model": profile_ratio_samples(
                linear_groups["linears"], linear_regions["model-forward"]
            ),
        }
        profile_passes = {
            name: {
                "wall": _timing(wall),
                "model_cuda": _timing(regions["model-forward"]),
                "groups": {
                    group_name: _timing(values)
                    for group_name, values in grouped_passes[name].items()
                },
                "regions": {
                    region_name: _timing(values)
                    for region_name, values in regions.items()
                },
                "registered_region_count": len(pass_regions[name]),
                "executed_region_count": len(regions),
            }
            for name, (regions, wall, _tokens) in raw_passes.items()
        }
        output_tokens = {
            name: list(tokens) for name, (_regions, _wall, tokens) in raw_passes.items()
        }
        all_output_tokens = tuple(token for tokens in output_tokens.values() for token in tokens)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        dispatch = {
            "replaced_linear_count": runtime.replaced_linear_count,
            "fused_rms_norm_count": runtime.fused_rms_norm_count,
            "fused_decode_rope_count": runtime.fused_decode_rope_count,
            "prefill_fallback_count": runtime.plans.prefill.plan.fallback_count,
            "decode_fallback_count": runtime.plans.decode.plan.fallback_count,
        }

    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": 1,
        "passed": True,
        "bundle": {
            "path": str(opened.root.resolve()),
            "descriptor_sha256": hash_file(opened.root / "nanoquant-runtime-bundle.json"),
            "packed_descriptor_sha256": opened.manifest.packed_descriptor_sha256,
        },
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
            "prompt": args.prompt,
            "chat_template": args.chat_template,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": args.max_new_tokens,
            "batch_size": 1,
            "input_dtype": args.input_dtype,
            "cache_dtype": args.cache_dtype,
            "fused_rms_norm": args.fused_rms_norm,
            "fused_decode_rope": args.fused_decode_rope,
            "attention": "eager",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
        },
        "dispatch": dispatch,
        "profile_passes": profile_passes,
        "kernel_profile": kernel_profile,
        "accounting": {
            name: {"samples": list(values), "distribution": _distribution(values)}
            for name, values in ratios.items()
        },
        "inventory": {
            "available_region_count": len(modules),
            "top_level_regions": list(groups["top_level"]),
            "block_regions": list(groups["blocks"]),
            "component_regions": list(groups["components"]),
            "linear_regions": list(groups["linears"]),
        },
        "output": {
            "second_generated_token_ids_by_pass": output_tokens,
            "deterministic": len(set(all_output_tokens)) == 1,
        },
        "memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "peak_allocated_bytes": peak_allocated,
            "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
            "baseline_reserved_bytes": baseline_reserved,
            "peak_reserved_bytes": peak_reserved,
            "device_free_after_bytes": free_bytes,
            "device_total_bytes": total_bytes,
            "peak_host_bytes": peak_process_memory_bytes(),
        },
        "instrumentation": {
            "event_pairs_preallocated": True,
            "synchronization": "before-and-after-each-profiled-decode",
            "independent_passes": ["top_level", "components", "linears"],
            "nested_regions_are_not_summed_across_levels": True,
            "wall_seconds": time.perf_counter() - started,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--cache-dtype", choices=tuple(_DTYPES), default="float16")
    parser.add_argument("--prompt", default="Write a short paragraph about quantization.")
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
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
        "--kernel-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also capture one separate warmed Kineto CPU/CUDA kernel trace",
    )
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens <= 1:
        parser.error("max-new-tokens must be greater than one")
    if args.warmups < 0 or args.repetitions <= 0:
        parser.error("warmups must be non-negative and repetitions must be positive")
    result = _profile(args)
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
