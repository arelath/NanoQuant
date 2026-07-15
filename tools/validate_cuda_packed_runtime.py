"""Validate the CUDA packed backend against FP32 semantics on a real packed artifact."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.runtime import CudaPackedBackend, PackedLayerState, WorkloadSpec, open_packed_artifact

_INPUT_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _reference(value: torch.Tensor, state: PackedLayerState) -> torch.Tensor:
    logical = state.to_logical()
    value_float = value.float()
    latent = torch.nn.functional.linear(
        value_float * logical.scale_pre.float(),
        logical.right_binary.float(),
    )
    output = torch.nn.functional.linear(
        latent * logical.scale_mid.float(),
        logical.left_binary.float() * logical.scale_post.float().reshape(-1, 1),
    )
    if logical.outlier_indices is not None and logical.outlier_values is not None:
        weights = logical.outlier_values.float()
        if logical.outlier_scales is not None:
            weights = weights * logical.outlier_scales.float()
        output += torch.nn.functional.linear(
            value_float.index_select(-1, logical.outlier_indices.long()),
            weights,
        )
    if logical.bias is not None:
        output += logical.bias.float()
    return output


def _shape_inventory(blocks: tuple[Any, ...]) -> list[dict[str, int]]:
    shapes = {
        (entry.spec.in_features, entry.spec.out_features, entry.spec.rank)
        for block in blocks
        for entry in block.layers
    }
    return [
        {"in_features": in_features, "out_features": out_features, "rank": rank}
        for in_features, out_features, rank in sorted(shapes)
    ]


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    packed = open_packed_artifact(args.packed_artifact, verify_hashes=True)
    selected_blocks = packed.manifest.blocks[: args.blocks]
    backend = CudaPackedBackend()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA packed validation requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _INPUT_DTYPES[args.input_dtype]
    generator = torch.Generator().manual_seed(args.seed)
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    maximum_normalized_error = 0.0
    maximum_error_layer = ""
    compared_elements = 0
    layer_count = 0
    started = time.perf_counter()
    with wait_for_device_lease(str(device), args.wait_for_device_seconds):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            baseline_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        for block in selected_blocks:
            for entry in block.layers:
                state = packed.load_layer(entry.spec.name)
                workload = WorkloadSpec(
                    "decode" if args.tokens == 1 else "prefill",
                    "cuda",
                    args.input_dtype,
                    1,
                    args.tokens,
                    deterministic=True,
                )
                support = backend.supports(state.spec, workload)
                if not support.supported:
                    raise ValueError(
                        f"CUDA backend rejected {state.spec.name}: {support.code}: {support.reason}"
                    )
                value = torch.randn(
                    args.tokens,
                    state.spec.in_features,
                    generator=generator,
                    dtype=dtype,
                )
                expected = _reference(value, state)
                prepared = backend.prepare(state, device)
                device_value = value.to(device)
                actual_device = backend.linear(device_value, prepared)
                repeated_device = backend.linear(device_value, prepared)
                if not torch.equal(actual_device, repeated_device):
                    raise ValueError(f"CUDA deterministic replay differs: {state.spec.name}")
                actual = actual_device.cpu()
                difference = torch.abs(actual - expected)
                tolerance = args.absolute_tolerance + args.relative_tolerance * torch.abs(expected)
                normalized = difference / tolerance
                layer_absolute_error = float(torch.max(difference))
                layer_normalized_error = float(torch.max(normalized))
                nonzero = torch.abs(expected) > 1e-12
                layer_relative_error = (
                    float(torch.max(difference[nonzero] / torch.abs(expected[nonzero])))
                    if bool(torch.any(nonzero))
                    else 0.0
                )
                maximum_absolute_error = max(maximum_absolute_error, layer_absolute_error)
                maximum_relative_error = max(maximum_relative_error, layer_relative_error)
                if layer_normalized_error > maximum_normalized_error:
                    maximum_normalized_error = layer_normalized_error
                    maximum_error_layer = state.spec.name
                if bool(torch.any(normalized > 1)):
                    index = int(torch.argmax(normalized))
                    raise ValueError(
                        f"CUDA output differs for {state.spec.name} at flat index {index}: "
                        f"maximum normalized error {layer_normalized_error}"
                    )
                compared_elements += actual.numel()
                layer_count += 1
                del (
                    actual,
                    actual_device,
                    device_value,
                    difference,
                    expected,
                    normalized,
                    prepared,
                    repeated_device,
                    state,
                    tolerance,
                    value,
                )
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            peak_increment = max(0, torch.cuda.max_memory_allocated() - baseline_allocated)
    return {
        "packed_artifact": str(packed.root),
        "packed_descriptor_sha256": hash_file(packed.root / "nanoquant-packed-model.json"),
        "backend_name": backend.name,
        "backend_version": backend.version,
        "packed_layout": backend.packed_layout,
        "reference_cuda_sha256": backend.reference_cuda_sha256,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "input_dtype": args.input_dtype,
        "tokens": args.tokens,
        "block_count": len(selected_blocks),
        "layer_count": layer_count,
        "shape_inventory": _shape_inventory(selected_blocks),
        "scale_dtypes": sorted(
            {entry.spec.scale_dtype for block in selected_blocks for entry in block.layers}
        ),
        "outlier_counts": sorted(
            {entry.spec.outlier_count for block in selected_blocks for entry in block.layers}
        ),
        "outlier_value_dtypes": sorted(
            {
                entry.spec.outlier_value_dtype
                for block in selected_blocks
                for entry in block.layers
                if entry.spec.outlier_value_dtype is not None
            }
        ),
        "compared_elements": compared_elements,
        "deterministic_replay": True,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "maximum_normalized_error": maximum_normalized_error,
        "maximum_error_layer": maximum_error_layer,
        "peak_cuda_allocated_increment_bytes": peak_increment,
        "wall_seconds": time.perf_counter() - started,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-dtype", choices=tuple(_INPUT_DTYPES), default="bfloat16")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--absolute-tolerance", type=float, default=0.03125)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--triton-cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tokens <= 0:
        raise ValueError("CUDA validation token count must be positive")
    if args.blocks is not None and args.blocks <= 0:
        raise ValueError("CUDA validation block count must be positive")
    if args.absolute_tolerance < 0 or args.relative_tolerance < 0:
        raise ValueError("CUDA validation tolerances must be non-negative")
    if args.triton_cache is not None:
        cache = args.triton_cache.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache)

    result = _validate(args)
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
