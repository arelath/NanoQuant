"""Validate separate prefill/decode execution plans on a complete packed artifact."""

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
from nanoquant.runtime import (
    CudaPackedBackend,
    WorkloadSpec,
    open_packed_artifact,
    plan_execution_workloads,
    prepare_execution_workloads,
)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    artifact = open_packed_artifact(args.packed_artifact, verify_hashes=True)
    entries = tuple(entry for block in artifact.manifest.blocks for entry in block.layers)
    states = {entry.spec.name: artifact.load_layer(entry.spec.name) for entry in entries}
    specs = tuple(entry.spec for entry in entries)
    backend = CudaPackedBackend()
    prefill_workload = WorkloadSpec(
        "prefill",
        "cuda",
        args.input_dtype,
        args.batch_size,
        args.prefill_tokens,
        deterministic=True,
    )
    decode_workload = WorkloadSpec(
        "decode",
        "cuda",
        args.input_dtype,
        args.batch_size,
        1,
        deterministic=True,
    )
    plans = plan_execution_workloads(
        specs,
        prefill=prefill_workload,
        decode=decode_workload,
        prefill_backends=(backend,),
        decode_backends=(backend,),
        strict=True,
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("runtime execution-plan validation requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _DTYPES[args.input_dtype]
    generator = torch.Generator().manual_seed(args.seed)
    started = time.perf_counter()
    with wait_for_device_lease(str(device), args.wait_for_device_seconds):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            baseline_allocated = torch.cuda.memory_allocated()
        prepared = prepare_execution_workloads(plans, states, (backend,), device)
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            prepared_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        shared_layers = sum(
            prefill.layer is decode.layer
            for prefill, decode in zip(
                prepared.prefill.dispatches,
                prepared.decode.dispatches,
                strict=True,
            )
        )
        prefill_checksum = torch.zeros((), dtype=torch.float64, device=device)
        decode_checksum = torch.zeros((), dtype=torch.float64, device=device)
        prefill_elements = 0
        decode_elements = 0
        for layer_index, entry in enumerate(entries):
            prefill_value = torch.randn(
                args.batch_size,
                args.prefill_tokens,
                entry.spec.in_features,
                generator=generator,
                dtype=dtype,
            ).to(device)
            prefill_output = prepared.prefill.linear_at(layer_index, prefill_value)
            expected_prefill_shape = (
                args.batch_size,
                args.prefill_tokens,
                entry.spec.out_features,
            )
            if tuple(prefill_output.shape) != expected_prefill_shape:
                raise ValueError(f"prefill output shape differs: {entry.spec.name}")
            prefill_checksum.add_(prefill_output.double().sum())
            prefill_elements += prefill_output.numel()

            decode_value = torch.randn(
                args.batch_size,
                entry.spec.in_features,
                generator=generator,
                dtype=dtype,
            ).to(device)
            decode_output = prepared.decode.linear_at(layer_index, decode_value)
            expected_decode_shape = (args.batch_size, entry.spec.out_features)
            if tuple(decode_output.shape) != expected_decode_shape:
                raise ValueError(f"decode output shape differs: {entry.spec.name}")
            decode_checksum.add_(decode_output.double().sum())
            decode_elements += decode_output.numel()
            del decode_output, decode_value, prefill_output, prefill_value
        torch.cuda.synchronize(device)
        prefill_checksum_value = float(prefill_checksum.cpu())
        decode_checksum_value = float(decode_checksum.cpu())
        with torch.cuda.device(device):
            execution_peak_increment = max(
                0,
                torch.cuda.max_memory_allocated() - prepared_allocated,
            )
    if shared_layers != len(entries):
        raise ValueError(
            f"prefill/decode plans share {shared_layers} prepared layers, expected {len(entries)}"
        )
    return {
        "packed_artifact": str(artifact.root),
        "packed_descriptor_sha256": hash_file(artifact.root / "nanoquant-packed-model.json"),
        "backend_name": backend.name,
        "backend_version": backend.version,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "input_dtype": args.input_dtype,
        "batch_size": args.batch_size,
        "prefill_tokens": args.prefill_tokens,
        "decode_tokens": 1,
        "block_count": len(artifact.manifest.blocks),
        "layer_count": len(entries),
        "prefill_fallback_count": plans.prefill.fallback_count,
        "decode_fallback_count": plans.decode.fallback_count,
        "prefill_backend_inventory": sorted({item.backend_name for item in plans.prefill.layers}),
        "decode_backend_inventory": sorted({item.backend_name for item in plans.decode.layers}),
        "shared_prepared_layer_count": shared_layers,
        "prepared_layer_count": len(entries),
        "preparation_allocated_increment_bytes": max(0, prepared_allocated - baseline_allocated),
        "execution_peak_allocated_increment_bytes": execution_peak_increment,
        "prefill_output_elements": prefill_elements,
        "decode_output_elements": decode_elements,
        "prefill_checksum": prefill_checksum_value,
        "decode_checksum": decode_checksum_value,
        "wall_seconds": time.perf_counter() - started,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-dtype", choices=tuple(_DTYPES), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--triton-cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.prefill_tokens <= 0:
        raise ValueError("runtime execution-plan batch and prefill tokens must be positive")
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
