"""Run a leased modified llama.cpp benchmark and retain identity-bound JSON samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float]:
    if not values or any(value <= 0 for value in values):
        raise ValueError("llama.cpp benchmark samples must be positive")
    return {
        "min": min(values),
        "p10": _percentile(values, 0.10),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _summarize(raw: list[dict[str, Any]], repetitions: int) -> list[dict[str, Any]]:
    summaries = []
    for index, case in enumerate(raw):
        samples_ns = case.get("samples_ns")
        samples_ts = case.get("samples_ts")
        if (
            not isinstance(samples_ns, list)
            or not isinstance(samples_ts, list)
            or len(samples_ns) != repetitions
            or len(samples_ts) != repetitions
        ):
            raise ValueError(f"llama.cpp benchmark case {index} has incomplete samples")
        latencies = [float(value) / 1_000_000_000.0 for value in samples_ns]
        throughputs = [float(value) for value in samples_ts]
        summaries.append(
            {
                "n_prompt": int(case["n_prompt"]),
                "n_gen": int(case["n_gen"]),
                "latency_seconds": _distribution(latencies),
                "throughput_tokens_per_second": _distribution(throughputs),
                "samples_seconds": latencies,
                "samples_tokens_per_second": throughputs,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--generated-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ubatch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="off")
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.prompt_tokens,
        args.generated_tokens,
        args.repetitions,
        args.batch_size,
        args.ubatch_size,
        args.threads,
    ) <= 0:
        parser.error("benchmark dimensions, repetitions, and thread count must be positive")
    binary = args.binary.resolve()
    model = args.model.resolve()
    command = [
        str(binary),
        "-m",
        str(model),
        "-p",
        str(args.prompt_tokens),
        "-n",
        str(args.generated_tokens),
        "-r",
        str(args.repetitions),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-t",
        str(args.threads),
        "-ngl",
        str(args.gpu_layers),
        "-ctk",
        args.cache_type_k,
        "-ctv",
        args.cache_type_v,
        "-fa",
        args.flash_attn,
        "-o",
        "json",
    ]
    started = time.perf_counter()
    with wait_for_device_lease(args.device, args.wait_for_device_seconds):
        process = subprocess.run(
            command,
            cwd=binary.parent,
            text=True,
            capture_output=True,
            check=False,
            timeout=600,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"llama.cpp benchmark failed ({process.returncode}):\n{process.stderr}"
        )
    try:
        raw = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("llama.cpp benchmark stdout is not JSON") from error
    if not isinstance(raw, list) or any(not isinstance(case, dict) for case in raw):
        raise ValueError("llama.cpp benchmark output must be an array of cases")
    expected = {(args.prompt_tokens, 0), (0, args.generated_tokens)}
    actual = {(int(case["n_prompt"]), int(case["n_gen"])) for case in raw}
    if actual != expected:
        raise ValueError(f"llama.cpp benchmark cases differ: {actual} != {expected}")
    adjacent = tuple(
        sorted(
            path
            for path in binary.parent.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".dll"
            and path.name in ("ggml-cpu.dll", "ggml-cuda.dll", "llama.dll")
        )
    )
    result = {
        "schema_version": 1,
        "binary": str(binary),
        "binary_sha256": _hash_file(binary),
        "backend_libraries": {path.name: _hash_file(path) for path in adjacent},
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": _hash_file(model),
        "device_lease": args.device,
        "command": command,
        "configuration": {
            "prompt_tokens": args.prompt_tokens,
            "generated_tokens": args.generated_tokens,
            "repetitions": args.repetitions,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "threads": args.threads,
            "gpu_layers": args.gpu_layers,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "flash_attention": args.flash_attn,
            "warmup": True,
        },
        "summaries": _summarize(raw, args.repetitions),
        "raw": raw,
        "stderr": process.stderr,
        "wall_seconds": time.perf_counter() - started,
        "passed": True,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
