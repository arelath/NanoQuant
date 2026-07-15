"""Benchmark repeated prompt-identical llama.cpp generation with parsed stage timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PROMPT = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+) ms\s*/\s*([0-9]+) tokens.*?\(\s*([0-9.]+) tokens per second"
)
_DECODE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([0-9.]+) ms\s*/\s*([0-9]+) runs.*?\(\s*([0-9.]+) tokens per second"
)
_SUMMARY = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+) t/s\s*\|\s*Generation:\s*([0-9.]+) t/s\s*\]"
)


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
        raise ValueError("llama.cpp generation samples must be positive")
    return {
        "min": min(values),
        "p10": _percentile(values, 0.10),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _stage(stderr: str, pattern: re.Pattern[str], name: str) -> tuple[float, int, float]:
    match = pattern.search(_ANSI.sub("", stderr))
    if match is None:
        raise ValueError(f"llama.cpp generation output has no {name} timing")
    return float(match.group(1)) / 1000.0, int(match.group(2)), float(match.group(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--generated-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ubatch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="off")
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.generated_tokens,
        args.prompt_tokens,
        args.repetitions,
        args.context_size,
        args.batch_size,
        args.ubatch_size,
        args.threads,
    ) <= 0:
        parser.error("generation and execution dimensions must be positive")
    binary = args.binary.resolve()
    model = args.model.resolve()
    command = [
        str(binary),
        "-m",
        str(model),
        "-p",
        args.prompt,
        "-n",
        str(args.generated_tokens),
        "--temp",
        "0",
        "--seed",
        "1",
        "-c",
        str(args.context_size),
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
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
    ]
    prompt_seconds: list[float] = []
    prompt_rates: list[float] = []
    decode_seconds: list[float] = []
    decode_rates: list[float] = []
    process_seconds: list[float] = []
    texts = []
    raw = []
    started = time.perf_counter()
    with wait_for_device_lease(args.device, args.wait_for_device_seconds):
        for _ in range(args.repetitions):
            process_started = time.perf_counter()
            process = subprocess.run(
                command,
                cwd=binary.parent,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=300,
            )
            process_seconds.append(time.perf_counter() - process_started)
            if process.returncode != 0:
                raise RuntimeError(
                    f"llama.cpp generation failed ({process.returncode}):\n{process.stderr}"
                )
            combined = _ANSI.sub("", process.stdout + "\n" + process.stderr)
            summary = _SUMMARY.search(combined)
            if summary is None:
                prompt_time, prompt_tokens, prompt_rate = _stage(
                    process.stderr, _PROMPT, "prompt"
                )
                decode_time, decode_runs, decode_rate = _stage(
                    process.stderr, _DECODE, "decode"
                )
            else:
                prompt_tokens = args.prompt_tokens
                decode_runs = args.generated_tokens
                prompt_rate = float(summary.group(1))
                decode_rate = float(summary.group(2))
                prompt_time = prompt_tokens / prompt_rate
                decode_time = decode_runs / decode_rate
            prompt_seconds.append(prompt_time)
            prompt_rates.append(prompt_rate)
            decode_seconds.append(decode_time)
            decode_rates.append(decode_rate)
            clean_stdout = _ANSI.sub("", process.stdout)
            summary_start = clean_stdout.rfind("\n[ Prompt:")
            response_region = clean_stdout if summary_start < 0 else clean_stdout[:summary_start]
            prompt_marker = f"> {args.prompt}\n\n"
            marker_start = response_region.rfind(prompt_marker)
            if marker_start < 0:
                raise ValueError("llama.cpp output has no rendered prompt boundary")
            text = response_region[marker_start + len(prompt_marker) :].strip()
            texts.append(text)
            raw.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "decode_runs": decode_runs,
                    "prompt_seconds": prompt_time,
                    "prompt_tokens_per_second": prompt_rate,
                    "decode_seconds": decode_time,
                    "decode_tokens_per_second": decode_rate,
                    "process_seconds": process_seconds[-1],
                }
            )
    if len(set(texts)) != 1:
        raise ValueError("llama.cpp generated text changed across repetitions")
    reference = None
    if args.reference_output is not None:
        payload = json.loads(args.reference_output.read_text(encoding="utf-8"))
        expected = payload.get("generated")
        if not isinstance(expected, str) or not texts[0].startswith(expected):
            raise ValueError(
                f"llama.cpp generation no longer matches the retained reference prefix: "
                f"{texts[0]!r}"
            )
        reference = {
            "path": str(args.reference_output.resolve()),
            "text": expected,
            "exact_prefix": True,
        }
    result = {
        "schema_version": 1,
        "binary": str(binary),
        "binary_sha256": _hash_file(binary),
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": _hash_file(model),
        "command": command,
        "configuration": {
            "prompt": args.prompt,
            "prompt_tokens": args.prompt_tokens,
            "generated_tokens": args.generated_tokens,
            "repetitions": args.repetitions,
            "context_size": args.context_size,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "threads": args.threads,
            "gpu_layers": args.gpu_layers,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "flash_attention": args.flash_attn,
            "temperature": 0,
            "seed": 1,
            "warmup": True,
        },
        "prompt_latency_seconds": _distribution(prompt_seconds),
        "prompt_throughput_tokens_per_second": _distribution(prompt_rates),
        "decode_latency_seconds": _distribution(decode_seconds),
        "decode_throughput_tokens_per_second": _distribution(decode_rates),
        "process_wall_seconds": _distribution(process_seconds),
        "generated_text": texts[0],
        "reference": reference,
        "raw": raw,
        "wall_seconds": time.perf_counter() - started,
        "passed": True,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
