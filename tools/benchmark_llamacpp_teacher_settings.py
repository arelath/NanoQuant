"""Benchmark llama.cpp teacher runtime settings against retained teacher responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from nanoquant.config.schema import ReasoningMode
from nanoquant.infrastructure.chat_behaviors import chat_behavior_for_snapshot
from nanoquant.infrastructure.environment import load_repository_dotenv
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.llamacpp_teacher_generation import (
    LlamaCppTeacherRuntimeOptions,
    open_llamacpp_teacher_session,
)
from nanoquant.teacher_dataset import (
    load_teacher_dataset_settings,
    resolve_teacher_snapshot,
    resolve_tokenizer_snapshot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--candidate", action="append", dest="candidates")
    return parser


def _accepted_thinking_records(root: Path, count: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for journal in sorted((root / "state" / "teacher-traces").glob("*.jsonl")):
        for line in journal.read_text(encoding="utf-8").splitlines()[1:]:
            value = json.loads(line)
            messages = value.get("messages")
            if (
                value.get("status") == "accepted"
                and isinstance(messages, list)
                and messages
                and isinstance(messages[-1], dict)
                and str(messages[-1].get("reasoning_content") or "").strip()
            ):
                records.append(value)
                if len(records) == count:
                    return records
    raise ValueError(f"benchmark needs {count} retained thinking responses; found {len(records)}")


def _candidates() -> tuple[tuple[str, LlamaCppTeacherRuntimeOptions], ...]:
    return (
        (
            "baseline",
            LlamaCppTeacherRuntimeOptions(
                flash_attention="auto",
                fit_target_mib=1024,
                parallelism=1,
            ),
        ),
        (
            "flash-f16-fit1024",
            LlamaCppTeacherRuntimeOptions(flash_attention="on", parallelism=1),
        ),
        (
            "flash-f16-fit768",
            LlamaCppTeacherRuntimeOptions(
                flash_attention="on",
                fit_target_mib=768,
                parallelism=1,
            ),
        ),
        (
            "flash-f16-fit512",
            LlamaCppTeacherRuntimeOptions(
                flash_attention="on",
                fit_target_mib=512,
                parallelism=1,
            ),
        ),
        (
            "flash-q8-fit1024",
            LlamaCppTeacherRuntimeOptions(
                flash_attention="on",
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                parallelism=1,
            ),
        ),
        (
            "flash-q8-fit768",
            LlamaCppTeacherRuntimeOptions(
                flash_attention="on",
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                fit_target_mib=768,
                parallelism=1,
            ),
        ),
        (
            "flash-q8k-f16v-fit768",
            LlamaCppTeacherRuntimeOptions(
                flash_attention="on",
                cache_type_k="q8_0",
                cache_type_v="f16",
                fit_target_mib=768,
                parallelism=1,
            ),
        ),
        (
            "flash-f16k-q8v-fit768",
            LlamaCppTeacherRuntimeOptions(
                flash_attention="on",
                cache_type_k="f16",
                cache_type_v="q8_0",
                fit_target_mib=768,
                parallelism=1,
            ),
        ),
        (
            "flash-q8-fit512",
            LlamaCppTeacherRuntimeOptions(
                flash_attention="on",
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                fit_target_mib=512,
                parallelism=1,
            ),
        ),
    )


def _interesting_log(line: str) -> bool:
    lowered = line.lower()
    return any(
        marker in lowered
        for marker in ("offload", "flash", "kv buffer", "fit", "cuda0", "model buffer")
    )


def main() -> int:
    args = _parser().parse_args()
    settings_path = args.settings.resolve()
    settings, settings_hash = load_teacher_dataset_settings(settings_path)
    if settings.teacher.gguf_filename is None:
        raise ValueError("benchmark requires a prebuilt teacher GGUF")
    load_repository_dotenv(Path(__file__).resolve().parents[1])
    snapshot = resolve_teacher_snapshot(settings.teacher)
    tokenizer_snapshot = resolve_tokenizer_snapshot(
        settings.teacher.tokenizer_source,
        settings.teacher.tokenizer_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=False)
    behavior = chat_behavior_for_snapshot(tokenizer_snapshot)
    retained = _accepted_thinking_records(settings_path.parent, args.prompts)
    prompts: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for record in retained:
        messages = record["messages"]
        if not isinstance(messages, list):
            raise ValueError("retained benchmark record has invalid messages")
        prompt = messages[:-1]
        prompt_ids = behavior.render_generation_prompt(
            tokenizer,
            prompt,
            ReasoningMode.THINKING,
        )
        complete = behavior.render_completed(
            tokenizer,
            messages,
            ReasoningMode.THINKING,
            assistant_target_weight=1.0,
            prompt_target_weight=0.0,
        ).input_ids
        response_tokens = int(record["response_tokens"])
        reference = tuple(
            complete[
                len(prompt_ids) : len(prompt_ids) + min(args.tokens, response_tokens)
            ]
        )
        prompts.append((prompt_ids, reference))

    results: list[dict[str, object]] = []
    candidates = tuple(
        (name, options)
        for name, options in _candidates()
        if args.candidates is None or name in args.candidates
    )
    if not candidates:
        raise ValueError("no benchmark candidates matched --candidate")
    for name, options in candidates:
        print(f"Benchmarking {name}: {asdict(options)}", flush=True)
        logs: list[str] = []

        def record_log(line: str, destination: list[str] = logs) -> None:
            if _interesting_log(line):
                destination.append(line)

        started = time.perf_counter()
        try:
            prompt_results: list[dict[str, object]] = []
            with open_llamacpp_teacher_session(
                snapshot,
                tokenizer,
                device=settings.teacher.device,
                sequence_length=settings.generation.sequence_length,
                gguf_path=snapshot / settings.teacher.gguf_filename,
                runtime_options=options,
                server_log=record_log,
            ) as session:
                startup_seconds = time.perf_counter() - started
                generation_started = time.perf_counter()
                total_tokens = 0
                for prompt_ids, reference in prompts:
                    prompt_started = time.perf_counter()
                    complete = session.generate(prompt_ids, args.tokens)
                    generated = complete[len(prompt_ids) :]
                    elapsed = time.perf_counter() - prompt_started
                    total_tokens += len(generated)
                    mismatch_index = next(
                        (
                            index
                            for index, (actual, expected) in enumerate(
                                zip(generated, reference, strict=False)
                            )
                            if actual != expected
                        ),
                        min(len(generated), len(reference))
                        if len(generated) != len(reference)
                        else None,
                    )
                    prompt_results.append(
                        {
                            "prompt_tokens": len(prompt_ids),
                            "generated_tokens": len(generated),
                            "reference_tokens": len(reference),
                            "elapsed_seconds": elapsed,
                            "tokens_per_second": len(generated) / elapsed,
                            "matches_reference": generated == reference,
                            "first_mismatch_token": mismatch_index,
                            "generated_text": (
                                tokenizer.decode(generated, skip_special_tokens=False)
                                if mismatch_index is not None
                                else None
                            ),
                            "reference_text": (
                                tokenizer.decode(reference, skip_special_tokens=False)
                                if mismatch_index is not None
                                else None
                            ),
                            "token_hash": hashlib.sha256(
                                json.dumps(generated, separators=(",", ":")).encode("utf-8")
                            ).hexdigest(),
                        }
                    )
                generation_seconds = time.perf_counter() - generation_started
            results.append(
                {
                    "name": name,
                    "status": "completed",
                    "options": asdict(options),
                    "startup_seconds": startup_seconds,
                    "generation_seconds": generation_seconds,
                    "generated_tokens": total_tokens,
                    "tokens_per_second": total_tokens / generation_seconds,
                    "all_match_reference": all(
                        bool(value["matches_reference"]) for value in prompt_results
                    ),
                    "prompts": prompt_results,
                    "server_log": logs,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "status": "failed",
                    "options": asdict(options),
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "server_log": logs,
                }
            )
        output = (
            args.output.resolve()
            if args.output is not None
            else settings_path.parent / "llamacpp-teacher-benchmark.json"
        )
        atomic_write_json(
            output,
            {
                "schema_version": 1,
                "settings": str(settings_path),
                "settings_hash": settings_hash,
                "teacher_source": settings.teacher.source,
                "teacher_revision": settings.teacher.revision,
                "gguf": settings.teacher.gguf_filename,
                "gguf_bytes": (snapshot / settings.teacher.gguf_filename).stat().st_size,
                "device": settings.teacher.device,
                "sequence_length": settings.generation.sequence_length,
                "prompt_count": len(prompts),
                "tokens_per_prompt": args.tokens,
                "python": platform.python_version(),
                "results": results,
            },
        )
        print(f"Recorded {name} in {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
