"""Durable source-model generation of complete Qwen behavior turns."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn
from transformers.models.auto.configuration_auto import AutoConfig

from nanoquant.config.codec import semantic_hash, to_dict
from nanoquant.config.schema import BehaviorSliceConfig, ReasoningMode
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.memory_cleanup import release_memory
from nanoquant.ports.chat_behavior import ChatBehaviorPort, RenderedBehaviorRecord, tensor_sha256

TeacherTraceProgress = Callable[[str, Mapping[str, object]], None]


class GenerateTokens(Protocol):
    def __call__(self, prompt: tuple[int, ...], maximum_new_tokens: int) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class PreparedTeacherTraces:
    messages: tuple[tuple[dict[str, object], ...], ...]
    reference: ArtifactRef
    identity: str
    ordered_response_hashes: tuple[str, ...]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _decode(tokenizer: Any, values: tuple[int, ...]) -> str:
    try:
        return str(
            tokenizer.decode(
                values,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(values))


def _subsequence_positions(values: tuple[int, ...], needle: tuple[int, ...]) -> tuple[int, ...]:
    if not needle:
        raise ValueError("teacher trace delimiter tokenization is empty")
    return tuple(
        index
        for index in range(len(values) - len(needle) + 1)
        if values[index : index + len(needle)] == needle
    )


def _ultrachat_prompt(record: dict[str, object]) -> list[dict[str, object]]:
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        raise ValueError("UltraChat record has no complete user/assistant turn")
    messages: list[dict[str, object]] = []
    for message in raw_messages:
        if isinstance(message, dict):
            messages.append(
                {
                    "role": str(message.get("role") or ""),
                    "content": str(message.get("content") or ""),
                }
            )
    if len(messages) != len(raw_messages):
        raise ValueError("UltraChat record contains a malformed message")
    if messages[-1]["role"] != "assistant":
        raise ValueError("UltraChat record does not end in an assistant response")
    prompt = messages[:-1]
    if not prompt or prompt[-1]["role"] != "user":
        raise ValueError("UltraChat generation prompt does not end in a user turn")
    if any(not str(message["content"]).strip() for message in prompt):
        raise ValueError("UltraChat generation prompt contains empty content")
    return prompt


def _completed_teacher_turn(
    tokenizer: Any,
    behavior: ChatBehaviorPort,
    prompt: list[dict[str, object]],
    prompt_ids: tuple[int, ...],
    complete_ids: tuple[int, ...],
    eos_token_ids: frozenset[int],
    mode: ReasoningMode,
) -> tuple[list[dict[str, object]], RenderedBehaviorRecord, tuple[int, ...]]:
    if complete_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("teacher generation changed the pinned prompt prefix")
    generated = complete_ids[len(prompt_ids) :]
    if not generated:
        raise ValueError("teacher generated no response tokens")
    if generated[-1] not in eos_token_ids:
        raise ValueError("teacher response stopped at the token limit")

    answer_end = len(complete_ids)
    while answer_end > len(prompt_ids) and complete_ids[answer_end - 1] in eos_token_ids:
        answer_end -= 1
    response_without_eos = complete_ids[len(prompt_ids) : answer_end]
    opening = tuple(int(value) for value in tokenizer.encode("<think>", add_special_tokens=False))
    closing = tuple(int(value) for value in tokenizer.encode("</think>", add_special_tokens=False))
    reasoning = ""
    if mode is ReasoningMode.THINKING:
        opening_positions = _subsequence_positions(complete_ids, opening)
        closing_positions = tuple(
            index
            for index in _subsequence_positions(complete_ids, closing)
            if index >= len(prompt_ids)
        )
        eligible_openings = (
            tuple(
                index
                for index in opening_positions
                if index + len(opening) >= len(prompt_ids) and index < closing_positions[0]
            )
            if closing_positions
            else ()
        )
        if len(closing_positions) != 1 or len(eligible_openings) != 1:
            raise ValueError("teacher response has missing or repeated thinking delimiters")
        open_index = eligible_openings[0]
        close_index = closing_positions[0]
        reasoning_ids = complete_ids[open_index + len(opening) : close_index]
        answer_ids = complete_ids[close_index + len(closing) : answer_end]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
    elif mode is ReasoningMode.NON_THINKING:
        if (
            _subsequence_positions(response_without_eos, opening)
            or _subsequence_positions(response_without_eos, closing)
        ):
            raise ValueError("non-thinking teacher response emitted reasoning delimiters")
        answer_ids = response_without_eos
    else:
        raise ValueError("teacher generation requires a chat reasoning mode")
    answer = _decode(tokenizer, answer_ids).strip()
    if (mode is ReasoningMode.THINKING and not reasoning) or not answer:
        raise ValueError("teacher response lacks reasoning or a final answer")

    messages: list[dict[str, object]] = list(prompt)
    response: dict[str, object] = {"role": "assistant", "content": answer}
    if mode is ReasoningMode.THINKING:
        response["reasoning_content"] = reasoning
    messages.append(response)
    rendered = behavior.render_completed(
        tokenizer,
        messages,
        mode,
        assistant_target_weight=1.0,
        prompt_target_weight=0.0,
    )
    rendered_ids = tuple(rendered.input_ids)
    if rendered_ids != complete_ids:
        trailing = rendered_ids[len(complete_ids) :] if rendered_ids[: len(complete_ids)] == complete_ids else ()
        if not trailing or _decode(tokenizer, trailing).strip():
            raise ValueError("teacher response does not round-trip through the pinned chat template")
    return messages, rendered, generated


def _checkpoint_dtype(config: Any) -> torch.dtype:
    value = str(getattr(config, "torch_dtype", "")).lower()
    if "float16" in value and "bfloat16" not in value:
        return torch.float16
    if "float32" in value:
        return torch.float32
    return torch.bfloat16


def _eos_token_ids(model: nn.Module, tokenizer: Any) -> frozenset[int]:
    values: list[int] = []
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(configured, int):
        values.append(configured)
    elif isinstance(configured, (list, tuple)):
        values.extend(int(value) for value in configured)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos, int):
        values.append(tokenizer_eos)
    if not values:
        raise ValueError("teacher generation requires at least one EOS token ID")
    return frozenset(values)


@contextmanager
def _open_generation_session(
    snapshot: Path,
    tokenizer: Any,
    device: str,
) -> Iterator[tuple[GenerateTokens, frozenset[int]]]:
    config = AutoConfig.from_pretrained(snapshot, local_files_only=False)
    model_type = str(getattr(config, "model_type", "")).lower()
    attention = "eager" if model_type.startswith("gemma") else "sdpa"
    with acquire_device_lease(device):
        loaded_model = load_causal_language_model(
            snapshot,
            torch_dtype=_checkpoint_dtype(config),
            attention_implementation=attention,
        ).eval().to(device)
        model_holder = [loaded_model]
        eos_ids = _eos_token_ids(loaded_model, tokenizer)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_token_id, int):
            pad_token_id = next(iter(eos_ids))

        def generate(prompt: tuple[int, ...], maximum_new_tokens: int) -> tuple[int, ...]:
            input_ids = torch.tensor((prompt,), dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            with torch.inference_mode():
                output = cast(
                    torch.Tensor,
                    cast(Any, model_holder[0]).generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=maximum_new_tokens,
                        use_cache=True,
                        pad_token_id=pad_token_id,
                        eos_token_id=sorted(eos_ids),
                    ),
                )
            return tuple(int(value) for value in output[0].detach().cpu().tolist())

        try:
            yield generate, eos_ids
        finally:
            model_holder.clear()
            del loaded_model
            release_memory(device)


def _append_journal(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(encoded + "\n")
        output.flush()
        os.fsync(output.fileno())


def _load_journal(path: Path, identity: str) -> list[dict[str, object]]:
    if not path.exists():
        _append_journal(path, {"schema_version": 1, "kind": "header", "identity": identity})
        return []
    raw = path.read_bytes()
    complete_bytes = raw[: raw.rfind(b"\n") + 1] if b"\n" in raw else b""
    if complete_bytes != raw:
        with path.open("r+b") as output:
            output.truncate(len(complete_bytes))
            output.flush()
            os.fsync(output.fileno())
    lines = complete_bytes.decode("utf-8").splitlines()
    if not lines:
        _append_journal(path, {"schema_version": 1, "kind": "header", "identity": identity})
        return []
    values = [json.loads(line) for line in lines]
    header = values[0]
    if not isinstance(header, dict) or header.get("kind") != "header" or header.get("identity") != identity:
        raise ValueError("teacher-trace journal identity does not match the requested generation")
    records = values[1:]
    if any(not isinstance(value, dict) or value.get("kind") != "attempt" for value in records):
        raise ValueError("teacher-trace journal contains an invalid record")
    return cast(list[dict[str, object]], records)


def _load_artifact(
    output: Path,
    identity: str,
    count: int,
) -> PreparedTeacherTraces | None:
    receipt_path = output / "state" / "teacher-traces" / f"{identity[7:]}.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError("teacher-trace receipt is not valid JSON") from exc
    if receipt.get("identity") != identity:
        raise ValueError("teacher-trace receipt identity does not match its path")
    receipt_record_count = int(receipt.get("record_count", -1))
    if receipt_record_count < 0:
        raise ValueError("teacher-trace receipt has an invalid record count")
    artifact_id = str(receipt["artifact_id"])
    artifacts = LocalArtifactStore(output / "artifacts")
    descriptor = artifacts.validate(artifact_id)
    if descriptor.artifact_type != "teacher-trace-dataset":
        raise ValueError("teacher-trace receipt references the wrong artifact type")
    root = artifacts.path_for(artifact_id)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    artifact_record_count = int(manifest.get("record_count", -1))
    if manifest.get("identity") != identity or artifact_record_count != receipt_record_count:
        raise ValueError("teacher-trace artifact manifest does not match its receipt")
    records = [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    messages = tuple(
        tuple(cast(list[dict[str, object]], record["messages"]))
        for record in records
    )
    hashes = tuple(str(record["response_token_hash"]) for record in records)
    if len(messages) != artifact_record_count:
        raise ValueError("teacher-trace artifact contains the wrong record count")
    if artifact_record_count < count:
        return None
    return PreparedTeacherTraces(
        messages[:count],
        ArtifactRef("teacher-trace-dataset", artifact_id, 1),
        identity,
        hashes[:count],
    )


def prepare_teacher_traces(
    snapshot: str | Path,
    output: str | Path,
    item: BehaviorSliceConfig,
    tokenizer: Any,
    behavior: ChatBehaviorPort,
    records: Iterable[dict[str, object]],
    *,
    teacher_source: str,
    teacher_revision: str,
    count: int,
    sequence_length: int,
    seed: int,
    device: str,
    progress: TeacherTraceProgress | None = None,
) -> PreparedTeacherTraces:
    """Generate, validate, checkpoint, and commit coherent teacher turns."""

    generation = item.teacher_trace_generation
    if generation is None:
        raise ValueError("teacher-trace preparation requires generation configuration")
    identity_payload = {
        "implementation": generation.implementation,
        "teacher": {"source": teacher_source, "revision": teacher_revision},
        "prompt_source": to_dict(item.source),
        "partition": item.partition,
        "mode": item.mode.value,
        "chat_policy": behavior.policy_identity(tokenizer),
        "sequence_length": sequence_length,
        "seed": seed,
        "generation": to_dict(generation),
    }
    identity = semantic_hash(identity_payload)
    output_path = Path(output)
    reused = _load_artifact(output_path, identity, count)
    if reused is not None:
        if progress is not None:
            progress(
                "teacher_trace_cache_reused",
                {"slice": item.name, "partition": item.partition, "record_count": count},
            )
        return reused

    state_root = output_path / "state" / "teacher-traces"
    journal_path = state_root / f"{identity[7:]}.jsonl"
    journal = _load_journal(journal_path, identity)
    accepted = [record for record in journal if record.get("status") == "accepted"]
    attempted = {str(record.get("source_hash")) for record in journal}
    maximum_attempts = count * generation.maximum_attempt_multiplier
    attempts = len(journal)
    started = time.perf_counter()
    if progress is not None:
        progress(
            "teacher_trace_generation_started",
            {
                "slice": item.name,
                "partition": item.partition,
                "teacher_source": teacher_source,
                "teacher_revision": teacher_revision,
                "accepted_records": len(accepted),
                "attempted_records": attempts,
                "target_records": count,
            },
        )

    if len(accepted) < count and attempts < maximum_attempts:
        with _open_generation_session(Path(snapshot), tokenizer, device) as (generate, eos_ids):
            for source_record in records:
                source_hash = _canonical_hash(source_record)
                if source_hash in attempted:
                    continue
                attempted.add(source_hash)
                attempts += 1
                attempt_started = time.perf_counter()
                result: dict[str, object] = {
                    "schema_version": 1,
                    "kind": "attempt",
                    "attempt": attempts,
                    "source_hash": source_hash,
                }
                try:
                    prompt = _ultrachat_prompt(source_record)
                    prompt_ids = behavior.render_generation_prompt(
                        tokenizer,
                        prompt,
                        item.mode,
                    )
                    available = sequence_length - len(prompt_ids)
                    maximum_new_tokens = min(generation.maximum_new_tokens, available)
                    if maximum_new_tokens < generation.minimum_new_tokens:
                        raise ValueError("teacher prompt leaves too little room for a complete response")
                    if progress is not None:
                        progress(
                            "teacher_trace_prompt_started",
                            {
                                "slice": item.name,
                                "partition": item.partition,
                                "attempt": attempts,
                                "accepted_records": len(accepted),
                                "target_records": count,
                                "prompt_tokens": len(prompt_ids),
                                "maximum_new_tokens": maximum_new_tokens,
                            },
                        )
                    complete_ids = generate(prompt_ids, maximum_new_tokens)
                    messages, rendered, generated_ids = _completed_teacher_turn(
                        tokenizer,
                        behavior,
                        prompt,
                        prompt_ids,
                        complete_ids,
                        eos_ids,
                        item.mode,
                    )
                    if len(generated_ids) < generation.minimum_new_tokens:
                        raise ValueError("teacher response is shorter than the configured minimum")
                    if len(rendered.input_ids) > sequence_length:
                        raise ValueError("teacher response exceeds the behavior sequence length")
                    response_hash = tensor_sha256(torch.tensor(generated_ids, dtype=torch.long))
                    result.update(
                        {
                            "status": "accepted",
                            "messages": messages,
                            "prompt_token_hash": tensor_sha256(
                                torch.tensor(prompt_ids, dtype=torch.long)
                            ),
                            "response_token_hash": response_hash,
                            "complete_token_hash": tensor_sha256(
                                torch.tensor(complete_ids, dtype=torch.long)
                            ),
                            "prompt_tokens": len(prompt_ids),
                            "response_tokens": len(generated_ids),
                            "stop_reason": "eos",
                        }
                    )
                except (TypeError, ValueError) as exc:
                    result.update(
                        {
                            "status": "rejected",
                            "reason_type": type(exc).__name__,
                            "reason": str(exc),
                        }
                    )
                result["elapsed_seconds"] = time.perf_counter() - attempt_started
                _append_journal(journal_path, result)
                if result["status"] == "accepted":
                    accepted.append(result)
                if progress is not None and (
                    result["status"] == "accepted" or attempts == 1 or attempts % 10 == 0
                ):
                    progress(
                        "teacher_trace_generation_progress",
                        {
                            "slice": item.name,
                            "partition": item.partition,
                            "accepted_records": len(accepted),
                            "attempted_records": attempts,
                            "rejected_records": attempts - len(accepted),
                            "target_records": count,
                            "last_status": result["status"],
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                    )
                if len(accepted) >= count or attempts >= maximum_attempts:
                    break
    if len(accepted) < count:
        raise ValueError(
            f"teacher trace generation produced {len(accepted)} complete records; expected {count} "
            f"after {attempts} attempts"
        )

    accepted = accepted[:count]
    artifacts = LocalArtifactStore(output_path / "artifacts")
    with artifacts.begin_write("teacher-trace-dataset") as writer:
        records_path = writer.path / "records.jsonl"
        with records_path.open("w", encoding="utf-8", newline="\n") as records_file:
            for record in accepted:
                records_file.write(
                    json.dumps(
                        {
                            "source_hash": record["source_hash"],
                            "messages": record["messages"],
                            "prompt_token_hash": record["prompt_token_hash"],
                            "response_token_hash": record["response_token_hash"],
                            "complete_token_hash": record["complete_token_hash"],
                            "prompt_tokens": record["prompt_tokens"],
                            "response_tokens": record["response_tokens"],
                            "stop_reason": record["stop_reason"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            records_file.flush()
            os.fsync(records_file.fileno())
        manifest = {
            "schema_version": 1,
            "producer": "teacher-trace-generation-v1",
            "identity": identity,
            "identity_payload": identity_payload,
            "record_count": count,
            "attempt_count": attempts,
            "ordered_source_hashes": [record["source_hash"] for record in accepted],
            "ordered_response_hashes": [record["response_token_hash"] for record in accepted],
        }
        (writer.path / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    atomic_write_json(
        state_root / f"{identity[7:]}.json",
        {
            "schema_version": 1,
            "identity": identity,
            "record_count": count,
            "artifact_id": descriptor.artifact_id,
        },
    )
    if progress is not None:
        progress(
            "teacher_trace_generation_completed",
            {
                "slice": item.name,
                "partition": item.partition,
                "record_count": count,
                "attempt_count": attempts,
                "artifact_id": descriptor.artifact_id,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    prepared_messages = tuple(
        tuple(cast(list[dict[str, object]], record["messages"]))
        for record in accepted
    )
    hashes = tuple(str(record["response_token_hash"]) for record in accepted)
    return PreparedTeacherTraces(
        prepared_messages,
        ArtifactRef("teacher-trace-dataset", descriptor.artifact_id, 1),
        identity,
        hashes,
    )


__all__ = [
    "PreparedTeacherTraces",
    "prepare_teacher_traces",
]
