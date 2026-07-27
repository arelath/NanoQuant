from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import nanoquant.infrastructure.teacher_trace_generation as trace_module
from nanoquant.config.schema import (
    LLAMACPP_TEACHER_TRACE_IMPLEMENTATION,
    BehaviorSliceConfig,
    DatasetSourceConfig,
    ReasoningMode,
    TeacherTraceGenerationConfig,
)
from nanoquant.infrastructure.chat_behaviors import Qwen3ChatBehavior
from nanoquant.infrastructure.teacher_trace_generation import prepare_teacher_traces


class TraceTokenizer:
    chat_template = "fixture-qwen3-template"
    special_tokens_map = {"eos_token": "<|im_end|>"}
    eos_token_id = 1
    pad_token_id = 2

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        values: list[int] = []
        while text:
            if text.startswith("<|im_end|>"):
                values.append(1)
                text = text[len("<|im_end|>") :]
            else:
                values.append(ord(text[0]) + 10)
                text = text[1:]
        return values

    @staticmethod
    def decode(values: object, **_kwargs: object) -> str:
        return "".join(
            "<|im_end|>" if int(value) == 1 else chr(int(value) - 10)
            for value in values  # type: ignore[union-attr]
        )

    @staticmethod
    def convert_tokens_to_ids(value: str) -> int:
        assert value == "<|im_end|>"
        return 1

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        truncation: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert tokenize and not truncation
        rendered = ""
        for message in messages:
            role = str(message["role"])
            rendered += f"<|im_start|>{role}\n"
            if role == "assistant" and enable_thinking:
                rendered += f"<think>\n{message.get('reasoning_content', '')}\n</think>\n\n"
            elif role == "assistant":
                rendered += "<think>\n\n</think>\n\n"
            rendered += str(message.get("content") or "") + "<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
            if not enable_thinking:
                rendered += "<think>\n\n</think>\n\n"
        return self.encode(rendered)


def _slice(mode: ReasoningMode = ReasoningMode.THINKING) -> BehaviorSliceConfig:
    return BehaviorSliceConfig(
        mode.value,
        mode,
        DatasetSourceConfig(
            "HuggingFaceH4/ultrachat_200k",
            revision="ultrachat-revision",
            split="train_sft",
        ),
        "ultrachat_messages",
        1.0,
        teacher_trace_generation=TeacherTraceGenerationConfig(
            maximum_new_tokens=256,
            minimum_new_tokens=1,
            maximum_attempt_multiplier=5,
        ),
    )


def _records() -> tuple[dict[str, object], ...]:
    return (
        {
            "messages": [
                {"role": "user", "content": "First prompt"},
                {"role": "assistant", "content": "discard this dataset answer"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Second prompt"},
                {"role": "assistant", "content": "discard this answer too"},
            ]
        },
    )


def _session(*, fail_after: int | None = None, thinking: bool = True) -> Any:
    @contextmanager
    def open_session(
        _snapshot: Path,
        tokenizer: TraceTokenizer,
        _device: str,
    ) -> Iterator[tuple[Any, frozenset[int]]]:
        calls = 0

        def generate(prompt: tuple[int, ...], _maximum: int) -> tuple[int, ...]:
            nonlocal calls
            calls += 1
            if fail_after is not None and calls > fail_after:
                raise RuntimeError("simulated interruption")
            response = (
                "<think>\na coherent teacher trace\n</think>\n\n"
                if thinking
                else ""
            )
            suffix = tokenizer.encode(f"{response}teacher answer {calls}<|im_end|>")
            return (*prompt, *suffix)

        yield generate, frozenset({tokenizer.eos_token_id})

    return open_session


def test_teacher_trace_generation_resumes_and_commits_whole_teacher_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = TraceTokenizer()
    behavior = Qwen3ChatBehavior()
    monkeypatch.setattr(trace_module, "_open_generation_session", _session(fail_after=1))

    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare_teacher_traces(
            tmp_path / "snapshot",
            tmp_path / "run",
            _slice(),
            tokenizer,
            behavior,
            _records(),
            teacher_source="Qwen/Qwen3-0.6B",
            teacher_revision="teacher-revision",
            count=2,
            sequence_length=512,
            seed=7,
            device="cpu",
        )

    monkeypatch.setattr(trace_module, "_open_generation_session", _session())
    prepared = prepare_teacher_traces(
        tmp_path / "snapshot",
        tmp_path / "run",
        _slice(),
        tokenizer,
        behavior,
        _records(),
        teacher_source="Qwen/Qwen3-0.6B",
        teacher_revision="teacher-revision",
        count=2,
        sequence_length=512,
        seed=7,
        device="cpu",
    )

    assert len(prepared.messages) == 2
    assert [turn[-1]["reasoning_content"] for turn in prepared.messages] == [
        "a coherent teacher trace",
        "a coherent teacher trace",
    ]
    assert [turn[-1]["content"] for turn in prepared.messages] == [
        "teacher answer 1",
        "teacher answer 1",
    ]
    assert all("discard this" not in str(turn[-1]) for turn in prepared.messages)
    artifact_root = (
        tmp_path / "run" / "artifacts" / prepared.reference.artifact_id[7:9]
        / prepared.reference.artifact_id
    )
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["identity_payload"]["teacher"]["revision"] == "teacher-revision"
    assert manifest["record_count"] == 2
    journal = next((tmp_path / "run" / "state" / "teacher-traces").glob("*.jsonl"))
    attempts = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()[1:]
    ]
    assert [record["status"] for record in attempts] == ["accepted", "accepted"]

    def unexpected_session(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("completed trace artifact should be reused without loading the teacher")

    monkeypatch.setattr(trace_module, "_open_generation_session", unexpected_session)
    reused = prepare_teacher_traces(
        tmp_path / "snapshot",
        tmp_path / "run",
        _slice(),
        tokenizer,
        behavior,
        _records(),
        teacher_source="Qwen/Qwen3-0.6B",
        teacher_revision="teacher-revision",
        count=2,
        sequence_length=512,
        seed=7,
        device="cpu",
    )
    assert reused == prepared


def test_non_thinking_generation_uses_the_complete_teacher_answer_without_reasoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = TraceTokenizer()
    behavior = Qwen3ChatBehavior()
    monkeypatch.setattr(
        trace_module,
        "_open_generation_session",
        _session(thinking=False),
    )
    source_records = iter(_records())

    prepared = prepare_teacher_traces(
        tmp_path / "snapshot",
        tmp_path / "run",
        _slice(ReasoningMode.NON_THINKING),
        tokenizer,
        behavior,
        source_records,
        teacher_source="Qwen/Qwen3-0.6B",
        teacher_revision="teacher-revision",
        count=1,
        sequence_length=512,
        seed=7,
        device="cpu",
    )

    response = prepared.messages[0][-1]
    assert response == {"role": "assistant", "content": "teacher answer 1"}
    assert "reasoning_content" not in response
    rendered = behavior.render_completed(
        tokenizer,
        list(prepared.messages[0]),
        ReasoningMode.NON_THINKING,
        assistant_target_weight=1.0,
        prompt_target_weight=0.0,
    )
    rendered_text = tokenizer.decode(rendered.input_ids)
    assert "<think>\n\n</think>" in rendered_text
    assert "discard this dataset answer" not in rendered_text

    extended = prepare_teacher_traces(
        tmp_path / "snapshot",
        tmp_path / "run",
        _slice(ReasoningMode.NON_THINKING),
        tokenizer,
        behavior,
        source_records,
        teacher_source="Qwen/Qwen3-0.6B",
        teacher_revision="teacher-revision",
        count=2,
        sequence_length=512,
        seed=7,
        device="cpu",
    )
    assert extended.identity == prepared.identity
    assert len(extended.messages) == 2
    assert extended.reference != prepared.reference


def test_llamacpp_teacher_implementation_generates_prompts_concurrently_in_source_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = TraceTokenizer()
    behavior = Qwen3ChatBehavior()
    barrier = threading.Barrier(2)
    worker_names: list[str] = []

    @contextmanager
    def open_backend(
        _snapshot: Path,
        _tokenizer: TraceTokenizer,
        _device: str,
        *,
        implementation: str,
        sequence_length: int,
        progress: object,
    ) -> Iterator[trace_module._GenerationBackend]:
        assert implementation == LLAMACPP_TEACHER_TRACE_IMPLEMENTATION
        assert sequence_length == 512
        assert progress is None

        def generate(prompt: tuple[int, ...], _maximum: int) -> tuple[int, ...]:
            worker_names.append(threading.current_thread().name)
            barrier.wait(timeout=5)
            rendered = tokenizer.decode(prompt)
            label = "first" if "First prompt" in rendered else "second"
            suffix = tokenizer.encode(
                f"<think>\nreasoning {label}\n</think>\n\nanswer {label}<|im_end|>"
            )
            return (*prompt, *suffix)

        yield trace_module._GenerationBackend(generate, frozenset({1}), parallelism=2)

    monkeypatch.setattr(trace_module, "_open_teacher_generation_backend", open_backend)
    item = replace(
        _slice(),
        teacher_trace_generation=replace(
            _slice().teacher_trace_generation,
            implementation=LLAMACPP_TEACHER_TRACE_IMPLEMENTATION,
        ),
    )

    prepared = prepare_teacher_traces(
        tmp_path / "snapshot",
        tmp_path / "run",
        item,
        tokenizer,
        behavior,
        _records(),
        teacher_source="Qwen/Qwen3-0.6B",
        teacher_revision="teacher-revision",
        count=2,
        sequence_length=512,
        seed=7,
        device="cuda",
    )

    assert [turn[-1]["content"] for turn in prepared.messages] == [
        "answer first",
        "answer second",
    ]
    assert len(set(worker_names)) == 2
    journal = next((tmp_path / "run" / "state" / "teacher-traces").glob("*.jsonl"))
    attempts = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()[1:]
    ]
    assert [record["attempt"] for record in attempts] == [1, 2]
