from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import nanoquant.infrastructure.llamacpp_teacher_generation as teacher
from nanoquant.infrastructure.llamacpp_teacher_generation import (
    LlamaCppTeacherSession,
)


def test_llamacpp_teacher_session_sends_exact_greedy_token_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def post(endpoint: str, payload: object, *, timeout: float) -> dict[str, Any]:
        observed.update({"endpoint": endpoint, "payload": payload, "timeout": timeout})
        return {
            "tokens": [30, 2],
            "stop_type": "eos",
            "truncated": False,
            "timings": {"prompt_n": 3, "cache_n": 0},
        }

    monkeypatch.setattr(teacher, "_post_json", post)
    session = LlamaCppTeacherSession(
        "http://127.0.0.1:8123",
        frozenset({2}),
        parallelism=4,
    )

    complete = session.generate((10, 20, 21), 128)

    assert complete == (10, 20, 21, 30, 2)
    assert observed["endpoint"] == "http://127.0.0.1:8123/completion"
    payload = observed["payload"]
    assert isinstance(payload, dict)
    assert payload["prompt"] == [10, 20, 21]
    assert payload["n_predict"] == 128
    assert payload["temperature"] == 0.0
    assert payload["top_k"] == 1
    assert payload["cache_prompt"] is False
    assert payload["return_tokens"] is True


def test_llamacpp_teacher_session_rejects_context_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        teacher,
        "_post_json",
        lambda *_args, **_kwargs: {
            "tokens": [30],
            "truncated": True,
        },
    )
    session = LlamaCppTeacherSession(
        "http://127.0.0.1:8123",
        frozenset({2}),
        parallelism=4,
    )

    with pytest.raises(RuntimeError, match="truncated"):
        session.generate((10, 20), 128)


def test_llamacpp_teacher_session_rejects_changed_prompt_token_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        teacher,
        "_post_json",
        lambda *_args, **_kwargs: {
            "tokens": [30, 2],
            "truncated": False,
            "timings": {"prompt_n": 1, "cache_n": 0},
        },
    )
    session = LlamaCppTeacherSession(
        "http://127.0.0.1:8123",
        frozenset({2}),
        parallelism=4,
    )

    with pytest.raises(RuntimeError, match="prompt-token count"):
        session.generate((10, 20), 128)


def test_prebuilt_teacher_gguf_is_reused_without_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    gguf = snapshot / "Qwen3-8B-UD-Q8_K_XL.gguf"
    gguf.write_bytes(b"prebuilt")
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        teacher,
        "_prepare_bfloat16_gguf",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("prebuilt Unsloth GGUF must not be converted")
        ),
    )

    selected = teacher._resolve_teacher_gguf(
        snapshot,
        tmp_path / "llama.cpp",
        gguf,
        lambda event, fields: events.append((event, fields)),
    )

    assert selected == gguf.resolve()
    assert events[0][0] == "teacher_llamacpp_prebuilt_reused"
