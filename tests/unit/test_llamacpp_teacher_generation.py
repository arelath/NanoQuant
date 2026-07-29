from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


def test_llamacpp_teacher_session_marks_context_truncation_as_a_rejected_record(
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

    with pytest.raises(ValueError, match="context limit"):
        session.generate((10, 20), 128)


def test_llamacpp_server_context_includes_per_slot_decode_headroom() -> None:
    assert teacher._server_context_size(2048, 4) == 8200


def test_adaptive_parallelism_uses_one_slot_for_qwen8_q8_on_12_gib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(teacher.LLAMACPP_TEACHER_PARALLELISM_ENV, raising=False)

    selection = teacher._select_parallelism(
        10_824_038_208,
        "cuda",
        2048,
        gpu_memory=(12_282 * 2**20, 12_282 * 2**20),
    )

    assert selection.parallelism == 1
    assert selection.reason == "adaptive_vram"


@pytest.mark.parametrize(
    ("headroom_gib", "expected"),
    ((1, 1), (2, 2), (4, 4)),
)
def test_adaptive_parallelism_scales_with_available_model_headroom(
    headroom_gib: int,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(teacher.LLAMACPP_TEACHER_PARALLELISM_ENV, raising=False)
    model_bytes = 5 * 2**30

    selection = teacher._select_parallelism(
        model_bytes,
        "cuda:0",
        2048,
        gpu_memory=(model_bytes + headroom_gib * 2**30, 12 * 2**30),
    )

    assert selection.parallelism == expected


def test_parallelism_override_is_explicit_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(teacher.LLAMACPP_TEACHER_PARALLELISM_ENV, "2")
    assert teacher._select_parallelism(1, "cuda", 2048).parallelism == 2

    monkeypatch.setenv(teacher.LLAMACPP_TEACHER_PARALLELISM_ENV, "3")
    with pytest.raises(ValueError, match="must be 1, 2, or 4"):
        teacher._select_parallelism(1, "cuda", 2048)


def test_selected_parallelism_controls_server_slots_and_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    gguf = snapshot / "teacher.gguf"
    gguf.write_bytes(b"gguf")
    observed: dict[str, object] = {}

    class Process:
        process = SimpleNamespace()

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait() -> SimpleNamespace:
            return SimpleNamespace()

    class Interop:
        @staticmethod
        def runtime_environment(_server: Path) -> dict[str, str]:
            return {}

        @staticmethod
        def request(command: tuple[object, ...], *, environment: object) -> tuple[object, ...]:
            observed["command"] = command
            observed["environment"] = environment
            return command

        @staticmethod
        def start_streaming(
            _request: object,
            *,
            on_stdout: object,
            on_stderr: object,
        ) -> Process:
            observed["on_stdout"] = on_stdout
            observed["on_stderr"] = on_stderr
            return Process()

    monkeypatch.setattr(teacher, "_llama_cpp_root", lambda: tmp_path)
    monkeypatch.setattr(teacher, "_server_executable", lambda _root: tmp_path / "server")
    monkeypatch.setattr(teacher, "_resolve_teacher_gguf", lambda *_args: gguf)
    monkeypatch.setattr(
        teacher,
        "_select_parallelism",
        lambda *_args: teacher._ParallelismSelection(2, "adaptive_vram", 4, 8, 16),
    )
    monkeypatch.setattr(teacher, "_available_port", lambda: 8123)
    monkeypatch.setattr(teacher, "LlamaCppInterop", lambda _root: Interop())
    monkeypatch.setattr(teacher, "_wait_until_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(teacher, "_validate_tokenizer", lambda *_args: None)
    monkeypatch.setattr(teacher, "_eos_token_ids", lambda *_args: frozenset({2}))
    events: list[tuple[str, object]] = []

    with teacher.open_llamacpp_teacher_session(
        snapshot,
        SimpleNamespace(),
        device="cpu",
        sequence_length=2048,
        gguf_path=gguf,
        progress=lambda event, fields: events.append((event, fields)),
    ) as session:
        assert session.parallelism == 2

    command = tuple(str(value) for value in observed["command"])  # type: ignore[arg-type]
    assert command[command.index("--parallel") + 1] == "2"
    assert command[command.index("--ctx-size") + 1] == "4100"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert command[command.index("--cache-type-k") + 1] == "f16"
    assert command[command.index("--cache-type-v") + 1] == "f16"
    assert command[command.index("--fit-target") + 1] == "768"
    selected = next(fields for event, fields in events if event.endswith("parallelism_selected"))
    assert isinstance(selected, dict)
    assert selected["parallelism"] == 2
    assert selected["reason"] == "adaptive_vram"


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


def test_huggingface_snapshot_gguf_symlink_preserves_logical_filename(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "models--unsloth--Qwen3-8B-GGUF"
    snapshot = cache / "snapshots" / "revision"
    blobs = cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "content-addressed-name-without-extension"
    blob.write_bytes(b"prebuilt")
    gguf = snapshot / "Qwen3-8B-UD-Q8_K_XL.gguf"
    gguf.symlink_to(blob)

    selected = teacher._resolve_teacher_gguf(
        snapshot,
        tmp_path / "llama.cpp",
        gguf,
        None,
    )

    assert selected == gguf.absolute()
    assert selected.suffix == ".gguf"
    assert selected.resolve() == blob.resolve()
