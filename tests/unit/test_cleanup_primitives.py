from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from nanoquant.application.telemetry import TelemetryContext
from nanoquant.config.codec import canonical_json, semantic_hash
from nanoquant.domain.errors import ErrorCode
from nanoquant.domain.linear_math import chunk_slices, chunked_reduce, parse_torch_dtype
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError
from nanoquant.infrastructure.io_utils import atomic_workspace
from nanoquant.infrastructure.memory_cleanup import explicit_memory_cleanup, gpu_memory_scope
from nanoquant.infrastructure.safetensors_io import SAFETENSORS, load_tensors
from nanoquant.infrastructure.subprocess_interop import SubprocessInterop, SubprocessRequest
from nanoquant.runtime.codec import RuntimeDecodeError, decode_dataclass
from nanoquant.runtime.io_utils import atomic_output_directory


def test_semantic_hash_matches_the_existing_canonical_identity() -> None:
    payload = {"b": (2, 3), "a": 1}
    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    assert semantic_hash(payload) == f"sha256:{expected}"


def test_chunked_reduction_covers_the_leading_dimension() -> None:
    value = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    assert tuple(chunk_slices(5, 2)) == (slice(0, 2), slice(2, 4), slice(4, 5))
    assert chunked_reduce(value, 2, lambda chunk: chunk.square().sum()) == value.square().sum()
    with pytest.raises(ValueError, match="positive"):
        tuple(chunk_slices(1, 0))


def test_dtype_parser_has_one_consistent_failure_mode() -> None:
    assert parse_torch_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="unsupported torch dtype"):
        parse_torch_dtype("float128")


def test_explicit_memory_cleanup_runs_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("nanoquant.infrastructure.memory_cleanup.gc.collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    with pytest.raises(RuntimeError, match="fixture"), explicit_memory_cleanup("cuda:0"):
        raise RuntimeError("fixture")
    assert calls == ["gc", "cuda"]


def test_safetensors_loader_validates_keys_and_places_tensors(tmp_path: Path) -> None:
    path = tmp_path / "fixture.safetensors"
    SAFETENSORS.save({"left": torch.arange(3), "right": torch.arange(2)}, path)
    loaded = load_tensors(path, ("right", "left"))
    assert tuple(loaded) == ("right", "left")
    assert torch.equal(loaded["left"], torch.arange(3))
    with pytest.raises(KeyError, match="missing"):
        load_tensors(path, ("absent",))


def test_atomic_workspaces_publish_success_and_clean_failure(tmp_path: Path) -> None:
    destination = tmp_path / "research-artifact"
    with atomic_workspace(destination) as staging:
        (staging / "value.txt").write_text("complete", encoding="utf-8")
    assert (destination / "value.txt").read_text(encoding="utf-8") == "complete"

    runtime_destination = tmp_path / "runtime-artifact"
    with atomic_output_directory(runtime_destination) as staging:
        (staging / "value.txt").write_text("complete", encoding="utf-8")
    assert runtime_destination.is_dir()

    failed = tmp_path / "failed-artifact"
    with pytest.raises(RuntimeError, match="fixture"):
        with atomic_workspace(failed) as staging:
            (staging / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("fixture")
    assert not failed.exists()
    assert not tuple(tmp_path.glob(".failed-artifact-*"))


def test_gpu_memory_scope_synchronizes_and_cleans_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("nanoquant.infrastructure.memory_cleanup.gc.collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: calls.append("sync"))
    with pytest.raises(RuntimeError, match="fixture"), gpu_memory_scope("cuda:0"):
        raise RuntimeError("fixture")
    assert calls == ["gc", "empty", "sync"]


def test_subprocess_interop_returns_typed_output_and_streams() -> None:
    interop = SubprocessInterop()
    request = SubprocessRequest(
        (
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        )
    )
    result = interop.run(request).require_success("fixture")
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"

    stderr: list[str] = []
    child = interop.start_streaming(request, on_stderr=stderr.append)
    streamed = child.wait().require_success("fixture")
    assert streamed.stdout.strip() == "out"
    assert stderr == ["err"]


def test_typed_diagnostic_error_formats_code_once() -> None:
    error = ArtifactCorruptionError("ART001 corrupt fixture")
    assert error.code is ErrorCode.ARTIFACT_CORRUPTION
    assert str(error) == "ART001 corrupt fixture"


def test_telemetry_context_emits_complete_and_failed_lifecycles() -> None:
    class Events:
        def __init__(self) -> None:
            self.names: list[str] = []

        def emit(self, _stage: str, _severity: str, name: str, **_fields: Any) -> None:
            self.names.append(name)

    events = Events()
    with TelemetryContext(events, "fixture").operation("work"):  # type: ignore[arg-type]
        pass
    with pytest.raises(RuntimeError, match="fixture"):
        with TelemetryContext(events, "fixture").operation("broken"):  # type: ignore[arg-type]
            raise RuntimeError("fixture")
    assert events.names == [
        "work.started",
        "work.completed",
        "broken.started",
        "broken.failed",
    ]


def test_safetensors_and_llamacpp_process_calls_stay_behind_typed_boundaries() -> None:
    source_root = Path("src/nanoquant")
    raw_tensor_imports = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from safetensors" in text or "import safetensors" in text:
            raw_tensor_imports.append(path.as_posix())
    assert sorted(raw_tensor_imports) == [
        "src/nanoquant/infrastructure/safetensors_io.py",
        "src/nanoquant/runtime/safetensors_io.py",
    ]

    llama_boundaries = (
        source_root / "llamacpp_quality.py",
        source_root / "infrastructure/gguf_export.py",
        source_root / "infrastructure/mmproj_export.py",
    )
    assert all(
        "subprocess.run" not in path.read_text(encoding="utf-8")
        and "subprocess.Popen" not in path.read_text(encoding="utf-8")
        for path in llama_boundaries
    )


@dataclass(frozen=True)
class _Child:
    value: int


@dataclass(frozen=True)
class _Manifest:
    name: str
    children: tuple[_Child, ...]
    enabled: bool = True


def test_runtime_decoder_handles_nested_tuples_defaults_and_paths() -> None:
    decoded = decode_dataclass(_Manifest, {"name": "fixture", "children": [{"value": 3}]})
    assert decoded == _Manifest("fixture", (_Child(3),))
    with pytest.raises(RuntimeDecodeError, match=r"manifest\.children\[0\]\.value"):
        decode_dataclass(_Manifest, {"name": "fixture", "children": [{"value": "bad"}]})
