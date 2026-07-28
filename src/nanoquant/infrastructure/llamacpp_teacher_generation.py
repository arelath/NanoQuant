"""Parallel llama.cpp server generation for pinned teacher checkpoints."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from nanoquant.config.codec import semantic_hash
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.subprocess_interop import (
    LlamaCppInterop,
    StreamingSubprocess,
)

LlamaCppTeacherProgress = Callable[[str, Mapping[str, object]], None]
LLAMACPP_TEACHER_PARALLELISM = 4


@dataclass(frozen=True, slots=True)
class LlamaCppTeacherSession:
    """Exact-token completion client for one local llama.cpp server."""

    endpoint: str
    eos_token_ids: frozenset[int]
    parallelism: int
    timeout_seconds: float = 1800.0

    def generate(self, prompt: tuple[int, ...], maximum_new_tokens: int) -> tuple[int, ...]:
        payload = {
            "prompt": list(prompt),
            "n_predict": maximum_new_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "xtc_probability": 0.0,
            "seed": 0,
            "cache_prompt": False,
            "return_tokens": True,
            "stream": False,
        }
        response = _post_json(
            f"{self.endpoint}/completion",
            payload,
            timeout=self.timeout_seconds,
        )
        if bool(response.get("truncated")):
            raise RuntimeError("llama.cpp teacher truncated the prompt or response context")
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or any(not isinstance(value, int) for value in tokens):
            raise RuntimeError("llama.cpp teacher returned invalid token IDs")
        timings = response.get("timings")
        if isinstance(timings, dict):
            prompt_count = timings.get("prompt_n")
            cached_count = timings.get("cache_n", 0)
            if (
                isinstance(prompt_count, int)
                and isinstance(cached_count, int)
                and prompt_count + cached_count != len(prompt)
            ):
                raise RuntimeError("llama.cpp teacher processed a different prompt-token count")
        return (*prompt, *(int(value) for value in tokens))


def _post_json(endpoint: str, payload: object, *, timeout: float) -> dict[str, Any]:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"llama.cpp teacher request failed: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("llama.cpp teacher returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("llama.cpp teacher returned a non-object response")
    return cast(dict[str, Any], value)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _llama_cpp_root() -> Path:
    configured = os.environ.get("NANOQUANT_LLAMA_CPP_ROOT")
    root = (
        Path(configured)
        if configured
        else _repository_root().parent / "llama.cpp"
    ).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            "llama.cpp source tree is missing; set NANOQUANT_LLAMA_CPP_ROOT"
        )
    return root


def _server_executable(root: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    candidates = (
        root / "build" / "bin" / "Release" / f"llama-server{suffix}",
        root / "build" / "bin" / f"llama-server{suffix}",
        root / f"llama-server{suffix}",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"llama.cpp server executable is missing; searched: {rendered}")


def _converter(root: Path) -> Path:
    converter = root / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"llama.cpp Hugging Face converter is missing: {converter}")
    return converter.resolve()


def _snapshot_signature(snapshot: Path) -> dict[str, object]:
    model_files = tuple(sorted(snapshot.glob("*.safetensors")))
    if not model_files:
        raise FileNotFoundError(f"teacher snapshot has no safetensors checkpoint: {snapshot}")
    return {
        "snapshot": str(snapshot.resolve()),
        "model_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "modified_ns": path.stat().st_mtime_ns,
            }
            for path in model_files
        ],
    }


def _prepare_bfloat16_gguf(
    snapshot: Path,
    root: Path,
    progress: LlamaCppTeacherProgress | None,
) -> Path:
    converter = _converter(root)
    signature = {
        "schema_version": 1,
        "snapshot": _snapshot_signature(snapshot),
        "converter_sha256": hash_file(converter),
        "outtype": "bf16",
    }
    cache_key = semantic_hash(signature)[7:23]
    cache_root = _repository_root() / ".nanoquant" / "teacher-models" / cache_key
    output = cache_root / "teacher-bf16.gguf"
    receipt_path = cache_root / "teacher-bf16.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        receipt = None
    if (
        isinstance(receipt, dict)
        and receipt.get("signature") == signature
        and output.is_file()
        and output.stat().st_size == int(receipt.get("bytes", -1))
    ):
        if progress is not None:
            progress(
                "teacher_llamacpp_conversion_reused",
                {"gguf": str(output), "bytes": output.stat().st_size},
            )
        return output

    cache_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".teacher-bf16-",
        suffix=".gguf",
        dir=cache_root,
    )
    os.close(descriptor)
    temporary = Path(temporary_name).resolve()
    started = time.perf_counter()
    if progress is not None:
        progress(
            "teacher_llamacpp_conversion_started",
            {"snapshot": str(snapshot), "gguf": str(output)},
        )
    interop = LlamaCppInterop(root)
    try:
        result = interop.run(
            interop.request(
                (
                    Path(sys.executable),
                    converter,
                    snapshot,
                    "--outfile",
                    temporary,
                    "--outtype",
                    "bf16",
                ),
                environment=interop.converter_environment(converter),
            )
        )
        result.require_success("llama.cpp teacher checkpoint conversion")
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("llama.cpp teacher conversion produced no GGUF")
        temporary.replace(output)
        atomic_write_json(
            receipt_path,
            {
                "schema_version": 1,
                "signature": signature,
                "gguf": str(output),
                "bytes": output.stat().st_size,
            },
        )
    finally:
        if temporary.is_file():
            temporary.unlink()
    if progress is not None:
        progress(
            "teacher_llamacpp_conversion_completed",
            {
                "gguf": str(output),
                "bytes": output.stat().st_size,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    return output


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until_ready(
    endpoint: str,
    process: StreamingSubprocess,
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            result = process.wait()
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"llama.cpp teacher server exited during startup ({returncode}): {detail}"
            )
        try:
            with urllib.request.urlopen(f"{endpoint}/health", timeout=2.0) as response:
                payload = json.loads(response.read())
            if isinstance(payload, dict) and payload.get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"llama.cpp teacher server did not become ready: {last_error}")


def _validate_tokenizer(endpoint: str, tokenizer: Any) -> None:
    probes = ("NanoQuant tokenizer probe", "<think></think><|im_end|>")
    for text in probes:
        expected = tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))
        response = _post_json(
            f"{endpoint}/tokenize",
            {"content": text, "add_special": False, "parse_special": True},
            timeout=30.0,
        )
        values = response.get("tokens")
        if not isinstance(values, list):
            raise RuntimeError("llama.cpp tokenizer probe returned no token IDs")
        actual = tuple(int(value) for value in values)
        if actual != expected:
            raise ValueError("llama.cpp and Hugging Face teacher tokenizers differ")


def _eos_token_ids(snapshot: Path, tokenizer: Any) -> frozenset[int]:
    values: set[int] = set()
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos, int):
        values.add(tokenizer_eos)
    generation_path = snapshot / "generation_config.json"
    if generation_path.is_file():
        try:
            configured = json.loads(generation_path.read_text(encoding="utf-8")).get(
                "eos_token_id"
            )
        except json.JSONDecodeError as exc:
            raise ValueError("teacher generation configuration is invalid JSON") from exc
        if isinstance(configured, int):
            values.add(configured)
        elif isinstance(configured, list):
            values.update(int(value) for value in configured)
    if not values:
        raise ValueError("llama.cpp teacher requires at least one EOS token ID")
    return frozenset(values)


def _stop_server(process: StreamingSubprocess) -> None:
    if process.poll() is None:
        process.process.terminate()
    try:
        process.wait()
    except BaseException:
        if process.poll() is None:
            process.process.kill()
        process.wait()


def _resolve_teacher_gguf(
    snapshot: Path,
    root: Path,
    gguf_path: str | Path | None,
    progress: LlamaCppTeacherProgress | None,
) -> Path:
    if gguf_path is None:
        return _prepare_bfloat16_gguf(snapshot, root, progress)
    gguf = Path(gguf_path).resolve(strict=True)
    if gguf.suffix.lower() != ".gguf":
        raise ValueError(f"prebuilt teacher model is not a GGUF file: {gguf}")
    try:
        gguf.relative_to(snapshot)
    except ValueError as exc:
        raise ValueError(
            f"prebuilt teacher GGUF is outside its pinned snapshot: {gguf}"
        ) from exc
    if progress is not None:
        progress(
            "teacher_llamacpp_prebuilt_reused",
            {"gguf": str(gguf), "bytes": gguf.stat().st_size},
        )
    return gguf


@contextmanager
def open_llamacpp_teacher_session(
    snapshot: str | Path,
    tokenizer: Any,
    *,
    device: str,
    sequence_length: int,
    gguf_path: str | Path | None = None,
    progress: LlamaCppTeacherProgress | None = None,
) -> Iterator[LlamaCppTeacherSession]:
    """Convert once, run one parallel local server, and yield an exact-token client."""

    snapshot_path = Path(snapshot).resolve()
    root = _llama_cpp_root()
    gguf = _resolve_teacher_gguf(snapshot_path, root, gguf_path, progress)
    server = _server_executable(root)
    parallelism = LLAMACPP_TEACHER_PARALLELISM
    port = _available_port()
    endpoint = f"http://127.0.0.1:{port}"
    interop = LlamaCppInterop(root)
    command = (
        server,
        "--model",
        gguf,
        "--ctx-size",
        str(sequence_length * parallelism),
        "--parallel",
        str(parallelism),
        "--cont-batching",
        "--no-context-shift",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--offline",
        "--no-webui",
        "--log-verbosity",
        "2",
        "--gpu-layers",
        "0" if device == "cpu" else "auto",
    )
    lease = nullcontext() if device == "cpu" else acquire_device_lease(device)
    started = time.perf_counter()
    with lease:
        process = interop.start_streaming(
            interop.request(
                command,
                environment=interop.runtime_environment(server),
            )
        )
        try:
            _wait_until_ready(endpoint, process, timeout_seconds=300.0)
            _validate_tokenizer(endpoint, tokenizer)
            eos_ids = _eos_token_ids(snapshot_path, tokenizer)
            if progress is not None:
                progress(
                    "teacher_llamacpp_server_ready",
                    {
                        "gguf": str(gguf),
                        "parallelism": parallelism,
                        "endpoint": endpoint,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
            yield LlamaCppTeacherSession(endpoint, eos_ids, parallelism)
        finally:
            _stop_server(process)


__all__ = [
    "LlamaCppTeacherSession",
    "open_llamacpp_teacher_session",
]
