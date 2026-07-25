"""Typed, cross-platform subprocess execution and streaming."""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass(frozen=True, slots=True)
class SubprocessRequest:
    command: tuple[str, ...]
    cwd: Path | None = None
    environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.command or any(not item for item in self.command):
            raise ValueError("subprocess command arguments must be non-empty")


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def require_success(self, description: str) -> SubprocessResult:
        if self.returncode:
            detail = self.stderr.strip() or self.stdout.strip()
            suffix = "" if not detail else f": {detail}"
            raise RuntimeError(f"{description} failed with exit code {self.returncode}{suffix}")
        return self


class SubprocessInterop:
    """Execute child processes without mutating the parent environment."""

    def run(
        self,
        request: SubprocessRequest,
        *,
        stdout: IO[str] | int | None = subprocess.PIPE,
        stderr: IO[str] | int | None = subprocess.PIPE,
    ) -> SubprocessResult:
        completed = subprocess.run(
            request.command,
            cwd=request.cwd,
            env=None if request.environment is None else dict(request.environment),
            stdout=stdout,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return SubprocessResult(
            request.command,
            completed.returncode,
            completed.stdout if isinstance(completed.stdout, str) else "",
            completed.stderr if isinstance(completed.stderr, str) else "",
        )

    def start_streaming(
        self,
        request: SubprocessRequest,
        *,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> StreamingSubprocess:
        return StreamingSubprocess(request, on_stdout=on_stdout, on_stderr=on_stderr)


class StreamingSubprocess:
    """A typed child-process handle whose output drains cannot deadlock."""

    def __init__(
        self,
        request: SubprocessRequest,
        *,
        on_stdout: Callable[[str], None] | None,
        on_stderr: Callable[[str], None] | None,
    ) -> None:
        self.request = request
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []
        self.process = subprocess.Popen(
            request.command,
            cwd=request.cwd,
            env=None if request.environment is None else dict(request.environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._threads = (
            self._drain(self.process.stdout, self._stdout_lines, on_stdout, "stdout"),
            self._drain(self.process.stderr, self._stderr_lines, on_stderr, "stderr"),
        )

    def _drain(
        self,
        stream: IO[str] | None,
        destination: list[str],
        callback: Callable[[str], None] | None,
        suffix: str,
    ) -> threading.Thread:
        if stream is None:
            raise RuntimeError(f"subprocess {suffix} stream is unavailable")

        def consume() -> None:
            with stream:
                for raw in stream:
                    line = raw.rstrip("\r\n")
                    destination.append(line)
                    if callback is not None:
                        callback(line)

        thread = threading.Thread(
            target=consume,
            name=f"nanoquant-subprocess-{suffix}",
            daemon=True,
        )
        thread.start()
        return thread

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self) -> SubprocessResult:
        returncode = self.process.wait()
        for thread in self._threads:
            thread.join()
        return SubprocessResult(
            self.request.command,
            returncode,
            "\n".join(self._stdout_lines),
            "\n".join(self._stderr_lines),
        )


class LlamaCppInterop(SubprocessInterop):
    """Construct isolated llama.cpp runtime and Python converter environments."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def runtime_environment(self, runner: str | Path) -> dict[str, str]:
        environment = os.environ.copy()
        binary_directories = (
            Path(runner).resolve().parent,
            self.root / "build" / "bin",
            self.root / "build" / "bin" / "Release",
        )
        environment["PATH"] = os.pathsep.join(
            (*(str(path) for path in binary_directories), environment.get("PATH", ""))
        )
        if os.name != "nt":
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                (*(str(path) for path in binary_directories), environment.get("LD_LIBRARY_PATH", ""))
            )
        return environment

    def converter_environment(self, converter: str | Path) -> dict[str, str] | None:
        if Path(converter).resolve().parent == self.root:
            return None
        environment = os.environ.copy()
        search = (self.root, self.root / "gguf-py")
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            (*(str(path) for path in search), *((existing,) if existing else ()))
        )
        environment["NO_LOCAL_GGUF"] = "1"
        return environment

    def request(
        self,
        command: Sequence[str | Path],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> SubprocessRequest:
        return SubprocessRequest(tuple(str(item) for item in command), environment=environment)


__all__ = [
    "LlamaCppInterop",
    "StreamingSubprocess",
    "SubprocessInterop",
    "SubprocessRequest",
    "SubprocessResult",
]
