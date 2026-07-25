"""Build the protocol-matched GGUF quality runner against a llama.cpp checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _default_llama_cpp_root() -> Path:
    return Path(os.environ.get("NANOQUANT_LLAMA_CPP_ROOT", r"D:\dev\research\llama.cpp"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cpp-root", type=Path, default=_default_llama_cpp_root())
    parser.add_argument("--config", default="Release")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    root = args.llama_cpp_root.resolve()
    repository = Path(__file__).resolve().parent.parent
    source = repository / "tools" / "llamacpp" / "quality_runner"
    build = root / "build" / "nanoquant-quality"
    if not (root / "include" / "llama.h").is_file():
        raise FileNotFoundError(f"llama.cpp headers are missing: {root}")
    subprocess.run(
        (
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DLLAMA_CPP_ROOT={root}",
            f"-DCMAKE_BUILD_TYPE={args.config}",
        ),
        check=True,
    )
    subprocess.run(
        (
            "cmake",
            "--build",
            str(build),
            "--config",
            args.config,
            f"-j{args.jobs}",
        ),
        check=True,
    )
    suffix = ".exe" if os.name == "nt" else ""
    candidates = (
        build / args.config / f"nanoquant-llamacpp-quality{suffix}",
        build / f"nanoquant-llamacpp-quality{suffix}",
    )
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        raise FileNotFoundError(f"quality runner build completed without an executable under {build}")
    print(executable.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
