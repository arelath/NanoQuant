"""Capture immutable Milestone-0 evidence from the two local reference trees.

The script deliberately uses an allowlist and never reads either repository's
dotenv files. Re-running writes a new capture directory keyed by UTC time; it
does not overwrite prior evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def command(arguments: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": arguments, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_capture(repository: Path, output: Path) -> dict[str, Any]:
    revision = command(["git", "rev-parse", "HEAD"], repository)
    status = command(["git", "status", "--short"], repository)
    patch = command(["git", "diff", "--binary", "--no-ext-diff"], repository)
    patch_path = output / "dirty.patch"
    patch_path.write_text(patch.get("stdout", ""), encoding="utf-8")
    return {
        "path_at_capture": str(repository),
        "revision": revision.get("stdout", "").strip(),
        "status": status.get("stdout", "").splitlines(),
        "dirty_patch": patch_path.name,
        "dirty_patch_sha256": sha256(patch_path),
    }


def copy_with_hash(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source_name": source.name,
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evidence/m0"))
    parser.add_argument("--capture-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    capture = args.output / args.capture_id
    if capture.exists():
        raise FileExistsError(f"capture already exists: {capture}")
    (capture / "legacy").mkdir(parents=True)
    (capture / "llama.cpp").mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "capture_id": args.capture_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "capture_tool": "tools/capture_m0_baseline.py",
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "nvidia_smi": command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
        },
        "legacy": git_capture(args.legacy, capture / "legacy"),
        "llama_cpp": git_capture(args.llama, capture / "llama.cpp"),
    }
    legacy_python = args.legacy / ".venv" / "Scripts" / "python.exe"
    manifest["legacy"]["packages"] = command([str(legacy_python), "-m", "pip", "freeze"])
    golden = []
    for name in ("019-phase1-weight-errors.md", "019-phase1-weight-errors.csv", "019-phase1-rank-utility.csv"):
        golden.append(copy_with_hash(args.legacy / "outputs" / name, capture / "golden" / name))
    manifest["golden"] = golden
    launcher = args.legacy / "019-compress-gemma-3-1b-it-phase1.py"
    manifest["experiment_019_launcher"] = copy_with_hash(launcher, capture / "legacy" / launcher.name)
    cuda = args.llama / "ggml" / "src" / "ggml-cuda" / "nanoquant.cu"
    converter = args.llama / "convert_nanoquant_to_gguf.py"
    manifest["llama_cpp"]["nanoquant_cuda"] = copy_with_hash(cuda, capture / "llama.cpp" / cuda.name)
    manifest["llama_cpp"]["converter"] = copy_with_hash(converter, capture / "llama.cpp" / converter.name)
    cache_candidates = sorted(args.llama.glob("build*/CMakeCache.txt"))
    manifest["llama_cpp"]["build_configurations"] = [
        copy_with_hash(path, capture / "llama.cpp" / "build" / f"{path.parent.name}-CMakeCache.txt")
        for path in cache_candidates
    ]
    benchmark_names = (
        "bench-nanoquant-current-pg512-tg128-r5.json",
        "bench-nanoquant-current-pp512-r5.json",
        "bench-nanoquant-current-tg128-r10.json",
    )
    manifest["llama_cpp"]["benchmark_json"] = [
        copy_with_hash(args.llama / name, capture / "llama.cpp" / "benchmarks" / name)
        for name in benchmark_names
        if (args.llama / name).exists()
    ]
    manifest_path = capture / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    (args.output / "LATEST").write_text(args.capture_id + "\n", encoding="ascii")
    print(capture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
