"""Build a local, hash-pinned llama.cpp imatrix dataset from calibration Parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import _paths  # noqa: F401

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.imatrix import load_imatrix
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.runs import RunLease

ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/gemma-3-270m-it"
MODEL_REVISION = "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
CALIBRATION_REPO = "eaddario/imatrix-calibration"
CALIBRATION_REVISION = "e87ed55dcba9d9c3a3e41539f3e728e981b1daa4"


def read_calibration_text(path: Path, parse_special: bool) -> bytes:
    data = path.read_bytes()
    text = data.decode("utf-8")
    if not text.strip() or text.startswith("\ufeff"):
        raise ValueError("local calibration must be nonempty UTF-8 without BOM")
    if any(token in text for token in ("<bos>", "<start_of_turn>", "<end_of_turn>")) and not parse_special:
        raise ValueError("Gemma-formatted text requires --parse-special")
    return data


def extract_calibration(parquet: Path, destination: Path, column: str) -> dict:
    """Preserve each nonempty string exactly; separate multiple records by two LF bytes."""
    import pyarrow.parquet as pq

    temporary = destination.with_suffix(".txt.partial")
    rows = 0
    skipped = 0
    try:
        with pq.ParquetFile(parquet) as source, temporary.open("wb") as output:
            if column not in source.schema_arrow.names:
                raise ValueError(f"missing calibration column {column!r}")
            for batch in source.iter_batches(batch_size=64, columns=[column]):
                for value in batch.column(0).to_pylist():
                    if value is None or value == "":
                        skipped += 1
                        continue
                    if not isinstance(value, str):
                        raise ValueError(f"calibration column {column!r} must contain strings")
                    if not value.strip():
                        skipped += 1
                        continue
                    if rows:
                        output.write(b"\n\n")
                    output.write(value.encode("utf-8"))
                    rows += 1
            if not rows:
                raise ValueError("calibration contains no nonempty text")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"filename": destination.name, "sha256": hash_file(destination),
            "bytes": destination.stat().st_size, "rows_written": rows, "rows_skipped": skipped,
            "encoding": "utf-8", "record_separator": "\n\n", "text_column": column}


def run_command(command: list[str], log: Path) -> None:
    print(f"Running: {subprocess.list2cmdline(command)}\nLog: {log}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        # The child remains attached to this supervisor; cancellation waits for termination
        # before releasing the device lease. No shell expansion is involved.
        with subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT) as child:
            try:
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
    if code:
        raise RuntimeError(f"command exited with {code}; see {log}")


def validate_completion(path: Path, log: Path, context: int, chunks: int) -> dict:
    """A periodic save or a zero exit code alone is not a completed dataset."""
    from gguf import GGUFReader

    data = load_imatrix(path)
    reader = GGUFReader(str(path), "r")
    count = reader.get_field("imatrix.chunk_count")
    size = reader.get_field("imatrix.chunk_size")
    if count is None or size is None:
        raise ValueError("imatrix is missing chunk metadata")
    completed = int(count.contents())
    chunk_size = int(size.contents())
    matches = re.findall(r"computing over (\d+) chunks, n_ctx=(\d+)", log.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError("cannot establish expected chunk count from llama-imatrix log")
    expected, logged_context = map(int, matches[0])
    if completed != expected or expected <= 0 or chunk_size != context or logged_context != context:
        raise ValueError(f"incomplete imatrix: {completed}/{expected} chunks; context {chunk_size}/{context}")
    if chunks != -1 and completed != chunks:
        raise ValueError(f"calibration exhausted before requested {chunks} chunks (got {completed})")
    return {"filename": path.name, "sha256": data.sha256, "bytes": path.stat().st_size,
            "format": data.format, "tensor_count": len(data.entries), "chunks": completed,
            "chunk_size": chunk_size, "context_tokens_processed": completed * chunk_size,
            "tensor_shapes": {key: list(value.shape) for key, value in sorted(data.entries.items())}}


def verify_files(root: Path, receipt: dict) -> None:
    for item in receipt["files"]:
        path = root / item["filename"]
        if not path.is_file() or hash_file(path) != item["sha256"]:
            raise ValueError(f"dataset artifact missing or changed: {path}")
    load_imatrix(root / receipt["filename"], receipt["sha256"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default=MODEL, help="Hugging Face model ID")
    result.add_argument("--revision", help="default: pinned Gemma revision, otherwise resolve main")
    result.add_argument("--dataset-dir", type=Path, help="default: Datasets/<model>-imatrix-eaddario-huge")
    result.add_argument("--llama-cpp-root", type=Path,
                        default=Path(os.environ.get("NANOQUANT_LLAMA_CPP_ROOT", r"D:\dev\research\llama.cpp")))
    result.add_argument("--imatrix-exe", type=Path, help="override llama-imatrix executable")
    result.add_argument("--model-gguf", type=Path, help="existing full-precision GGUF for this exact model")
    result.add_argument("--calibration-repo", default=CALIBRATION_REPO)
    result.add_argument("--calibration-revision", default=CALIBRATION_REVISION)
    result.add_argument("--calibration-file", default="combined_all_huge.parquet")
    result.add_argument("--calibration-text", type=Path, help="local UTF-8 corpus instead of a Hub Parquet")
    result.add_argument("--parse-special", action="store_true", help="parse Gemma/chat control tokens in the corpus")
    result.add_argument("--text-column", default="content")
    result.add_argument("--context-size", type=int, default=512)
    result.add_argument("--chunks", type=int, default=-1, help="-1: every complete context in the full corpus")
    result.add_argument("--gpu-layers", type=int, default=99, help="0 for CPU; default 99")
    result.add_argument("--prepare-only", action="store_true", help="download, extract and convert; no inference")
    result.add_argument("--local-files-only", action="store_true")
    return result


def build(args: argparse.Namespace) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    if args.context_size < 32 or args.chunks == 0 or args.chunks < -1 or args.gpu_layers < 0:
        raise ValueError("require context-size >= 32, chunks = -1 or positive, gpu-layers >= 0")
    root = (args.dataset_dir or ROOT / "Datasets" / f"{args.model.split('/')[-1]}-imatrix-eaddario-huge").resolve()
    llama = args.llama_cpp_root.resolve()
    candidates = [llama / "build/bin/Release/llama-imatrix.exe", llama / "build/bin/llama-imatrix",
                  llama / "build/bin/llama-imatrix.exe"]
    executable = args.imatrix_exe or next((p for p in candidates if p.is_file()), candidates[0])
    executable = executable.resolve()
    converter = llama / "convert_hf_to_gguf.py"
    if not executable.is_file() or (not args.model_gguf and not converter.is_file()):
        raise FileNotFoundError("build llama-imatrix and supply --llama-cpp-root (or --imatrix-exe)")
    requested = {"model": args.model, "revision": args.revision or (MODEL_REVISION if args.model == MODEL else "main"),
                 "calibration_repo": args.calibration_repo, "calibration_revision": args.calibration_revision,
                 "calibration_file": args.calibration_file, "text_column": args.text_column,
                 "context_size": args.context_size, "chunks": args.chunks, "gpu_layers": args.gpu_layers,
                 "model_gguf": str(args.model_gguf.resolve()) if args.model_gguf else None,
                 "executable_sha256": hash_file(executable)}
    if args.calibration_text:
        local_data = read_calibration_text(args.calibration_text, args.parse_special)
        requested["calibration_text"] = {"path": str(args.calibration_text.resolve()),
                                          "sha256": hashlib.sha256(local_data).hexdigest()}
    if args.parse_special:
        requested["parse_special"] = True
    root.mkdir(parents=True, exist_ok=True)
    with RunLease(root / ".active-lease.json"):
        receipt_path = root / "source.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["request"] != requested:
                raise ValueError("dataset settings differ; choose a new --dataset-dir")
            verify_files(root, receipt)
            print(f"Verified existing dataset: {receipt_path}", flush=True)
            return receipt_path
        request_path = root / "request.json"
        if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != requested:
            raise ValueError("prepared dataset settings differ; choose a new --dataset-dir")
        atomic_write_json(request_path, requested)
        if args.calibration_text:
            temporary = root / "calibration.txt.partial"
            temporary.write_bytes(local_data)
            temporary.replace(root / "calibration.txt")
            calibration = {"filename": "calibration.txt", "sha256": hash_file(root / "calibration.txt"),
                           "bytes": len(local_data), "encoding": "utf-8",
                           "source_path": str(args.calibration_text.resolve()),
                           "parse_special": args.parse_special}
        else:
            parquet = Path(hf_hub_download(args.calibration_repo, args.calibration_file, repo_type="dataset",
                                          revision=args.calibration_revision, local_files_only=args.local_files_only))
            calibration = extract_calibration(parquet, root / "calibration.txt", args.text_column)
            calibration.update({"repository": args.calibration_repo, "revision": parquet.parent.name,
                                "source_filename": args.calibration_file, "source_sha256": hash_file(parquet)})
        atomic_write_json(root / "calibration.json", calibration)
        snapshot = Path(snapshot_download(args.model, revision=requested["revision"],
                                         allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"],
                                         local_files_only=args.local_files_only))
        model_files = {str(p.relative_to(snapshot)): hash_file(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
        model = {"repository": args.model, "revision": snapshot.name, "files": model_files}
        gguf = args.model_gguf.resolve() if args.model_gguf else root / "model-f16.gguf"
        conversion = ({"model": model, "outtype": "f16", "converter_sha256": hash_file(converter)}
                      if not args.model_gguf else {"model": model, "external_gguf": str(gguf)})
        converted_path = root / "model.json"
        if converted_path.exists():
            converted = json.loads(converted_path.read_text(encoding="utf-8"))
            if converted["input"] != conversion or hash_file(gguf) != converted["sha256"]:
                raise ValueError("prepared model changed; choose a new --dataset-dir")
        else:
            if not args.model_gguf:
                if gguf.exists():
                    raise ValueError("unreceipted model GGUF exists; choose a new --dataset-dir")
                temporary = root / "model-f16.partial.gguf"
                run_command([sys.executable, str(converter), str(snapshot), "--outfile", str(temporary),
                             "--outtype", "f16"], root / "conversion.log")
                temporary.replace(gguf)
            converted = {"input": conversion, "sha256": hash_file(gguf), "bytes": gguf.stat().st_size}
            atomic_write_json(converted_path, converted)
        if args.prepare_only:
            print(f"Prepared {root}; rerun without --prepare-only to collect importance.", flush=True)
            return root / "calibration.json"
        partial = root / "imatrix.partial.gguf"
        final = root / "imatrix.gguf"
        if partial.exists() or final.exists():
            raise ValueError("unfinished imatrix exists; preserve it and use a new --dataset-dir; "
                             "no partial is published")
        command = [str(executable), "-m", str(gguf), "-f", str(root / "calibration.txt"),
                   "-o", str(partial), "--output-format", "gguf", "--no-ppl", "--output-frequency", "10",
                   "--ctx-size", str(args.context_size), "--batch-size", str(args.context_size),
                   "--ubatch-size", str(min(args.context_size, 512)), "--chunks", str(args.chunks),
                   "--gpu-layers", str(args.gpu_layers)]
        if args.parse_special:
            command.append("--parse-special")
        log = root / "imatrix.log"
        start = time.monotonic()
        with acquire_device_lease("cuda:0") if args.gpu_layers else nullcontext():
            run_command(command, log)
        result = validate_completion(partial, log, args.context_size, args.chunks)
        partial.replace(final)
        result["filename"] = final.name
        files = [{"filename": name, "sha256": hash_file(root / name)} for name in
                 (final.name, "calibration.txt", "calibration.json", "model.json", "request.json", "imatrix.log")]
        commit = subprocess.run(["git", "-C", str(llama), "rev-parse", "HEAD"], capture_output=True, text=True)
        result.update({"schema": "nanoquant-imatrix-dataset-v1", "status": "complete", "request": requested,
                       "model": model, "model_gguf": converted, "calibration": calibration,
                       "command": command, "llama_cpp_revision": commit.stdout.strip(),
                       "elapsed_seconds": time.monotonic() - start, "files": files,
                       "normalization": "llama-cpp-sums-over-counts"})
        atomic_write_json(receipt_path, result)
        print(f"Completed dataset: {receipt_path}\nSHA-256: {result['sha256']}", flush=True)
        return receipt_path


def main() -> int:
    build(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
