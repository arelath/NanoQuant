"""Validated, resumable export through the pinned modified llama.cpp converter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.runtime import (
    LlamaCppCheckpointManifest,
    export_llamacpp_checkpoint,
    open_llamacpp_checkpoint,
    open_packed_artifact,
)

GGUF_EXPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GgufExportResult:
    output: Path
    checkpoint: Path
    converter: Path
    bytes: int
    sha256: str
    reused: bool


def _checkpoint_for_packed(
    packed_root: Path,
    checkpoint_root: Path,
) -> LlamaCppCheckpointManifest:
    packed = open_packed_artifact(packed_root, verify_hashes=True)
    descriptor_hash = hash_file(packed.root / "nanoquant-packed-model.json")
    if checkpoint_root.exists():
        checkpoint = open_llamacpp_checkpoint(checkpoint_root, verify_hashes=True)
    else:
        checkpoint = export_llamacpp_checkpoint(packed.root, checkpoint_root)
    if checkpoint.model != packed.manifest.model:
        raise ValueError("llama.cpp checkpoint model differs from packed artifact")
    if checkpoint.reference != packed.manifest.layout.reference:
        raise ValueError("llama.cpp checkpoint reference differs from packed artifact")
    if checkpoint.source_packed_descriptor_sha256 != descriptor_hash:
        raise ValueError("llama.cpp checkpoint is bound to a different packed artifact")
    return checkpoint


def _receipt_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".export.json")


def _reuse_existing(
    output: Path,
    checkpoint_root: Path,
    converter: Path,
    packed_descriptor_hash: str,
) -> GgufExportResult:
    receipt_path = _receipt_path(output)
    if not output.is_file() or not receipt_path.is_file():
        raise FileExistsError(
            "GGUF output or its export receipt exists only partially; remove the partial output before retrying: "
            f"{output}"
        )
    try:
        receipt = cast(dict[str, Any], json.loads(receipt_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("GGUF export receipt is invalid") from exc
    expected = {
        "schema_version": GGUF_EXPORT_SCHEMA_VERSION,
        "packed_descriptor_sha256": packed_descriptor_hash,
        "converter_sha256": hash_file(converter),
        "gguf_sha256": hash_file(output),
        "gguf_bytes": output.stat().st_size,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"GGUF export receipt field differs: {name}")
    if receipt.get("checkpoint") != str(checkpoint_root.resolve()):
        raise ValueError("GGUF export receipt checkpoint path differs")
    return GgufExportResult(
        output.resolve(),
        checkpoint_root.resolve(),
        converter.resolve(),
        output.stat().st_size,
        str(receipt["gguf_sha256"]),
        True,
    )


def export_llamacpp_gguf(
    packed_root: str | Path,
    source_model: str | Path,
    checkpoint_root: str | Path,
    output: str | Path,
    reference_root: str | Path,
    *,
    python_executable: str | Path = sys.executable,
) -> GgufExportResult:
    """Export one packed artifact to GGUF and bind it to a durable receipt.

    Existing complete outputs are hash-verified and reused. Partial or mismatched
    outputs fail closed so an interrupted conversion cannot be mistaken for a
    valid deployment artifact.
    """

    packed = open_packed_artifact(packed_root, verify_hashes=True)
    source = Path(source_model).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"GGUF source model snapshot is missing: {source}")
    reference = Path(reference_root).resolve()
    converter = reference / "convert_nanoquant_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"modified llama.cpp converter is missing: {converter}")
    converter_hash = hash_file(converter)
    expected_converter_hash = packed.manifest.layout.reference.converter_sha256
    if converter_hash != expected_converter_hash:
        raise ValueError(
            "modified llama.cpp converter hash differs from packed provenance: "
            f"{converter_hash} != {expected_converter_hash}"
        )
    checkpoint_path = Path(checkpoint_root).resolve()
    _checkpoint_for_packed(packed.root, checkpoint_path)
    destination = Path(output).resolve()
    packed_descriptor_hash = hash_file(packed.root / "nanoquant-packed-model.json")
    if destination.exists() or _receipt_path(destination).exists():
        return _reuse_existing(
            destination,
            checkpoint_path,
            converter,
            packed_descriptor_hash,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=".gguf",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    stdout_path = destination.with_suffix(destination.suffix + ".converter.stdout.log")
    stderr_path = destination.with_suffix(destination.suffix + ".converter.stderr.log")
    command = (
        str(Path(python_executable)),
        str(converter),
        str(source),
        "--nanoquant-checkpoint",
        str(checkpoint_path),
        "--outfile",
        str(temporary),
        "--outtype",
        "bf16",
        "--no-lazy",
    )
    try:
        with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"modified llama.cpp GGUF converter failed with exit code {completed.returncode}; "
                f"see {stderr_path}"
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("modified llama.cpp converter did not produce a non-empty GGUF")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    digest = hash_file(destination)
    receipt = {
        "schema_version": GGUF_EXPORT_SCHEMA_VERSION,
        "packed_artifact": str(packed.root),
        "packed_descriptor_sha256": packed_descriptor_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_manifest": asdict(open_llamacpp_checkpoint(checkpoint_path, verify_hashes=True)),
        "converter": str(converter),
        "converter_sha256": converter_hash,
        "source_model": str(source),
        "gguf": str(destination),
        "gguf_sha256": digest,
        "gguf_bytes": destination.stat().st_size,
        "command": command,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    atomic_write_json(_receipt_path(destination), receipt)
    return GgufExportResult(
        destination,
        checkpoint_path,
        converter,
        destination.stat().st_size,
        digest,
        False,
    )


__all__ = ["GGUF_EXPORT_SCHEMA_VERSION", "GgufExportResult", "export_llamacpp_gguf"]
