"""Validated, resumable BF16 multimodal-projector export through llama.cpp."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.runtime import PACKED_REFERENCE_COMMIT

MMPROJ_EXPORT_SCHEMA_VERSION = 1
MMPROJ_OUTPUT_NAME = "mmproj-BF16.gguf"
MMPROJ_REFERENCE_COMMIT = PACKED_REFERENCE_COMMIT
MMPROJ_CONVERTER_SHA256 = "3b4064d368d8e5a2c6fe64e031652c463787d5c47a4aaa08e5f68314d6307ea3"
_MOSTLY_BFLOAT16_FILE_TYPE = 32


@dataclass(frozen=True, slots=True)
class MmprojExportResult:
    output: Path
    converter: Path
    bytes: int
    sha256: str
    tensor_count: int
    tensor_types: tuple[str, ...]
    reused: bool


def source_has_vision_stack(snapshot: str | Path) -> bool:
    """Return whether a local Hugging Face snapshot declares a vision stack."""

    source = Path(snapshot).resolve()
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    try:
        config = cast(dict[str, Any], json.loads(config_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"model config is invalid: {config_path}") from exc
    vision = config.get("vision_config")
    if vision is None:
        return False
    if not isinstance(vision, Mapping) or not vision:
        raise ValueError("model vision_config must be a non-empty object")
    return True


def _receipt_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".export.json")


def _inspect_mmproj(
    output: Path,
    reference: Path,
    python_executable: str | Path,
) -> tuple[str, int, int, tuple[str, ...]]:
    gguf_python = reference / "gguf-py"
    if not gguf_python.is_dir():
        raise FileNotFoundError(f"llama.cpp GGUF Python package is missing: {gguf_python}")
    program = """import json
import sys
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
reader = GGUFReader(sys.argv[2])
def scalar(key):
    field = reader.get_field(key)
    if field is None:
        raise SystemExit(f'{key} is missing')
    value = field.contents()
    return value.item() if hasattr(value, 'item') else value
print(json.dumps({
    'general_type': str(scalar('general.type')),
    'file_type': int(scalar('general.file_type')),
    'tensor_count': len(reader.tensors),
    'tensor_types': sorted({tensor.tensor_type.name.lower() for tensor in reader.tensors}),
}))
"""
    completed = subprocess.run(
        (str(Path(python_executable)), "-c", program, str(gguf_python), str(output)),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"failed to inspect mmproj GGUF: {detail}")
    try:
        payload = cast(dict[str, Any], json.loads(completed.stdout))
        return (
            str(payload["general_type"]).lower(),
            int(payload["file_type"]),
            int(payload["tensor_count"]),
            tuple(str(value).lower() for value in payload["tensor_types"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("mmproj GGUF inspection returned invalid output") from exc


def _validate_contract(
    general_type: str,
    file_type: int,
    tensor_count: int,
    tensor_types: tuple[str, ...],
) -> None:
    if general_type != "mmproj":
        raise ValueError(f"GGUF general.type is not mmproj: {general_type}")
    if file_type != _MOSTLY_BFLOAT16_FILE_TYPE:
        raise ValueError(f"mmproj GGUF is not MOSTLY_BF16: file type {file_type}")
    if tensor_count <= 0 or not tensor_types:
        raise ValueError("mmproj GGUF contains no tensors")


def export_mmproj_bfloat16(
    source_model: str | Path,
    output: str | Path,
    reference_root: str | Path,
    *,
    python_executable: str | Path = sys.executable,
) -> MmprojExportResult:
    """Export one vision stack to a validated ``mmproj-BF16.gguf`` artifact."""

    source = Path(source_model).resolve()
    if not source_has_vision_stack(source):
        raise ValueError("model snapshot does not declare a vision stack")
    destination = Path(output).resolve()
    if destination.name != MMPROJ_OUTPUT_NAME:
        raise ValueError(f"mmproj output must be named {MMPROJ_OUTPUT_NAME}")
    reference = Path(reference_root).resolve()
    converter = reference / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"llama.cpp multimodal converter is missing: {converter}")
    converter_hash = hash_file(converter)
    if converter_hash != MMPROJ_CONVERTER_SHA256:
        raise ValueError(
            "llama.cpp multimodal converter hash differs from pinned provenance: "
            f"{converter_hash} != {MMPROJ_CONVERTER_SHA256}"
        )
    config_hash = hash_file(source / "config.json")
    receipt_path = _receipt_path(destination)
    if destination.exists() or receipt_path.exists():
        if not destination.is_file() or not receipt_path.is_file():
            raise FileExistsError(
                "mmproj output or receipt exists only partially; remove the partial output before retrying: "
                f"{destination}"
            )
        try:
            receipt = cast(dict[str, Any], json.loads(receipt_path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("mmproj export receipt is invalid") from exc
        general_type, file_type, tensor_count, tensor_types = _inspect_mmproj(
            destination, reference, python_executable
        )
        _validate_contract(general_type, file_type, tensor_count, tensor_types)
        expected = {
            "schema_version": MMPROJ_EXPORT_SCHEMA_VERSION,
            "source_model": str(source),
            "source_config_sha256": config_hash,
            "reference_commit": MMPROJ_REFERENCE_COMMIT,
            "converter_sha256": converter_hash,
            "output_sha256": hash_file(destination),
            "output_bytes": destination.stat().st_size,
            "tensor_count": tensor_count,
            "tensor_types": list(tensor_types),
            "file_type": "mostly_bfloat16",
        }
        for name, value in expected.items():
            if receipt.get(name) != value:
                raise ValueError(f"mmproj export receipt field differs: {name}")
        return MmprojExportResult(
            destination,
            converter,
            destination.stat().st_size,
            str(receipt["output_sha256"]),
            tensor_count,
            tensor_types,
            True,
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
        "--mmproj",
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
                f"llama.cpp mmproj converter failed with exit code {completed.returncode}; see {stderr_path}"
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("llama.cpp mmproj converter did not produce a non-empty GGUF")
        general_type, file_type, tensor_count, tensor_types = _inspect_mmproj(
            temporary, reference, python_executable
        )
        _validate_contract(general_type, file_type, tensor_count, tensor_types)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    digest = hash_file(destination)
    atomic_write_json(
        receipt_path,
        {
            "schema_version": MMPROJ_EXPORT_SCHEMA_VERSION,
            "source_model": str(source),
            "source_config_sha256": config_hash,
            "reference_commit": MMPROJ_REFERENCE_COMMIT,
            "converter": str(converter),
            "converter_sha256": converter_hash,
            "output": str(destination),
            "output_sha256": digest,
            "output_bytes": destination.stat().st_size,
            "general_type": general_type,
            "file_type": "mostly_bfloat16",
            "tensor_count": tensor_count,
            "tensor_types": list(tensor_types),
            "command": command,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        },
    )
    return MmprojExportResult(
        destination,
        converter,
        destination.stat().st_size,
        digest,
        tensor_count,
        tensor_types,
        False,
    )


__all__ = [
    "MMPROJ_CONVERTER_SHA256",
    "MMPROJ_EXPORT_SCHEMA_VERSION",
    "MMPROJ_OUTPUT_NAME",
    "MMPROJ_REFERENCE_COMMIT",
    "MmprojExportResult",
    "export_mmproj_bfloat16",
    "source_has_vision_stack",
]
