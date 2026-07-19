"""Validated, resumable export through the pinned modified llama.cpp converter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_canonical_text_file, hash_file
from nanoquant.infrastructure.mmproj_export import (
    MMPROJ_OUTPUT_NAME,
    MmprojExportResult,
    export_mmproj_bfloat16,
    source_has_vision_stack,
)
from nanoquant.runtime import (
    LlamaCppCheckpointManifest,
    export_llamacpp_checkpoint,
    open_llamacpp_checkpoint,
    open_packed_artifact,
)

GGUF_EXPORT_SCHEMA_VERSION = 3
DEFAULT_TOKEN_EMBEDDING_TYPE = "q8_0"
SUPPORTED_TOKEN_EMBEDDING_TYPES = frozenset(
    {
        "q4_0",
        "q4_1",
        "q4_k",
        "q4_k_m",
        "q4_k_s",
        "q5_0",
        "q5_1",
        "q5_k",
        "q5_k_m",
        "q5_k_s",
        "q6_k",
        "q8_0",
    }
)


@dataclass(frozen=True, slots=True)
class GgufExportResult:
    output: Path
    checkpoint: Path
    converter: Path
    bytes: int
    sha256: str
    reused: bool
    token_embedding_type: str = DEFAULT_TOKEN_EMBEDDING_TYPE
    quantizer: Path | None = None
    mmproj: MmprojExportResult | None = None


def normalize_token_embedding_type(value: str) -> str:
    """Return a supported llama.cpp token-embedding quantization type."""

    normalized = value.strip().lower()
    if normalized not in SUPPORTED_TOKEN_EMBEDDING_TYPES:
        supported = ", ".join(sorted(SUPPORTED_TOKEN_EMBEDDING_TYPES))
        raise ValueError(f"unsupported token embedding quantization type {value!r}; choose one of: {supported}")
    return normalized


def _find_quantizer(reference: Path) -> Path:
    candidates = (
        reference / "build" / "bin" / "Release" / "llama-quantize.exe",
        reference / "build" / "bin" / "llama-quantize.exe",
        reference / "build" / "bin" / "llama-quantize",
        reference / "llama-quantize.exe",
        reference / "llama-quantize",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"llama.cpp quantizer is missing; searched: {searched}")


def _inspect_gguf_tensor_contract(
    gguf_path: Path,
    reference: Path,
    python_executable: str | Path,
) -> tuple[str, int, tuple[str, ...]]:
    """Read deployment-critical tensor types with llama.cpp's pinned GGUF reader."""

    gguf_python = reference / "gguf-py"
    if not gguf_python.is_dir():
        raise FileNotFoundError(f"llama.cpp GGUF Python package is missing: {gguf_python}")
    program = """import json
import sys
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
reader = GGUFReader(sys.argv[2])
embedding_type = None
scale_types = []
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        embedding_type = tensor.tensor_type.name.lower()
    if tensor.name.endswith(('.nq_scale_pre', '.nq_scale_mid', '.nq_scale_post')):
        scale_types.append(tensor.tensor_type.name.lower())
if embedding_type is None:
    raise SystemExit('token_embd.weight is missing')
print(json.dumps({
    'token_embedding_type': embedding_type,
    'nanoquant_scale_tensor_count': len(scale_types),
    'nanoquant_scale_types': sorted(set(scale_types)),
}))
"""
    completed = subprocess.run(
        (str(Path(python_executable)), "-c", program, str(gguf_python), str(gguf_path)),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"failed to inspect GGUF tensor contract: {detail}")
    try:
        payload = cast(dict[str, Any], json.loads(completed.stdout))
        embedding_type = str(payload["token_embedding_type"]).lower()
        scale_count = int(payload["nanoquant_scale_tensor_count"])
        scale_types = tuple(str(value).lower() for value in payload["nanoquant_scale_types"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("GGUF tensor contract inspection returned invalid output") from exc
    return embedding_type, scale_count, scale_types


def _require_bfloat16_nanoquant_scales(
    scale_count: int,
    scale_types: tuple[str, ...],
    expected_scale_count: int,
) -> None:
    if scale_count != expected_scale_count:
        raise ValueError(f"GGUF NanoQuant scale tensor count differs: {scale_count} != {expected_scale_count}")
    if scale_types != ("bf16",):
        rendered = ", ".join(scale_types) or "none"
        raise ValueError(f"GGUF NanoQuant scale tensors must all be BF16, found: {rendered}")


def _export_mmproj_for_source(
    source: Path,
    destination: Path,
    reference: Path,
    python_executable: str | Path,
) -> MmprojExportResult | None:
    if not source_has_vision_stack(source):
        return None
    return export_mmproj_bfloat16(
        source,
        destination.parent / MMPROJ_OUTPUT_NAME,
        reference,
        python_executable=python_executable,
    )


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
    quantizer: Path,
    packed_descriptor_hash: str,
    token_embedding_type: str,
    expected_scale_count: int,
    reference: Path,
    python_executable: str | Path,
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
    actual_type, scale_count, scale_types = _inspect_gguf_tensor_contract(output, reference, python_executable)
    _require_bfloat16_nanoquant_scales(scale_count, scale_types, expected_scale_count)
    expected = {
        "schema_version": GGUF_EXPORT_SCHEMA_VERSION,
        "packed_descriptor_sha256": packed_descriptor_hash,
        "converter_sha256": hash_canonical_text_file(converter),
        "quantizer_sha256": hash_file(quantizer),
        "token_embedding_type": token_embedding_type,
        "nanoquant_scale_type": "bf16",
        "nanoquant_scale_tensor_count": scale_count,
        "gguf_sha256": hash_file(output),
        "gguf_bytes": output.stat().st_size,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"GGUF export receipt field differs: {name}")
    if receipt.get("checkpoint") != str(checkpoint_root.resolve()):
        raise ValueError("GGUF export receipt checkpoint path differs")
    if actual_type != token_embedding_type:
        raise ValueError(
            f"GGUF token embedding tensor type differs from export recipe: {actual_type} != {token_embedding_type}"
        )
    return GgufExportResult(
        output.resolve(),
        checkpoint_root.resolve(),
        converter.resolve(),
        output.stat().st_size,
        str(receipt["gguf_sha256"]),
        True,
        token_embedding_type,
        quantizer,
    )


def export_llamacpp_gguf(
    packed_root: str | Path,
    source_model: str | Path,
    checkpoint_root: str | Path,
    output: str | Path,
    reference_root: str | Path,
    *,
    python_executable: str | Path = sys.executable,
    token_embedding_type: str = DEFAULT_TOKEN_EMBEDDING_TYPE,
    converter_path: str | Path | None = None,
) -> GgufExportResult:
    """Export one packed artifact to GGUF and bind it to a durable receipt.

    Existing complete outputs are hash-verified and reused. Partial or mismatched
    outputs fail closed so an interrupted conversion cannot be mistaken for a
    valid deployment artifact.
    """

    embedding_type = normalize_token_embedding_type(token_embedding_type)
    packed = open_packed_artifact(packed_root, verify_hashes=True)
    source = Path(source_model).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"GGUF source model snapshot is missing: {source}")
    reference = Path(reference_root).resolve()
    converter = reference / "convert_nanoquant_to_gguf.py" if converter_path is None else Path(converter_path).resolve()
    if not converter.is_file():
        raise FileNotFoundError(f"modified llama.cpp converter is missing: {converter}")
    converter_hash = hash_canonical_text_file(converter)
    expected_converter_hash = packed.manifest.layout.reference.converter_sha256
    if converter_hash != expected_converter_hash:
        raise ValueError(
            "modified llama.cpp converter hash differs from packed provenance: "
            f"{converter_hash} != {expected_converter_hash}"
        )
    quantizer = _find_quantizer(reference)
    checkpoint_path = Path(checkpoint_root).resolve()
    _checkpoint_for_packed(packed.root, checkpoint_path)
    destination = Path(output).resolve()
    packed_descriptor_hash = hash_file(packed.root / "nanoquant-packed-model.json")
    expected_scale_count = packed.manifest.layer_count * 3
    if destination.exists() or _receipt_path(destination).exists():
        result = _reuse_existing(
            destination,
            checkpoint_path,
            converter,
            quantizer,
            packed_descriptor_hash,
            embedding_type,
            expected_scale_count,
            reference,
            python_executable,
        )
        mmproj = _export_mmproj_for_source(source, destination, reference, python_executable)
        return replace(result, mmproj=mmproj)

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, converted_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=".converted.gguf",
        dir=destination.parent,
    )
    os.close(descriptor)
    converted = Path(converted_name)
    converted.unlink()
    descriptor, quantized_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=".quantized.gguf",
        dir=destination.parent,
    )
    os.close(descriptor)
    quantized = Path(quantized_name)
    quantized.unlink()
    converter_stdout_path = destination.with_suffix(destination.suffix + ".converter.stdout.log")
    converter_stderr_path = destination.with_suffix(destination.suffix + ".converter.stderr.log")
    quantizer_stdout_path = destination.with_suffix(destination.suffix + ".quantizer.stdout.log")
    quantizer_stderr_path = destination.with_suffix(destination.suffix + ".quantizer.stderr.log")
    converter_command = (
        str(Path(python_executable)),
        str(converter),
        str(source),
        "--nanoquant-checkpoint",
        str(checkpoint_path),
        "--outfile",
        str(converted),
        "--outtype",
        "bf16",
        "--no-lazy",
    )
    converter_environment = None
    if converter.parent != reference:
        converter_environment = os.environ.copy()
        python_paths = (reference, reference / "gguf-py")
        existing_python_path = converter_environment.get("PYTHONPATH")
        converter_environment["PYTHONPATH"] = os.pathsep.join(
            (
                *(str(path) for path in python_paths),
                *((existing_python_path,) if existing_python_path else ()),
            )
        )
        converter_environment["NO_LOCAL_GGUF"] = "1"
    # COPY disables llama.cpp's per-tensor overrides. F16 is intentional here:
    # the converter's NanoQuant sidecars are already BF16/F16/I32/F32, so this base
    # type leaves them alone while allowing token_embd.weight to be overridden.
    quantizer_command = (
        str(quantizer),
        "--token-embedding-type",
        embedding_type.upper(),
        str(converted),
        str(quantized),
        "F16",
    )
    try:
        with (
            converter_stdout_path.open("w", encoding="utf-8", newline="\n") as stdout,
            converter_stderr_path.open("w", encoding="utf-8", newline="\n") as stderr,
        ):
            completed = subprocess.run(
                converter_command,
                stdout=stdout,
                stderr=stderr,
                check=False,
                env=converter_environment,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"modified llama.cpp GGUF converter failed with exit code {completed.returncode}; "
                f"see {converter_stderr_path}"
            )
        if not converted.is_file() or converted.stat().st_size == 0:
            raise RuntimeError("modified llama.cpp converter did not produce a non-empty GGUF")
        with (
            quantizer_stdout_path.open("w", encoding="utf-8", newline="\n") as stdout,
            quantizer_stderr_path.open("w", encoding="utf-8", newline="\n") as stderr,
        ):
            completed = subprocess.run(quantizer_command, stdout=stdout, stderr=stderr, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp token embedding quantization failed with exit code {completed.returncode}; "
                f"see {quantizer_stderr_path}"
            )
        if not quantized.is_file() or quantized.stat().st_size == 0:
            raise RuntimeError("llama.cpp quantizer did not produce a non-empty GGUF")
        actual_type, scale_count, scale_types = _inspect_gguf_tensor_contract(quantized, reference, python_executable)
        if actual_type != embedding_type:
            raise RuntimeError(
                "GGUF token embedding quantization did not produce the requested tensor type: "
                f"{actual_type} != {embedding_type}"
            )
        try:
            _require_bfloat16_nanoquant_scales(scale_count, scale_types, expected_scale_count)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        os.replace(quantized, destination)
    finally:
        converted.unlink(missing_ok=True)
        quantized.unlink(missing_ok=True)

    digest = hash_file(destination)
    receipt = {
        "schema_version": GGUF_EXPORT_SCHEMA_VERSION,
        "packed_artifact": str(packed.root),
        "packed_descriptor_sha256": packed_descriptor_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_manifest": asdict(open_llamacpp_checkpoint(checkpoint_path, verify_hashes=True)),
        "converter": str(converter),
        "converter_sha256": converter_hash,
        "quantizer": str(quantizer),
        "quantizer_sha256": hash_file(quantizer),
        "source_model": str(source),
        "token_embedding_type": embedding_type,
        "token_embedding_tensor": "token_embd.weight",
        "nanoquant_scale_type": "bf16",
        "nanoquant_scale_tensor_count": scale_count,
        "gguf": str(destination),
        "gguf_sha256": digest,
        "gguf_bytes": destination.stat().st_size,
        "converter_command": converter_command,
        "converter_stdout_log": str(converter_stdout_path),
        "converter_stderr_log": str(converter_stderr_path),
        "quantizer_command": quantizer_command,
        "quantizer_stdout_log": str(quantizer_stdout_path),
        "quantizer_stderr_log": str(quantizer_stderr_path),
    }
    atomic_write_json(_receipt_path(destination), receipt)
    mmproj = _export_mmproj_for_source(source, destination, reference, python_executable)
    return GgufExportResult(
        destination,
        checkpoint_path,
        converter,
        destination.stat().st_size,
        digest,
        False,
        embedding_type,
        quantizer,
        mmproj,
    )


__all__ = [
    "DEFAULT_TOKEN_EMBEDDING_TYPE",
    "GGUF_EXPORT_SCHEMA_VERSION",
    "SUPPORTED_TOKEN_EMBEDDING_TYPES",
    "GgufExportResult",
    "export_llamacpp_gguf",
    "normalize_token_embedding_type",
]
