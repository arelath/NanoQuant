"""Validated, resumable export through the pinned modified llama.cpp converter."""

from __future__ import annotations

import json
import os
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
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.subprocess_interop import LlamaCppInterop
from nanoquant.runtime import (
    LlamaCppCheckpointManifest,
    export_llamacpp_checkpoint,
    open_llamacpp_checkpoint,
    open_packed_artifact,
)

GGUF_EXPORT_SCHEMA_VERSION = 6
DEFAULT_TOKEN_EMBEDDING_TYPE = "q8_0"
DEFAULT_OUTPUT_TENSOR_TYPE = "q8_0"


class _StaleAuxiliaryTensorQuantizationError(ValueError):
    """A valid legacy GGUF must be superseded under the current tensor policy."""
SUPPORTED_AUXILIARY_TENSOR_TYPES = frozenset(
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
SUPPORTED_TOKEN_EMBEDDING_TYPES = SUPPORTED_AUXILIARY_TENSOR_TYPES
SUPPORTED_OUTPUT_TENSOR_TYPES = SUPPORTED_AUXILIARY_TENSOR_TYPES


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
    output_tensor_type: str = DEFAULT_OUTPUT_TENSOR_TYPE
    output_tensor_present: bool = False


def _normalize_auxiliary_tensor_type(value: str, tensor: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_AUXILIARY_TENSOR_TYPES:
        supported = ", ".join(sorted(SUPPORTED_AUXILIARY_TENSOR_TYPES))
        raise ValueError(f"unsupported {tensor} quantization type {value!r}; choose one of: {supported}")
    return normalized


def normalize_token_embedding_type(value: str) -> str:
    """Return a supported llama.cpp token-embedding quantization type."""

    return _normalize_auxiliary_tensor_type(value, "token embedding")


def normalize_output_tensor_type(value: str) -> str:
    """Return a supported llama.cpp output-weight quantization type."""

    return _normalize_auxiliary_tensor_type(value, "output tensor")


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


def _quantizer_tensor_policy(packed: Any) -> tuple[str, tuple[str, ...]]:
    """Select a base type that lets llama.cpp preserve typed outlier weights."""

    outlier_types = {
        layer.spec.outlier_value_dtype
        for block in packed.manifest.blocks
        for layer in block.layers
        if layer.spec.outlier_count > 0
    }
    if "int8" not in outlier_types:
        return "F16", ()
    if outlier_types != {"int8"}:
        rendered = ", ".join(sorted(str(value) for value in outlier_types))
        raise ValueError(
            "GGUF export does not support mixed INT8 and floating NanoQuant outlier storage: "
            f"{rendered}"
        )
    # llama-quantize applies --tensor-type only when the base type is itself
    # quantized. With F16 it tries to dequantize I8 before preserving the tensor.
    return "Q8_0", (r"\.nq_salient_weight=I8",)


def _quantizer_command(
    quantizer: Path,
    converted: Path,
    quantized: Path,
    embedding_type: str,
    output_type: str,
    base_type: str,
    tensor_overrides: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        str(quantizer),
        "--output-tensor-type",
        output_type.upper(),
        "--token-embedding-type",
        embedding_type.upper(),
        *(item for override in tensor_overrides for item in ("--tensor-type", override)),
        str(converted),
        str(quantized),
        base_type,
    )


def _inspect_gguf_tensor_contract(
    gguf_path: Path,
    reference: Path,
    python_executable: str | Path,
) -> tuple[str, str | None, int, tuple[str, ...]]:
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
output_type = None
scale_types = []
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        embedding_type = tensor.tensor_type.name.lower()
    if tensor.name == 'output.weight':
        output_type = tensor.tensor_type.name.lower()
    if tensor.name.endswith(('.nq_scale_pre', '.nq_scale_mid', '.nq_scale_post')):
        scale_types.append(tensor.tensor_type.name.lower())
if embedding_type is None:
    raise SystemExit('token_embd.weight is missing')
print(json.dumps({
    'token_embedding_type': embedding_type,
    'output_tensor_type': output_type,
    'nanoquant_scale_tensor_count': len(scale_types),
    'nanoquant_scale_types': sorted(set(scale_types)),
}))
"""
    interop = LlamaCppInterop(reference)
    completed = interop.run(
        interop.request((Path(python_executable), "-c", program, gguf_python, gguf_path))
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"failed to inspect GGUF tensor contract: {detail}")
    try:
        payload = cast(dict[str, Any], json.loads(completed.stdout))
        embedding_type = str(payload["token_embedding_type"]).lower()
        raw_output_type = payload["output_tensor_type"]
        output_type = None if raw_output_type is None else str(raw_output_type).lower()
        scale_count = int(payload["nanoquant_scale_tensor_count"])
        scale_types = tuple(str(value).lower() for value in payload["nanoquant_scale_types"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("GGUF tensor contract inspection returned invalid output") from exc
    return embedding_type, output_type, scale_count, scale_types


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


def _source_output_tensor_names(source: Path) -> tuple[str, ...]:
    inventory = SafetensorsModelSource(
        source,
        source=str(source),
        revision="gguf-export",
        verify_hashes=False,
    ).tensor_metadata()
    return tuple(
        metadata.key
        for metadata in inventory
        if metadata.key in {"lm_head.weight", "model.lm_head.weight", "output.weight", "model.output.weight"}
    )


def _require_output_tensor_type(
    actual_type: str | None,
    requested_type: str,
    source_output_tensors: tuple[str, ...] = (),
) -> None:
    if source_output_tensors and actual_type is None:
        rendered = ", ".join(source_output_tensors)
        raise ValueError(
            f"source output tensor {rendered} did not map to canonical GGUF output.weight"
        )
    if actual_type is not None and actual_type != requested_type:
        raise ValueError(
            f"GGUF output tensor type differs from export recipe: {actual_type} != {requested_type}"
        )


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
    output_tensor_type: str,
    source_output_tensors: tuple[str, ...],
    expected_scale_count: int,
    reference: Path,
    python_executable: str | Path,
    quantizer_base_type: str,
    quantizer_tensor_overrides: tuple[str, ...],
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
    actual_embedding_type, actual_output_type, scale_count, scale_types = _inspect_gguf_tensor_contract(
        output,
        reference,
        python_executable,
    )
    _require_bfloat16_nanoquant_scales(scale_count, scale_types, expected_scale_count)
    if actual_embedding_type != token_embedding_type:
        raise ValueError(
            "GGUF token embedding tensor type differs from export recipe: "
            f"{actual_embedding_type} != {token_embedding_type}"
        )
    legacy_binding = {
        "packed_descriptor_sha256": packed_descriptor_hash,
        "converter_sha256": hash_canonical_text_file(converter),
        "token_embedding_type": token_embedding_type,
        "nanoquant_scale_type": "bf16",
        "nanoquant_scale_tensor_count": scale_count,
    }
    schema_version = receipt.get("schema_version")
    legacy_bound = (
        bool(source_output_tensors)
        and type(schema_version) is int
        and schema_version < GGUF_EXPORT_SCHEMA_VERSION
        and all(receipt.get(name) == value for name, value in legacy_binding.items())
        and receipt.get("checkpoint") == str(checkpoint_root.resolve())
    )
    if legacy_bound:
        raise _StaleAuxiliaryTensorQuantizationError(
            "legacy GGUF must be rebuilt under the current auxiliary tensor policy"
        )
    _require_output_tensor_type(actual_output_type, output_tensor_type, source_output_tensors)
    expected = {
        "packed_descriptor_sha256": packed_descriptor_hash,
        "converter_sha256": hash_canonical_text_file(converter),
        "quantizer_sha256": hash_file(quantizer),
        "token_embedding_type": token_embedding_type,
        "output_tensor_type": output_tensor_type,
        "output_tensor_present": actual_output_type is not None,
        "source_output_tensors": list(source_output_tensors),
        "nanoquant_scale_type": "bf16",
        "nanoquant_scale_tensor_count": scale_count,
        "gguf_sha256": hash_file(output),
        "gguf_bytes": output.stat().st_size,
    }
    if schema_version == GGUF_EXPORT_SCHEMA_VERSION:
        expected.update(
            {
                "schema_version": GGUF_EXPORT_SCHEMA_VERSION,
                "quantizer_base_type": quantizer_base_type.lower(),
                "quantizer_tensor_overrides": list(quantizer_tensor_overrides),
            }
        )
    elif schema_version == 5 and quantizer_base_type == "F16" and not quantizer_tensor_overrides:
        # Schema 5 used this exact floating-sidecar policy but did not name it
        # separately from quantizer_command. Preserve validated historical GGUFs.
        expected["schema_version"] = 5
    else:
        raise ValueError("GGUF export receipt schema differs from the active tensor policy")
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
        token_embedding_type,
        quantizer,
        output_tensor_type=output_tensor_type,
        output_tensor_present=actual_output_type is not None,
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
    output_tensor_type: str = DEFAULT_OUTPUT_TENSOR_TYPE,
    converter_path: str | Path | None = None,
) -> GgufExportResult:
    """Export one packed artifact to GGUF and bind it to a durable receipt.

    Existing complete outputs are hash-verified and reused. Bound legacy exports
    that predate independent output-tensor quantization are atomically rebuilt.
    Partial or current-schema mismatched outputs fail closed so an interrupted
    conversion cannot be mistaken for a valid deployment artifact.
    """

    embedding_type = normalize_token_embedding_type(token_embedding_type)
    requested_output_type = normalize_output_tensor_type(output_tensor_type)
    packed = open_packed_artifact(packed_root, verify_hashes=True)
    quantizer_base_type, quantizer_tensor_overrides = _quantizer_tensor_policy(packed)
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
    source_output_tensors = _source_output_tensor_names(source)
    quantizer = _find_quantizer(reference)
    checkpoint_path = Path(checkpoint_root).resolve()
    _checkpoint_for_packed(packed.root, checkpoint_path)
    destination = Path(output).resolve()
    packed_descriptor_hash = hash_file(packed.root / "nanoquant-packed-model.json")
    expected_scale_count = packed.manifest.layer_count * 3
    if destination.exists() or _receipt_path(destination).exists():
        try:
            result = _reuse_existing(
                destination,
                checkpoint_path,
                converter,
                quantizer,
                packed_descriptor_hash,
                embedding_type,
                requested_output_type,
                source_output_tensors,
                expected_scale_count,
                reference,
                python_executable,
                quantizer_base_type,
                quantizer_tensor_overrides,
            )
        except _StaleAuxiliaryTensorQuantizationError:
            print(
                "Existing legacy GGUF uses stale auxiliary tensor quantization; "
                f"rebuilding atomically: {destination}",
                flush=True,
            )
        else:
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
    interop = LlamaCppInterop(reference)
    converter_environment = interop.converter_environment(converter)
    # COPY disables llama.cpp's per-tensor overrides. Floating sidecars use F16;
    # I8 salient weights require a quantized base before their exact preservation
    # override is honored. Embedding/output overrides remain explicit in both cases.
    quantizer_command = _quantizer_command(
        quantizer,
        converted,
        quantized,
        embedding_type,
        requested_output_type,
        quantizer_base_type,
        quantizer_tensor_overrides,
    )
    try:
        with (
            converter_stdout_path.open("w", encoding="utf-8", newline="\n") as stdout,
            converter_stderr_path.open("w", encoding="utf-8", newline="\n") as stderr,
        ):
            completed = interop.run(
                interop.request(converter_command, environment=converter_environment),
                stdout=stdout,
                stderr=stderr,
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
            completed = interop.run(
                interop.request(quantizer_command),
                stdout=stdout,
                stderr=stderr,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp auxiliary tensor quantization failed with exit code {completed.returncode}; "
                f"see {quantizer_stderr_path}"
            )
        if not quantized.is_file() or quantized.stat().st_size == 0:
            raise RuntimeError("llama.cpp quantizer did not produce a non-empty GGUF")
        actual_embedding_type, actual_output_type, scale_count, scale_types = _inspect_gguf_tensor_contract(
            quantized,
            reference,
            python_executable,
        )
        if actual_embedding_type != embedding_type:
            raise RuntimeError(
                "GGUF token embedding quantization did not produce the requested tensor type: "
                f"{actual_embedding_type} != {embedding_type}"
            )
        try:
            _require_output_tensor_type(
                actual_output_type,
                requested_output_type,
                source_output_tensors,
            )
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
        "quantizer_base_type": quantizer_base_type.lower(),
        "quantizer_tensor_overrides": list(quantizer_tensor_overrides),
        "source_model": str(source),
        "token_embedding_type": embedding_type,
        "token_embedding_tensor": "token_embd.weight",
        "output_tensor_type": requested_output_type,
        "output_tensor": "output.weight" if actual_output_type is not None else None,
        "output_tensor_present": actual_output_type is not None,
        "source_output_tensors": list(source_output_tensors),
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
        requested_output_type,
        actual_output_type is not None,
    )


__all__ = [
    "DEFAULT_OUTPUT_TENSOR_TYPE",
    "DEFAULT_TOKEN_EMBEDDING_TYPE",
    "GGUF_EXPORT_SCHEMA_VERSION",
    "SUPPORTED_OUTPUT_TENSOR_TYPES",
    "SUPPORTED_TOKEN_EMBEDDING_TYPES",
    "GgufExportResult",
    "export_llamacpp_gguf",
    "normalize_output_tensor_type",
    "normalize_token_embedding_type",
]
