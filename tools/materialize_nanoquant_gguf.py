"""Materialize staged NanoQuant GGUF projections as FP32 for quality evaluation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from weight_image_common import DEFAULT_LLAMA_ROOT, load_gguf_module, reconstruct_nanoquant_weight, tensor_map

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file


def _gemma_qkv_sizes(fields: dict[str, Any], block: int) -> tuple[int, int, int]:
    def integer(key: str) -> int:
        value = fields[f"gemma3.attention.{key}"].contents()
        return int(value[block] if isinstance(value, list) else value)

    return (
        integer("head_count") * integer("key_length"),
        integer("head_count_kv") * integer("key_length"),
        integer("head_count_kv") * integer("value_length"),
    )


def _verify_unchanged(
    output: Path, gguf: Any, tensors: dict[str, Any], untouched: set[str], projection_count: int,
) -> None:
    written = tensor_map(gguf.GGUFReader(output, "r"))
    if len(written) != projection_count + len(untouched):
        raise ValueError("Materialized GGUF has an unexpected tensor inventory")
    for name in untouched:
        original, copied = tensors[name], written[name]
        if copied.tensor_type != original.tensor_type or not np.array_equal(
            copied.data.view(np.uint8), original.data.view(np.uint8),
        ):
            raise ValueError(f"Untouched tensor {name} changed during materialization")


def materialize_gguf(source: Path, output: Path, llama_cpp_root: Path) -> dict[str, Any]:
    if source.resolve() == output.resolve():
        raise ValueError("The materialized file must differ from the packed source")
    receipt_path = output.with_suffix(".materialization.json")
    source_hash = hash_file(source)
    if output.is_file() and receipt_path.is_file():
        receipt = cast(dict[str, Any], json.loads(receipt_path.read_text(encoding="utf-8")))
        if receipt["source_sha256"] == source_hash and receipt["materialized_sha256"] == hash_file(output):
            return receipt
    gguf = load_gguf_module(llama_cpp_root)
    reader = gguf.GGUFReader(source, "r")
    fields = reader.fields
    architecture = fields["general.architecture"].contents()
    if fields["quantization.nanoquant.stage_format"].contents() != "additive_binary_v1":
        raise ValueError("Expected independently scaled additive NanoQuant factors")
    tensors = tensor_map(reader)
    bases = {
        name.removesuffix(suffix)
        for name in tensors for suffix in (".nq_stage0_v", ".nq_v")
        if name.endswith(suffix)
    }
    if not bases:
        raise ValueError("No staged NanoQuant projections found")
    writer = gguf.GGUFWriter(None, architecture, endianess=reader.endianess)
    for name, field in fields.items():
        if (
            name.startswith(("GGUF.", "quantization.nanoquant."))
            or name in {"general.architecture", "general.file_type"}
        ):
            continue
        if name == "general.alignment":
            writer.add_custom_alignment(int(field.contents()))
        else:
            value_type = field.types[0]
            subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(name, field.contents(), value_type, sub_type=subtype)
    untouched = set(tensors)
    projection_count = 0
    for base in sorted(bases):
        count_field = fields.get(f"quantization.nanoquant.{base}.residual_stage_count")
        terms: tuple[str, ...]
        if count_field is None:
            terms = ("nq",)
        else:
            terms = ("nq_stage0", *(f"nq_residual_{index}" for index in range(int(count_field.contents()))))
        dense = None
        for term in terms:
            suffixes: tuple[str, ...] = ("v", "u", "scale_pre", "scale_mid", "scale_post")
            if term == "nq":
                suffixes += ("salient_idx", "salient_weight", "salient_scale")
            untouched.difference_update(f"{base}.{term}_{suffix}" for suffix in suffixes)
            weight = reconstruct_nanoquant_weight(tensors, base, gguf, term=term)
            if dense is None:
                dense = weight
            else:
                if dense.shape != weight.shape:
                    raise ValueError(f"Stage shape differs for {base}.{term}")
                dense += weight
        assert dense is not None
        if base.endswith(".attn_qkv"):
            if architecture != "gemma3":
                raise ValueError(f"Stacked QKV materialization is unsupported for {architecture}")
            sizes = _gemma_qkv_sizes(fields, int(base.split(".")[1]))
            if sum(sizes) != dense.shape[0]:
                raise ValueError(f"Invalid stacked QKV shape for {base}: {dense.shape}")
            block = base.removesuffix("_qkv")
            for name, matrix in zip(("q", "k", "v"), np.split(dense, np.cumsum(sizes)[:-1], axis=0), strict=True):
                writer.add_tensor(f"{block}_{name}.weight", matrix)
            projection_count += 3
        else:
            writer.add_tensor(f"{base}.weight", dense)
            projection_count += 1
    for name in sorted(untouched):
        if ".nq_" in name:
            raise ValueError(f"Unconsumed NanoQuant sidecar {name}")
        tensor = tensors[name]
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".materialize-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        writer.write_header_to_file(staged)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        _verify_unchanged(staged, gguf, tensors, untouched, projection_count)
        os.replace(staged, output)
    receipt = {
        "source": str(source.resolve()), "source_sha256": source_hash, "source_bytes": source.stat().st_size,
        "materialized": str(output.resolve()), "materialized_sha256": hash_file(output),
        "materialized_bytes": output.stat().st_size, "fp32_projections": projection_count,
        "unchanged_tensors": len(untouched),
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--llama-cpp-root", type=Path, default=DEFAULT_LLAMA_ROOT)
    args = parser.parse_args()
    print(json.dumps(materialize_gguf(args.source, args.output, args.llama_cpp_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
