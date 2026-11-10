"""Compare every NanoQuant tensor in a modified llama.cpp GGUF with packed rewrite state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from nanoquant.infrastructure.io_utils import hash_canonical_text_file
from nanoquant.runtime import gemma_gguf_tensor_prefix, open_packed_artifact


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal(name: str, expected: np.ndarray, actual: np.ndarray) -> int:
    if expected.dtype != actual.dtype:
        raise ValueError(f"GGUF dtype differs: {name}: {actual.dtype} != {expected.dtype}")
    if expected.shape != actual.shape:
        raise ValueError(f"GGUF shape differs: {name}: {actual.shape} != {expected.shape}")
    if not np.array_equal(expected, actual):
        raise ValueError(f"GGUF tensor value differs: {name}")
    return int(actual.size)


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()

    packed = open_packed_artifact(args.packed_artifact, verify_hashes=True)
    converter = args.reference_root / "convert_nanoquant_to_gguf.py"
    if hash_canonical_text_file(converter) != packed.manifest.layout.reference.converter_sha256:
        raise ValueError("modified llama.cpp converter hash differs from packed provenance")
    sys.path.insert(0, str(args.reference_root / "gguf-py"))
    import gguf  # type: ignore[import-not-found]  # noqa: PLC0415

    reader = gguf.GGUFReader(args.gguf, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError("GGUF tensor names are duplicated")

    has_residual = any(layer.spec.residual_ranks for block in packed.manifest.blocks for layer in block.layers)
    has_outliers = any(layer.spec.outlier_count for block in packed.manifest.blocks for layer in block.layers)
    expected_metadata = {
        "quantization.nanoquant.version": 3 if has_residual else 2 if has_outliers else 1,
        "quantization.nanoquant.bit_order": "lsb_first",
        "quantization.nanoquant.positive_bit": "0",
        "quantization.nanoquant.tensor_count": packed.manifest.layer_count,
    }
    if has_outliers:
        expected_metadata["quantization.nanoquant.outlier_format"] = "column_side_path"
    if has_residual:
        expected_metadata["quantization.nanoquant.stage_format"] = "additive_binary_v1"
    metadata = {key: reader.fields[key].contents() for key in expected_metadata}
    if metadata != expected_metadata:
        raise ValueError(f"GGUF NanoQuant metadata differs: {metadata}")

    expected_names: set[str] = set()
    array_count = 0
    element_count = 0
    for block in packed.manifest.blocks:
        for layer in block.layers:
            state = packed.load_layer(layer.spec.name)
            prefix = gemma_gguf_tensor_prefix(layer.spec.name)
            terms = [("nq_stage0" if state.residual_stages else "nq", state)] + [
                (f"nq_residual_{index}", stage) for index, stage in enumerate(state.residual_stages)
            ]
            comparisons: list[tuple[str, np.ndarray]] = []
            for term_prefix, term in terms:
                comparisons.extend(
                    [
                        (term_prefix + "_v", _numpy(term.right_words)),
                        (term_prefix + "_u", _numpy(term.left_words)),
                        (term_prefix + "_scale_pre", _numpy(term.scale_pre.to(torch.bfloat16).float())),
                        (term_prefix + "_scale_mid", _numpy(term.scale_mid.to(torch.bfloat16).float())),
                        (term_prefix + "_scale_post", _numpy(term.scale_post.to(torch.bfloat16).float())),
                    ]
                )
            if state.residual_stages:
                key = f"quantization.nanoquant.{prefix}.residual_stage_count"
                if reader.fields[key].contents() != len(state.residual_stages):
                    raise ValueError(f"GGUF residual stage count differs: {prefix}")
            if state.outlier_indices is not None and state.outlier_values is not None:
                expected_values = (
                    state.outlier_values
                    if state.outlier_values.dtype == torch.int8
                    else state.outlier_values.to(torch.float16)
                )
                comparisons.extend(
                    (
                        ("nq_salient_idx", _numpy(state.outlier_indices.int())),
                        ("nq_salient_weight", _numpy(expected_values)),
                    )
                )
            if state.outlier_scales is not None:
                comparisons.append(("nq_salient_scale", _numpy(state.outlier_scales.float())))
            for suffix, expected in comparisons:
                name = f"{prefix}.{suffix}"
                expected_names.add(name)
                try:
                    tensor = cast(Any, tensors[name])
                    actual = tensor.data
                    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                        actual = gguf.dequantize(actual, tensor.tensor_type)
                except KeyError as error:
                    raise ValueError(f"GGUF omitted NanoQuant tensor: {name}") from error
                element_count += _equal(name, expected, actual)
                array_count += 1

    actual_names = {name for name in tensors if ".nq_" in name}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"GGUF NanoQuant inventory differs: missing={missing}, extra={extra}")

    print(
        json.dumps(
            {
                "packed_artifact": str(packed.root),
                "gguf": str(args.gguf.resolve()),
                "gguf_sha256": _hash_file(args.gguf),
                "gguf_bytes": args.gguf.stat().st_size,
                "total_tensor_count": len(tensors),
                "model_shell_tensor_count": len(tensors) - len(actual_names),
                "nanoquant_layer_count": packed.manifest.layer_count,
                "nanoquant_tensor_count": array_count,
                "nanoquant_element_count": element_count,
                "metadata": metadata,
                "exact": True,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
