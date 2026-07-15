"""Validate rewrite checkpoint shards through the pinned modified llama.cpp converter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import torch

from nanoquant.runtime import (
    LLAMACPP_CHECKPOINT_FORMAT,
    LLAMACPP_CHECKPOINT_SCHEMA_VERSION,
    gemma_hf_checkpoint_prefix,
    llamacpp_checkpoint_tensors,
    open_packed_artifact,
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_converter(reference_root: Path, expected_sha256: str) -> ModuleType:
    converter_path = reference_root / "convert_nanoquant_to_gguf.py"
    if _hash_file(converter_path) != expected_sha256:
        raise ValueError("modified llama.cpp converter hash differs from packed provenance")
    module_name = "nanoquant_pinned_llamacpp_converter"
    sys.path.insert(0, str(reference_root))
    specification = importlib.util.spec_from_file_location(module_name, converter_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load modified llama.cpp converter: {converter_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _equal(name: str, expected: np.ndarray, actual: np.ndarray) -> int:
    if expected.dtype != actual.dtype:
        raise ValueError(
            f"modified llama.cpp sidecar dtype differs: {name}: {actual.dtype} != {expected.dtype}"
        )
    if expected.shape != actual.shape:
        raise ValueError(
            f"modified llama.cpp sidecar shape differs: {name}: {actual.shape} != {expected.shape}"
        )
    if not np.array_equal(expected, actual):
        raise ValueError(f"modified llama.cpp sidecar value differs: {name}")
    return int(actual.size)


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()

    packed = open_packed_artifact(args.packed_artifact, verify_hashes=True)
    descriptor_path = args.checkpoint / "nanoquant-llamacpp-checkpoint.json"
    descriptor = cast(dict[str, Any], json.loads(descriptor_path.read_text(encoding="utf-8")))
    if descriptor.get("schema_version") != LLAMACPP_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("llama.cpp checkpoint schema differs")
    if descriptor.get("artifact_format") != LLAMACPP_CHECKPOINT_FORMAT:
        raise ValueError("llama.cpp checkpoint format differs")
    packed_descriptor_hash = _hash_file(packed.root / "nanoquant-packed-model.json")
    if descriptor.get("source_packed_descriptor_sha256") != packed_descriptor_hash:
        raise ValueError("llama.cpp checkpoint is bound to a different packed descriptor")
    if descriptor.get("model") != {
        "source": packed.manifest.model.source,
        "revision": packed.manifest.model.revision,
        "family": packed.manifest.model.family,
        "config_hash": packed.manifest.model.config_hash,
        "tokenizer_hash": packed.manifest.model.tokenizer_hash,
    }:
        raise ValueError("llama.cpp checkpoint model metadata differs")

    shards = cast(list[dict[str, Any]], descriptor.get("shards"))
    for shard in shards:
        path = args.checkpoint / str(shard["path"])
        if path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"llama.cpp checkpoint shard size differs: {path.name}")
        if _hash_file(path) != str(shard["sha256"]):
            raise ValueError(f"llama.cpp checkpoint shard hash differs: {path.name}")

    reference = cast(dict[str, str], descriptor["reference"])
    converter = _load_converter(args.reference_root, reference["converter_sha256"])
    checkpoint_state = cast(dict[str, torch.Tensor], converter.load_checkpoint(args.checkpoint))
    sidecars = cast(dict[str, Any], converter.build_sidecars(checkpoint_state))

    expected_keys: set[str] = set()
    expected_prefixes: set[str] = set()
    array_count = 0
    element_count = 0
    for block in packed.manifest.blocks:
        for layer in block.layers:
            state = packed.load_layer(layer.spec.name)
            prefix = gemma_hf_checkpoint_prefix(layer.spec.name)
            expected_prefixes.add(prefix)
            expected_keys.update(llamacpp_checkpoint_tensors(state, prefix))
            try:
                sidecar = sidecars[prefix]
            except KeyError as error:
                raise ValueError(f"modified llama.cpp omitted sidecar group: {prefix}") from error
            comparisons = [
                ("nq_v", _numpy(state.right_words), sidecar.v_packed),
                ("nq_u", _numpy(state.left_words), sidecar.u_packed),
                ("scale_pre", _numpy(state.scale_pre.float()), sidecar.scale_pre),
                ("scale_mid", _numpy(state.scale_mid.float()), sidecar.scale_mid),
                ("scale_post", _numpy(state.scale_post.float()), sidecar.scale_post),
            ]
            if state.outlier_indices is not None and state.outlier_values is not None:
                expected_values = (
                    state.outlier_values
                    if state.outlier_values.dtype == torch.int8
                    else state.outlier_values.to(torch.float16)
                )
                comparisons.extend(
                    (
                        ("salient_idx", _numpy(state.outlier_indices.int()), sidecar.salient_idx),
                        ("salient_weight", _numpy(expected_values), sidecar.salient_weight),
                    )
                )
            if state.outlier_scales is not None:
                comparisons.append(
                    ("salient_scale", _numpy(state.outlier_scales.float()), sidecar.salient_scale)
                )
            for suffix, expected, actual in comparisons:
                if actual is None:
                    raise ValueError(f"modified llama.cpp omitted sidecar tensor: {prefix}.{suffix}")
                element_count += _equal(f"{prefix}.{suffix}", expected, actual)
                array_count += 1

    if set(checkpoint_state) != expected_keys:
        missing = sorted(expected_keys - set(checkpoint_state))
        extra = sorted(set(checkpoint_state) - expected_keys)
        raise ValueError(f"llama.cpp checkpoint tensor inventory differs: missing={missing}, extra={extra}")
    if set(sidecars) != expected_prefixes:
        missing = sorted(expected_prefixes - set(sidecars))
        extra = sorted(set(sidecars) - expected_prefixes)
        raise ValueError(f"modified llama.cpp sidecar inventory differs: missing={missing}, extra={extra}")

    print(
        json.dumps(
            {
                "packed_artifact": str(packed.root),
                "checkpoint": str(args.checkpoint.resolve()),
                "reference_root": str(args.reference_root.resolve()),
                "reference_converter_sha256": reference["converter_sha256"],
                "block_count": len(shards),
                "layer_count": len(sidecars),
                "checkpoint_tensor_count": len(checkpoint_state),
                "converted_array_count": array_count,
                "converted_element_count": element_count,
                "exact": True,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
