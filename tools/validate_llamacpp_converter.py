"""Validate rewrite checkpoint sidecars through the pinned modified llama.cpp converter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from nanoquant.runtime import open_packed_artifact
from nanoquant.runtime.llamacpp import gemma_hf_checkpoint_prefix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_converter(llama_root: Path, expected_sha256: str) -> ModuleType:
    converter = llama_root / "convert_nanoquant_to_gguf.py"
    actual = _sha256(converter)
    if actual != expected_sha256:
        raise ValueError(
            f"modified llama.cpp converter hash differs: {actual} != {expected_sha256}"
        )
    sys.path.insert(0, str(llama_root))
    spec = importlib.util.spec_from_file_location("nanoquant_pinned_llamacpp_converter", converter)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load modified llama.cpp converter: {converter}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _exact(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError(
            f"{label} dtype/shape differs: {actual.dtype} {actual.shape} != "
            f"{expected.dtype} {expected.shape}"
        )
    if not np.array_equal(actual, expected):
        raise ValueError(f"{label} values differ")


class _CaptureWriter:
    def __init__(self) -> None:
        self.tensors: dict[str, np.ndarray] = {}

    def add_tensor(self, name: str, value: np.ndarray) -> None:
        if name in self.tensors:
            raise ValueError(f"modified llama.cpp writer emitted a duplicate tensor: {name}")
        self.tensors[name] = value


class _CaptureModel:
    undo_permute = False

    def __init__(self) -> None:
        self.gguf_writer = _CaptureWriter()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    packed = open_packed_artifact(args.packed_artifact, verify_hashes=True)
    reference = packed.manifest.layout.reference
    converter = _load_converter(args.llama_root.resolve(), reference.converter_sha256)
    hparams = converter.ModelBase.load_hparams(args.model.resolve(), is_mistral_format=False)
    architecture = converter.get_model_architecture(hparams, converter.ModelType.TEXT)
    model_class = converter.get_model_class(architecture, mmproj=False)
    if bool(getattr(model_class, "undo_permute", False)):
        raise ValueError("reference converter model requires an unimplemented output-row permutation")
    if getattr(model_class, "model_arch", None) != converter.gguf.MODEL_ARCH.GEMMA3:
        raise ValueError(f"reference converter selected an unexpected model architecture: {architecture}")

    checkpoint_state = converter.load_checkpoint(args.checkpoint.resolve())
    sidecars = converter.build_sidecars(checkpoint_state)
    name_map = converter.gguf.get_tensor_name_map(
        converter.gguf.MODEL_ARCH.GEMMA3,
        len(packed.manifest.blocks),
    )
    expected_prefixes: set[str] = set()
    gguf_tensor_count = 0
    normalized_salient_elements = 0
    maximum_salient_normalization_error = 0.0
    for block in packed.manifest.blocks:
        for entry in block.layers:
            state = packed.load_layer(entry.spec.name)
            prefix = gemma_hf_checkpoint_prefix(entry.spec.name)
            expected_prefixes.add(prefix)
            sidecar = sidecars.get(prefix)
            if sidecar is None:
                raise ValueError(f"modified llama.cpp converter omitted sidecar: {prefix}")

            _exact(f"{prefix}.nq_v", sidecar.v_packed, state.right_words.numpy())
            _exact(f"{prefix}.nq_u", sidecar.u_packed, state.left_words.numpy())
            _exact(
                f"{prefix}.nq_scale_pre",
                sidecar.scale_pre,
                state.scale_pre.float().numpy(),
            )
            _exact(
                f"{prefix}.nq_scale_mid",
                sidecar.scale_mid,
                state.scale_mid.float().numpy(),
            )
            _exact(
                f"{prefix}.nq_scale_post",
                sidecar.scale_post,
                state.scale_post.float().numpy(),
            )
            if state.outlier_indices is None or state.outlier_values is None:
                if sidecar.salient_idx is not None or sidecar.salient_weight is not None:
                    raise ValueError(f"modified llama.cpp converter added salient tensors: {prefix}")
            else:
                assert sidecar.salient_idx is not None
                assert sidecar.salient_weight is not None
                _exact(
                    f"{prefix}.nq_salient_idx",
                    sidecar.salient_idx,
                    state.outlier_indices.numpy(),
                )
                if state.outlier_values.dtype == torch.int8:
                    expected_weight = state.outlier_values.numpy()
                else:
                    expected_weight = state.outlier_values.to(torch.float16).numpy()
                    source = state.outlier_values.float()
                    normalized = state.outlier_values.to(torch.float16).float()
                    normalized_salient_elements += int(torch.count_nonzero(source != normalized))
                    maximum_salient_normalization_error = max(
                        maximum_salient_normalization_error,
                        float(torch.max(torch.abs(source - normalized))),
                    )
                _exact(
                    f"{prefix}.nq_salient_weight",
                    sidecar.salient_weight,
                    expected_weight,
                )
                if state.outlier_scales is None:
                    if sidecar.salient_scale is not None:
                        raise ValueError(f"modified llama.cpp converter added salient scales: {prefix}")
                else:
                    assert sidecar.salient_scale is not None
                    _exact(
                        f"{prefix}.nq_salient_scale",
                        sidecar.salient_scale,
                        state.outlier_scales.float().numpy(),
                    )

            mapped_base = name_map.get_name(prefix, try_suffixes=(".weight",))
            if mapped_base is None:
                raise ValueError(f"modified llama.cpp GGUF mapper rejected layer: {prefix}")
            capture = _CaptureModel()
            converter.write_sidecar(capture, mapped_base, f"{prefix}.weight", sidecar)
            expected_names = {
                f"{mapped_base}.nq_v",
                f"{mapped_base}.nq_u",
                f"{mapped_base}.nq_scale_pre",
                f"{mapped_base}.nq_scale_mid",
                f"{mapped_base}.nq_scale_post",
            }
            if sidecar.salient_idx is not None:
                expected_names.update(
                    (
                        f"{mapped_base}.nq_salient_idx",
                        f"{mapped_base}.nq_salient_weight",
                    )
                )
            if sidecar.salient_scale is not None:
                expected_names.add(f"{mapped_base}.nq_salient_scale")
            if set(capture.gguf_writer.tensors) != expected_names:
                raise ValueError(f"modified llama.cpp GGUF tensor names differ: {prefix}")
            gguf_tensor_count += len(expected_names)

    if set(sidecars) != expected_prefixes:
        raise ValueError("modified llama.cpp converter sidecar inventory differs")
    print(
        json.dumps(
            {
                "packed_artifact": str(packed.root),
                "checkpoint": str(args.checkpoint.resolve()),
                "llama_root": str(args.llama_root.resolve()),
                "model": str(args.model.resolve()),
                "reference_commit": reference.commit,
                "reference_dirty_diff_git_object": reference.dirty_diff_git_object,
                "reference_converter_sha256": reference.converter_sha256,
                "architecture": str(architecture),
                "block_count": len(packed.manifest.blocks),
                "layer_count": packed.manifest.layer_count,
                "checkpoint_tensor_count": len(checkpoint_state),
                "sidecar_group_count": len(sidecars),
                "gguf_tensor_count": gguf_tensor_count,
                "sign_words_exact": True,
                "scale_values_exact_after_f32_normalization": True,
                "salient_values_exact_after_f16_normalization": True,
                "normalized_salient_element_count": normalized_salient_elements,
                "maximum_salient_normalization_error": maximum_salient_normalization_error,
                "passed": True,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
