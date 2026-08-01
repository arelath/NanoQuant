from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nanoquant.infrastructure.foldable_mlp_initializer import load_foldable_mlp_initializer
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.safetensors_io import SAFETENSORS


def _artifact(root: Path) -> str:
    root.mkdir()
    tensors = {
        "model.layers.0.mlp.gate_proj.output_log_multiplier": torch.tensor(
            [0.0, 0.25], dtype=torch.float32
        )
    }
    tensor_path = root / "multipliers.safetensors"
    SAFETENSORS.save(tensors, tensor_path)
    digest = hash_file(tensor_path)
    atomic_write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "semantics": "foldable-mlp-log-multiplier-initializer",
            "model": {"source": "model/source", "revision": "revision"},
            "tensor_sha256": digest,
            "tensor_count": 1,
            "tensors": {
                name: {"shape": list(value.shape), "dtype": "float32"}
                for name, value in tensors.items()
            },
        },
    )
    return digest


def test_initializer_loader_checks_hash_model_and_inventory(tmp_path: Path) -> None:
    root = tmp_path / "initializer"
    digest = _artifact(root)

    loaded = load_foldable_mlp_initializer(
        root,
        expected_sha256=digest,
        model_source="model/source",
        model_revision="revision",
    )

    assert loaded.tensor_sha256 == digest
    assert tuple(loaded.tensors) == (
        "model.layers.0.mlp.gate_proj.output_log_multiplier",
    )
    with pytest.raises(ValueError, match="identity or inventory"):
        load_foldable_mlp_initializer(
            root,
            expected_sha256="0" * 64,
            model_source="model/source",
            model_revision="revision",
        )
