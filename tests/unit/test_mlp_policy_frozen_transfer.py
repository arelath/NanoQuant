from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from nanoquant.application.layers import FrozenReferenceLinear
from nanoquant.infrastructure.io_utils import hash_file
from nanoquant.infrastructure.safetensors_io import SAFETENSORS


def _probe_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("probe_mlp_policy_frozen_transfer")


def _write_overlay(root: Path) -> None:
    root.mkdir()
    tensors = {
        "model.layers.0.mlp.gate_proj.weight": torch.tensor(
            [[1.0, 2.0]],
            dtype=torch.bfloat16,
        ),
        "model.layers.0.mlp.up_proj.weight": torch.tensor(
            [[3.0, 4.0]],
            dtype=torch.bfloat16,
        ),
        "model.layers.0.mlp.down_proj.weight": torch.tensor(
            [[5.0]],
            dtype=torch.bfloat16,
        ),
    }
    tensor_path = root / "weights.safetensors"
    SAFETENSORS.save(tensors, tensor_path)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "arm": "hybrid",
                "layer_count": 3,
                "blocks": [0],
                "tensor_sha256": hash_file(tensor_path),
                "tensors": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": "bfloat16",
                    }
                    for name, value in tensors.items()
                },
            }
        ),
        encoding="utf-8",
    )


def test_transfer_probe_loads_strict_overlay(tmp_path: Path) -> None:
    probe = _probe_module()
    overlay = tmp_path / "overlay"
    _write_overlay(overlay)

    tensors, manifest = probe._load_overlay(overlay)

    assert manifest["arm"] == "hybrid"
    assert len(tensors) == 3
    manifest_path = overlay / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["tensor_sha256"] = "wrong"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity or hash"):
        probe._load_overlay(overlay)


def test_transfer_probe_installs_dense_linear() -> None:
    probe = _probe_module()

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.gate_proj = FrozenReferenceLinear(
                torch.ones((1, 1)),
                torch.ones((1, 2)),
                torch.ones(2),
                torch.ones(1),
                torch.ones(1),
                torch.tensor([0.25]),
            )

    block = Block()
    weight = torch.tensor([[2.0, 3.0]])
    probe._install_dense_linear(
        block,
        "mlp.gate_proj",
        weight,
        device="cpu",
    )

    assert isinstance(block.mlp.gate_proj, nn.Linear)
    assert torch.equal(block.mlp.gate_proj.weight, weight)
    assert torch.equal(
        block.mlp.gate_proj.bias,
        torch.tensor([0.25]),
    )
