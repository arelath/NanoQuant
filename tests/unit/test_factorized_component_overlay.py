from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.config.codec import to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.factorized_component_overlay import (
    apply_factorized_component_overlay,
    load_factorized_component_overlay,
)
from nanoquant.infrastructure.io_utils import hash_file
from nanoquant.infrastructure.safetensors_io import SAFETENSORS


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.gate_proj = FactorizedReferenceLinear(
            torch.tensor([[1.0], [-1.0]]),
            torch.tensor([[1.0, -1.0, 1.0]]),
            torch.ones(3),
            torch.ones(1),
            torch.ones(2),
        )
        self.model.layers = nn.ModuleList([block])


def _write_overlay(root: Path, reference: ArtifactRef) -> None:
    tensors = {
        "model.layers.0.mlp.gate_proj.scale_pre": torch.tensor([2.0, 3.0, 4.0]),
        "model.layers.0.mlp.gate_proj.scale_post": torch.tensor([5.0, 6.0]),
    }
    root.mkdir()
    tensor_path = root / "components.safetensors"
    SAFETENSORS.save(tensors, tensor_path)
    byte_count = sum(value.numel() * value.element_size() for value in tensors.values())
    manifest = {
        "schema_version": 2,
        "semantics": "replace-existing-factorized-components",
        "source_dense_tensor_sha256": "dense",
        "frozen_identity": {"model_hash": "m", "config_hash": "c", "plan_hash": "p"},
        "global_tuning": to_dict(reference),
        "policy": {"0": "joint"},
        "tensor_sha256": hash_file(tensor_path),
        "tensor_count": len(tensors),
        "replaced_payload_bytes": byte_count,
        "replacement_payload_bytes": byte_count,
        "payload_byte_delta": 0,
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
            for name, value in tensors.items()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_component_overlay_validates_identity_and_replaces_existing_terms(tmp_path: Path) -> None:
    reference = ArtifactRef("global-tuning-result", "sha256-global", 1)
    root = tmp_path / "overlay"
    _write_overlay(root, reference)
    overlay = load_factorized_component_overlay(
        root,
        frozen_identity={"model_hash": "m", "config_hash": "c", "plan_hash": "p"},
        global_tuning=reference,
    )
    model = _Model()

    applied = apply_factorized_component_overlay(model, overlay)

    module = model.model.layers[0].mlp.gate_proj
    assert isinstance(module, FactorizedReferenceLinear)
    torch.testing.assert_close(module.scale_pre, torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(module.scale_post, torch.tensor([5.0, 6.0]))
    assert applied.replaced_bytes == applied.replacement_bytes == 20
    assert applied.tensor_count == 2
    assert applied.layer_count == 1


def test_component_overlay_rejects_another_global_tuning_result(tmp_path: Path) -> None:
    reference = ArtifactRef("global-tuning-result", "sha256-global", 1)
    root = tmp_path / "overlay"
    _write_overlay(root, reference)

    with pytest.raises(ValueError, match="identity or replacement contract"):
        load_factorized_component_overlay(
            root,
            frozen_identity={"model_hash": "m", "config_hash": "c", "plan_hash": "p"},
            global_tuning=ArtifactRef("global-tuning-result", "different", 1),
        )
