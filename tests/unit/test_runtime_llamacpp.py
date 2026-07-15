from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from nanoquant.runtime import (
    LogicalLayerState,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    convert_logical_to_packed,
    export_llamacpp_checkpoint,
    gemma_gguf_tensor_prefix,
    gemma_hf_checkpoint_prefix,
    llamacpp_checkpoint_tensors,
    write_logical_artifact,
)


def _state(name: str, *, outliers: bool, bias: bool = False) -> LogicalLayerState:
    spec = QuantizedLinearSpec(
        name,
        "nanoquant-v1",
        35,
        3,
        33,
        "float32",
        "float32",
        outlier_count=2 if outliers else 0,
        outlier_value_dtype="float16" if outliers else None,
        has_bias=bias,
    )
    return LogicalLayerState(
        spec,
        torch.where(
            torch.arange(99).reshape(3, 33) % 2 == 0,
            torch.ones(3, 33),
            -torch.ones(3, 33),
        ),
        torch.where(
            torch.arange(33 * 35).reshape(33, 35) % 3 == 0,
            -torch.ones(33, 35),
            torch.ones(33, 35),
        ),
        (
            torch.cat((torch.tensor([0.5, 0.0]), torch.linspace(0.5, 1.5, 32), torch.zeros(1)))
            if outliers
            else torch.linspace(0.5, 1.5, 35)
        ),
        torch.linspace(0.75, 1.25, 33),
        torch.linspace(1.0, 1.5, 3),
        bias=torch.zeros(3) if bias else None,
        outlier_indices=torch.tensor([1, 34], dtype=torch.int32) if outliers else None,
        outlier_values=(
            torch.tensor([[1, -2], [3, -4], [5, -6]], dtype=torch.float16)
            if outliers
            else None
        ),
    )


def _export(tmp_path: Path):  # type: ignore[no-untyped-def]
    first = _state("blocks.0.self_attn.q_proj", outliers=True)
    second = _state("blocks.1.mlp.down_proj", outliers=False)
    logical = write_logical_artifact(
        tmp_path / "logical",
        RuntimeModelMetadata("fixture/model", "revision", "gemma3", "config", "tokenizer"),
        {0: (first,), 1: (second,)},
    )
    packed = convert_logical_to_packed(logical.root, tmp_path / "packed")
    manifest = export_llamacpp_checkpoint(packed.root, tmp_path / "checkpoint")
    return packed, manifest, tmp_path / "checkpoint"


def test_gemma_checkpoint_prefix_maps_only_supported_layers() -> None:
    assert (
        gemma_hf_checkpoint_prefix("blocks.12.self_attn.q_proj")
        == "model.layers.12.self_attn.q_proj"
    )
    assert gemma_gguf_tensor_prefix("blocks.12.self_attn.q_proj") == "blk.12.attn_q"
    assert gemma_gguf_tensor_prefix("blocks.3.mlp.down_proj") == "blk.3.ffn_down"
    with pytest.raises(ValueError, match="unsupported Gemma runtime layer name"):
        gemma_hf_checkpoint_prefix("blocks.0.self_attn.q_norm")


def test_llamacpp_checkpoint_tensors_match_pinned_converter_contract(tmp_path: Path) -> None:
    packed, _manifest, _root = _export(tmp_path)
    state = packed.load_layer("blocks.0.self_attn.q_proj")

    tensors = llamacpp_checkpoint_tensors(state, "model.layers.0.self_attn.q_proj")

    assert set(tensors) == {
        "model.layers.0.self_attn.q_proj.V_packed",
        "model.layers.0.self_attn.q_proj.V_shape",
        "model.layers.0.self_attn.q_proj.U_packed",
        "model.layers.0.self_attn.q_proj.U_shape",
        "model.layers.0.self_attn.q_proj.scale_pre",
        "model.layers.0.self_attn.q_proj.scale_mid",
        "model.layers.0.self_attn.q_proj.scale_post",
        "model.layers.0.self_attn.q_proj.salient_idx",
        "model.layers.0.self_attn.q_proj.salient_weight",
    }
    assert torch.equal(tensors["model.layers.0.self_attn.q_proj.V_packed"], state.right_words)
    assert torch.equal(tensors["model.layers.0.self_attn.q_proj.U_packed"], state.left_words)
    assert tensors["model.layers.0.self_attn.q_proj.V_shape"].tolist() == [33, 35]
    assert tensors["model.layers.0.self_attn.q_proj.U_shape"].tolist() == [3, 33]


def test_llamacpp_checkpoint_tensors_reject_unmapped_bias(tmp_path: Path) -> None:
    logical = write_logical_artifact(
        tmp_path / "logical",
        RuntimeModelMetadata("fixture/model", "revision", "gemma3", "config", "tokenizer"),
        {0: (_state("blocks.0.self_attn.q_proj", outliers=False, bias=True),)},
    )
    packed = convert_logical_to_packed(logical.root, tmp_path / "packed")
    state = packed.load_layer("blocks.0.self_attn.q_proj")

    with pytest.raises(ValueError, match="do not carry bias"):
        llamacpp_checkpoint_tensors(
            state,
            "model.layers.0.self_attn.q_proj",
        )


def test_llamacpp_checkpoint_export_is_block_sharded_and_source_bound(tmp_path: Path) -> None:
    packed, manifest, root = _export(tmp_path)

    packed_hash = hashlib.sha256(
        (packed.root / "nanoquant-packed-model.json").read_bytes()
    ).hexdigest()
    assert manifest.source_packed_descriptor_sha256 == packed_hash
    assert manifest.layer_count == 2
    assert manifest.tensor_count == 16
    assert [shard.path for shard in manifest.shards] == [
        "block-00000.safetensors",
        "block-00001.safetensors",
    ]
    descriptor = json.loads((root / "nanoquant-llamacpp-checkpoint.json").read_text())
    assert descriptor["reference"]["converter_sha256"] == (
        "92b0d31c1ce83d0fe3668bbb20cee6a4da24ec3e9476f6699890d01540241e4d"
    )
    with safe_open(root / manifest.shards[0].path, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {
            "model.layers.0.self_attn.q_proj.V_packed",
            "model.layers.0.self_attn.q_proj.V_shape",
            "model.layers.0.self_attn.q_proj.U_packed",
            "model.layers.0.self_attn.q_proj.U_shape",
            "model.layers.0.self_attn.q_proj.scale_pre",
            "model.layers.0.self_attn.q_proj.scale_mid",
            "model.layers.0.self_attn.q_proj.scale_post",
            "model.layers.0.self_attn.q_proj.salient_idx",
            "model.layers.0.self_attn.q_proj.salient_weight",
        }


def test_llamacpp_checkpoint_export_refuses_overwrite(tmp_path: Path) -> None:
    packed, _manifest, root = _export(tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        export_llamacpp_checkpoint(packed.root, root)
    assert not list(tmp_path.glob(".nanoquant-llamacpp-*"))


def test_llamacpp_checkpoint_export_rejects_unmapped_model_family(tmp_path: Path) -> None:
    state = _state("blocks.0.self_attn.q_proj", outliers=False)
    logical = write_logical_artifact(
        tmp_path / "logical",
        RuntimeModelMetadata("fixture/model", "revision", "llama", "config", "tokenizer"),
        {0: (state,)},
    )
    packed = convert_logical_to_packed(logical.root, tmp_path / "packed")

    with pytest.raises(ValueError, match="does not support model family: llama"):
        export_llamacpp_checkpoint(packed.root, tmp_path / "checkpoint")
