from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from nanoquant.runtime import (
    GenerationRequest,
    PackedLayerState,
    PackedReferenceBackend,
    QuantizedLinearSpec,
    RuntimeBundleError,
    RuntimeModelMetadata,
    TransformersGenerationModel,
    batch_prompts,
    generate,
    hybrid_cache_factory,
    load_transformers_runtime,
    open_runtime_bundle,
    pack_sign_matrix,
    write_packed_artifact,
    write_runtime_bundle,
)


def _config() -> Gemma3TextConfig:
    return Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        sliding_window_pattern=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
    )


def _packed_state(name: str, layer: nn.Linear) -> PackedLayerState:
    rank = 4
    generator = torch.Generator().manual_seed(sum(name.encode("utf-8")))
    left = torch.randint(
        0,
        2,
        (layer.out_features, rank),
        generator=generator,
        dtype=torch.int64,
    ).mul_(2).sub_(1)
    right = torch.randint(
        0,
        2,
        (rank, layer.in_features),
        generator=generator,
        dtype=torch.int64,
    ).mul_(2).sub_(1)
    spec = QuantizedLinearSpec(
        name,
        "nanoquant-v1",
        layer.in_features,
        layer.out_features,
        rank,
        "float32",
        "float32",
        has_bias=layer.bias is not None,
    )
    return PackedLayerState(
        spec,
        "llama.cpp-i32-lsb-v1",
        pack_sign_matrix(left.contiguous()),
        pack_sign_matrix(right.contiguous()),
        torch.full((layer.in_features,), 0.05),
        torch.full((rank,), 0.05),
        torch.full((layer.out_features,), 0.05),
        None if layer.bias is None else torch.zeros(layer.out_features),
    )


@pytest.fixture()
def runtime_bundle(tmp_path: Path) -> Path:
    torch.manual_seed(20260715)
    model = Gemma3ForCausalLM(_config()).eval()
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    (source / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (source / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": {"x": 0}, "unk_token": "x"},
            }
        ),
        encoding="utf-8",
    )
    states = []
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear) and path.startswith("model.layers.0."):
            name = "blocks.0." + path.removeprefix("model.layers.0.")
            states.append(_packed_state(name, module))
    assert len(states) == 7
    metadata = RuntimeModelMetadata("fixture/gemma", "revision", "gemma3", "config", "tokenizer")
    packed = write_packed_artifact(
        tmp_path / "packed",
        metadata,
        "0" * 64,
        {0: states},
    )
    bundle = write_runtime_bundle(tmp_path / "bundle", packed.root, source)
    return bundle.root


def test_runtime_bundle_loads_shell_and_generates_without_source_checkpoint(
    runtime_bundle: Path,
) -> None:
    opened = open_runtime_bundle(runtime_bundle)
    assert opened.manifest.model.source == "fixture/gemma"
    assert len(opened.manifest.excluded_linear_modules) == 7
    assert not any(
        tensor.name.endswith(
            (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            )
        )
        for tensor in opened.manifest.shell_tensors
    )

    loaded = load_transformers_runtime(
        opened,
        PackedReferenceBackend(),
        device="cpu",
        input_dtype="float32",
        batch_size=1,
        prefill_tokens=3,
    )
    tokens, mask = batch_prompts(((2, 4, 5),), pad_token_id=0)
    result = generate(
        GenerationRequest(tokens, mask, 2, (1000,), 0),
        TransformersGenerationModel(
            loaded.model,
            hybrid_cache_factory(loaded.model.config),
        ),
    )
    control = load_transformers_runtime(
        opened,
        PackedReferenceBackend(),
        device="cpu",
        input_dtype="float32",
        batch_size=1,
        prefill_tokens=3,
        fuse_rms_norm=False,
    )
    control_result = generate(
        GenerationRequest(tokens, mask, 2, (1000,), 0),
        TransformersGenerationModel(
            control.model,
            hybrid_cache_factory(control.model.config),
        ),
    )

    assert loaded.replaced_linear_count == 7
    assert loaded.fused_rms_norm_count == 7
    assert loaded.fused_decode_rope_count == 0
    assert loaded.short_sliding_mask_count == 1
    assert control.fused_rms_norm_count == 0
    assert control.fused_decode_rope_count == 0
    assert control.short_sliding_mask_count == 1
    assert torch.equal(result.token_ids, control_result.token_ids)
    assert result.lengths == control_result.lengths
    assert result.token_ids.shape == (1, 2)
    assert result.lengths == (2,)
    assert not any(parameter.is_meta for parameter in loaded.model.parameters())
    assert loaded.model.model.embed_tokens.embed_scale.item() == pytest.approx(32**0.5)
    assert bool(torch.all(loaded.model.model.rotary_emb.inv_freq > 0))
    assert bool(torch.all(loaded.model.model.rotary_emb_local.inv_freq > 0))


def test_runtime_bundle_rejects_member_corruption(runtime_bundle: Path) -> None:
    asset = runtime_bundle / "model" / "tokenizer_config.json"
    asset.write_text('{"corrupt": true}\n', encoding="utf-8")

    with pytest.raises(RuntimeBundleError, match="member (size|hash) differs"):
        open_runtime_bundle(runtime_bundle)


def test_runtime_import_does_not_load_research_packages(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import sys\n"
        "import nanoquant.runtime\n"
        "forbidden = ('nanoquant.config', 'nanoquant.domain', 'nanoquant.application', "
        "'nanoquant.infrastructure')\n"
        "loaded = sorted(name for name in sys.modules if name.startswith(forbidden))\n"
        "assert loaded == [], loaded\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
