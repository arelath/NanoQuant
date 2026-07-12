import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.domain.models import BlockId
from nanoquant.infrastructure.model_adapters import UnsupportedModelVariant, adapter_for_config
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource

CONFIGS = (
    LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    ),
    Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    ),
    Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    ),
    OPTConfig(
        vocab_size=32, hidden_size=16, ffn_dim=32, num_hidden_layers=1, num_attention_heads=4, word_embed_proj_dim=16
    ),
)


def _source(tmp_path: Path, config: object) -> tuple[SafetensorsModelSource, dict[str, torch.Tensor]]:
    values = config.to_dict()  # type: ignore[attr-defined]
    adapter = adapter_for_config(values)
    definition = adapter.definition
    block = definition.block_factory(config, 0)
    prefix = definition.block_prefix.format(index=0)
    state = {f"{prefix}.{key}": value.detach().clone().contiguous() for key, value in block.state_dict().items()}
    snapshot = tmp_path / values["model_type"]
    snapshot.mkdir()
    save_file(state, snapshot / "model.safetensors")
    (snapshot / "config.json").write_text(json.dumps(values), encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    return SafetensorsModelSource(
        snapshot, source=f"fixture/{values['model_type']}", revision="abc", verify_hashes=False
    ), state


@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.model_type)
def test_adapter_contract_inventory_mapping_loading_and_order(tmp_path: Path, config: object) -> None:
    source, expected_state = _source(tmp_path, config)
    adapter = adapter_for_config(source.inventory().config)
    assert adapter.attention_implementation == ("eager" if config.model_type.startswith("gemma") else "sdpa")
    assert adapter.decoder_block_count(source) == 1
    inventory = adapter.model_inventory(source)
    assert len(inventory.blocks) == 1
    block_inventory = inventory.blocks[0]
    assert len(block_inventory.quantizable_layers) in {6, 7}
    loaded = adapter.load_block(source, BlockId(0), "cpu")
    layers = adapter.quantizable_layers(loaded, BlockId(0))
    assert tuple(layer.path for layer in layers) == adapter.definition.layer_paths
    prefix = adapter.definition.block_prefix.format(index=0) + "."
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, expected_state[prefix + key])


def test_unsupported_variant_is_explicit() -> None:
    with pytest.raises(UnsupportedModelVariant, match="SRC001.*unsupported"):
        adapter_for_config({"model_type": "mixtral"})


def test_gemma3_text_stack_capture_preserves_position_and_attention_metadata() -> None:
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    model = Gemma3ForCausalLM(config)
    tokens = torch.tensor([[1, 2, 3]])
    capture = capture_prefix_invocations(model.model.layers[0], (lambda: model(input_ids=tokens, use_cache=False),))[0]
    assert isinstance(capture.positional[0], torch.Tensor)
    assert capture.positional[0].shape == (1, 3, 16)
    assert {"position_embeddings_global", "position_embeddings_local", "attention_mask", "position_ids"} <= set(
        capture.keyword
    )
