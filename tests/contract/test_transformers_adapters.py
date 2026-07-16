import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.domain.models import BlockId, CheckpointInventory, SourceTensor
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
    GemmaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    ),
    Gemma2Config(
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

MODEL_FACTORIES = {
    "llama": LlamaForCausalLM,
    "gemma": GemmaForCausalLM,
    "gemma2": Gemma2ForCausalLM,
    "gemma3_text": Gemma3ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
    "opt": OPTForCausalLM,
}


class TrackingSource:
    def __init__(self, source: SafetensorsModelSource) -> None:
        self.source = source.source
        self.revision = source.revision
        self._source = source
        self.active_reads = 0
        self.maximum_active_reads = 0
        self.read_keys: list[str] = []

    def inventory(self) -> CheckpointInventory:
        return self._source.inventory()

    @contextmanager
    def read_tensor(self, tensor: SourceTensor, device: str = "cpu") -> Iterator[object]:
        self.active_reads += 1
        self.maximum_active_reads = max(self.maximum_active_reads, self.active_reads)
        self.read_keys.append(tensor.source_key)
        try:
            with self._source.read_tensor(tensor, device) as value:
                yield value
        finally:
            self.active_reads -= 1


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


def test_gemma3_multimodal_wrapper_maps_only_language_model_tensors(tmp_path: Path) -> None:
    text_config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    values = {
        "model_type": "gemma3",
        "torch_dtype": "bfloat16",
        "text_config": text_config.to_dict(),
    }
    adapter = adapter_for_config(values)
    block = adapter.definition.block_factory(text_config, 0)
    prefix = adapter.definition.block_prefix.format(index=0)
    state = {
        f"{prefix}.{key}": value.detach().clone().contiguous()
        for key, value in block.state_dict().items()
    }
    state["language_model.model.embed_tokens.weight"] = torch.zeros(32, 16)
    state["vision_tower.probe.weight"] = torch.zeros(2, 2)
    snapshot = tmp_path / "gemma3-wrapper"
    snapshot.mkdir()
    save_file(state, snapshot / "model.safetensors")
    (snapshot / "config.json").write_text(json.dumps(values), encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    source = SafetensorsModelSource(snapshot, source="fixture/gemma3-wrapper", revision="abc", verify_hashes=False)

    assert adapter.decoder_block_count(source) == 1
    inventory = adapter.model_inventory(source)
    assert all(tensor.source_key.startswith("language_model.") for tensor in inventory.shared_tensors)
    assert not any(tensor.source_key.startswith("vision_tower.") for tensor in inventory.shared_tensors)
    loaded = adapter.load_block(source, BlockId(0), "cpu")
    assert isinstance(loaded, torch.nn.Module)
    assert next(loaded.parameters()).dtype is torch.bfloat16


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


@pytest.mark.parametrize(
    ("model_type", "softcap", "expected_version"),
    (
        ("llama", None, "2"),
        ("qwen3", None, "2"),
        ("opt", None, "3"),
        ("gemma", None, "3"),
        ("gemma2", None, "3"),
        ("gemma3_text", None, "2"),
        ("gemma3_text", 30.0, "3"),
    ),
)
def test_adapter_identity_versions_only_changed_replay_contracts(
    tmp_path: Path,
    model_type: str,
    softcap: float | None,
    expected_version: str,
) -> None:
    base_config = next(config for config in CONFIGS if config.model_type == model_type)
    values = base_config.to_dict()
    if model_type == "gemma3_text":
        values["final_logit_softcapping"] = softcap
    adapter = adapter_for_config(values)
    source, _state = _source(tmp_path, adapter.definition.config_factory(values))

    assert adapter.model_inventory(source).model.adapter.version == expected_version


@pytest.mark.parametrize("base_config", CONFIGS, ids=lambda config: config.model_type)
def test_adapter_full_replay_tied_inventory_and_streamed_loading(tmp_path: Path, base_config: object) -> None:
    values = base_config.to_dict()  # type: ignore[attr-defined]
    values.update(num_hidden_layers=2, tie_word_embeddings=True, use_cache=False)
    adapter = adapter_for_config(values)
    config = adapter.definition.config_factory(values)
    model = MODEL_FACTORIES[values["model_type"]](config).eval()
    tokens = torch.tensor(((1, 2, 3, 4), (4, 3, 2, 1)))
    layers = adapter.get_decoder_layers(model)

    first_capture = capture_prefix_invocations(
        layers[0],
        (lambda: adapter.run_decoder_forward(model, tokens),),
    )[0]
    second_capture = capture_prefix_invocations(
        layers[1],
        (lambda: adapter.run_decoder_forward(model, tokens),),
    )[0]
    with torch.no_grad():
        reference_logits = adapter.run_full_forward(model, tokens)
        hidden = adapter.run_prefix(model, tokens)
        torch.testing.assert_close(hidden, first_capture.positional[0], rtol=0, atol=0)
        hidden = adapter.run_block(layers[0], hidden, **first_capture.keyword)
        torch.testing.assert_close(hidden, second_capture.positional[0], rtol=0, atol=0)
        hidden = adapter.run_block(layers[1], hidden, **second_capture.keyword)
        replay_logits = adapter.run_suffix(model, hidden)
    torch.testing.assert_close(replay_logits, reference_logits, rtol=0, atol=0)
    assert adapter.lm_head(model).weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()

    snapshot = tmp_path / str(values["model_type"])
    model.save_pretrained(snapshot, safe_serialization=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    source = SafetensorsModelSource(
        snapshot,
        source=f"fixture/{values['model_type']}",
        revision="full-contract",
        verify_hashes=False,
    )
    inventory = adapter.model_inventory(source)
    mapped_keys = [tensor.source_key for block in inventory.blocks for tensor in block.source_tensors] + [
        tensor.source_key for tensor in inventory.shared_tensors
    ]
    checkpoint_keys = [tensor.key for tensor in source.inventory().tensors]
    assert len(inventory.blocks) == 2
    assert [block.block.index for block in inventory.blocks] == [0, 1]
    assert len(mapped_keys) == len(set(mapped_keys))
    assert set(mapped_keys) == set(checkpoint_keys)

    tracking = TrackingSource(source)
    loaded = adapter.load_block(tracking, BlockId(1), "cpu")  # type: ignore[arg-type]
    assert tracking.active_reads == 0
    assert tracking.maximum_active_reads == 1
    assert set(tracking.read_keys) == {tensor.source_key for tensor in inventory.blocks[1].source_tensors}
    expected = layers[1].state_dict()
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, expected[key])
