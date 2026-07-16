"""Explicit architecture adapters for the retained Transformers families."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from nanoquant.domain.models import (
    BlockId,
    BlockInventory,
    CheckpointInventory,
    ComponentRef,
    LayerId,
    LayerInventory,
    ModelIdentity,
    ModelInventory,
    SourceTensor,
    TensorId,
)
from nanoquant.ports.model_source import ModelSource


class UnsupportedModelVariant(ValueError):
    code = "SRC001"


@dataclass(frozen=True, slots=True)
class AdapterDefinition:
    family: str
    model_types: tuple[str, ...]
    block_prefix: str
    layer_paths: tuple[str, ...]
    config_factory: Callable[[dict[str, object]], Any]
    block_factory: Callable[[Any, int], nn.Module]
    block_count_field: str = "num_hidden_layers"


def _config(cls: type[PretrainedConfig]) -> Callable[[dict[str, object]], PretrainedConfig]:
    return lambda values: cls.from_dict(values)


def _gemma3_wrapper_text_config(values: dict[str, object]) -> Gemma3TextConfig:
    text_config = values.get("text_config")
    if not isinstance(text_config, dict):
        raise UnsupportedModelVariant("SRC001 Gemma 3 wrapper contains no text_config object")
    return cast(Gemma3TextConfig, Gemma3TextConfig.from_dict(cast(dict[str, object], text_config)))


DEFINITIONS = (
    AdapterDefinition(
        "llama",
        ("llama",),
        "model.layers.{index}",
        (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        _config(LlamaConfig),
        lambda config, index: LlamaDecoderLayer(config, index),
    ),
    AdapterDefinition(
        "gemma",
        ("gemma",),
        "model.layers.{index}",
        (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        _config(GemmaConfig),
        lambda config, index: GemmaDecoderLayer(config, index),
    ),
    AdapterDefinition(
        "gemma",
        ("gemma2",),
        "model.layers.{index}",
        (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        _config(Gemma2Config),
        lambda config, index: Gemma2DecoderLayer(config, index),
    ),
    AdapterDefinition(
        "gemma",
        ("gemma3_text",),
        "model.layers.{index}",
        (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        _config(Gemma3TextConfig),
        lambda config, index: Gemma3DecoderLayer(config, index),
    ),
    AdapterDefinition(
        "gemma",
        ("gemma3",),
        "language_model.model.layers.{index}",
        (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        _gemma3_wrapper_text_config,
        lambda config, index: Gemma3DecoderLayer(config, index),
    ),
    AdapterDefinition(
        "qwen",
        ("qwen3",),
        "model.layers.{index}",
        (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        _config(Qwen3Config),
        lambda config, index: Qwen3DecoderLayer(config, index),
    ),
    AdapterDefinition(
        "opt",
        ("opt",),
        "model.decoder.layers.{index}",
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"),
        _config(OPTConfig),
        lambda config, index: OPTDecoderLayer(config, index),
    ),
)


class TransformersModelAdapter:
    def __init__(self, definition: AdapterDefinition) -> None:
        self.definition = definition
        self.family = definition.family

    def _contract_version(self, config: dict[str, object]) -> str:
        """Return the persisted contract version for this model variant.

        Adapter v3 completed replay behavior for OPT, Gemma/Gemma2, and
        Gemma3 checkpoints that use final-logit softcapping.  Llama, Qwen,
        and Gemma3 checkpoints without softcapping retained their exact v2
        prefix/block/suffix behavior, so their existing preprocessing remains
        compatible and must not be invalidated by an unrelated global bump.
        """
        model_type = config.get("model_type")
        if model_type == "gemma3":
            text_config = config.get("text_config")
            softcap = text_config.get("final_logit_softcapping") if isinstance(text_config, dict) else None
            return "3" if softcap is not None else "2"
        if model_type in {"opt", "gemma", "gemma2"}:
            return "3"
        if model_type == "gemma3_text" and config.get("final_logit_softcapping") is not None:
            return "3"
        return "2"

    def _checkpoint(self, source: ModelSource) -> CheckpointInventory:
        inventory = source.inventory()
        model_type = inventory.config.get("model_type")
        if model_type not in self.definition.model_types:
            raise UnsupportedModelVariant(
                f"SRC001 adapter {self.family!r} does not support model_type={model_type!r}; "
                f"expected one of {self.definition.model_types}"
            )
        return inventory

    @property
    def attention_implementation(self) -> str:
        """Return the attention backend used by the legacy model loader."""
        return "eager" if self.family == "gemma" else "sdpa"

    def decoder_block_count(self, source: ModelSource) -> int:
        config = self._checkpoint(source).config
        if config.get("model_type") == "gemma3":
            text_config = config.get("text_config")
            value = text_config.get(self.definition.block_count_field) if isinstance(text_config, dict) else None
        else:
            value = config.get(self.definition.block_count_field)
        if not isinstance(value, int) or value <= 0:
            raise UnsupportedModelVariant(f"SRC001 invalid {self.definition.block_count_field}: {value!r}")
        return value

    def _prefix(self, block_id: BlockId) -> str:
        return self.definition.block_prefix.format(index=block_id.index)

    def source_key(self, layer: LayerId, tensor_name: str = "weight") -> str:
        return f"{self._prefix(layer.block)}.{layer.path}.{tensor_name}"

    def block_inventory(self, source: ModelSource, block_id: BlockId) -> BlockInventory:
        checkpoint = self._checkpoint(source)
        prefix = self._prefix(block_id) + "."
        metadata = {item.key: item for item in checkpoint.tensors}
        source_tensors = tuple(
            SourceTensor(
                TensorId(None, item.key),
                item.key,
                item.shard,
                item.spec,
                f"{item.shard_hash or 'unverified'}#{item.key}",
            )
            for item in checkpoint.tensors
            if item.key.startswith(prefix)
        )
        layers = []
        for path in self.definition.layer_paths:
            layer_id = LayerId(block_id, path)
            weight_key = self.source_key(layer_id)
            if weight_key not in metadata:
                raise UnsupportedModelVariant(f"SRC001 checkpoint is missing required tensor {weight_key}")
            item = metadata[weight_key]
            weight = SourceTensor(
                TensorId(layer_id, "weight"),
                weight_key,
                item.shard,
                item.spec,
                f"{item.shard_hash or 'unverified'}#{weight_key}",
            )
            bias_item = metadata.get(self.source_key(layer_id, "bias"))
            bias = (
                None
                if bias_item is None
                else SourceTensor(
                    TensorId(layer_id, "bias"),
                    bias_item.key,
                    bias_item.shard,
                    bias_item.spec,
                    f"{bias_item.shard_hash or 'unverified'}#{bias_item.key}",
                )
            )
            if len(item.spec.shape) != 2:
                raise UnsupportedModelVariant(f"SRC001 quantizable weight is not a matrix: {weight_key}")
            layers.append(LayerInventory(layer_id, weight, bias, item.spec.shape[1], item.spec.shape[0]))
        return BlockInventory(block_id, source_tensors, tuple(layers))

    def model_inventory(self, source: ModelSource) -> ModelInventory:
        checkpoint = self._checkpoint(source)
        blocks = tuple(
            self.block_inventory(source, BlockId(index)) for index in range(self.decoder_block_count(source))
        )
        block_keys = {tensor.source_key for block in blocks for tensor in block.source_tensors}
        shared = tuple(
            SourceTensor(
                TensorId(None, item.key),
                item.key,
                item.shard,
                item.spec,
                f"{item.shard_hash or 'unverified'}#{item.key}",
            )
            for item in checkpoint.tensors
            if item.key not in block_keys
            and (checkpoint.config.get("model_type") != "gemma3" or item.key.startswith("language_model."))
        )
        config_hash = hashlib.sha256(json.dumps(checkpoint.config, sort_keys=True).encode()).hexdigest()
        identity = ModelIdentity(
            checkpoint.source,
            checkpoint.revision,
            f"sha256:{config_hash}",
            checkpoint.source,
            checkpoint.revision,
            ComponentRef(self.family, self._contract_version(checkpoint.config)),
        )
        return ModelInventory(1, identity, blocks, shared, checkpoint.total_shard_bytes)

    def construct_block(self, source: ModelSource, block_id: BlockId, device: str) -> nn.Module:
        checkpoint = self._checkpoint(source)
        config = self.definition.config_factory(checkpoint.config)
        config._attn_implementation = self.attention_implementation
        block = self.definition.block_factory(config, block_id.index)
        dtype = getattr(config, "torch_dtype", None)
        return block.to(device=device, dtype=dtype) if isinstance(dtype, torch.dtype) else block.to(device)

    def load_block(self, source: ModelSource, block_id: BlockId, device: str) -> nn.Module:
        block = self.construct_block(source, block_id, device)
        prefix = self._prefix(block_id) + "."
        state: dict[str, torch.Tensor] = {}
        for item in self.block_inventory(source, block_id).source_tensors:
            with source.read_tensor(item, device=device) as value:
                state[item.source_key.removeprefix(prefix)] = cast(torch.Tensor, value).clone()
        missing, unexpected = block.load_state_dict(state, strict=False)
        material_missing = [name for name in missing if not name.endswith("rotary_emb.inv_freq")]
        if material_missing or unexpected:
            raise UnsupportedModelVariant(
                f"SRC001 block state mismatch: missing={material_missing}, unexpected={unexpected}"
            )
        return block

    def quantizable_layers(self, block: nn.Module, block_id: BlockId) -> tuple[LayerId, ...]:
        modules = dict(block.named_modules())
        missing = [path for path in self.definition.layer_paths if not isinstance(modules.get(path), nn.Linear)]
        if missing:
            raise UnsupportedModelVariant(f"SRC001 block lacks expected linear modules: {missing}")
        return tuple(LayerId(block_id, path) for path in self.definition.layer_paths)

    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        base = getattr(model, "model", None)
        container = getattr(base, "decoder", None) if self.family == "opt" else base
        layers = getattr(container, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise UnsupportedModelVariant("SRC001 model has no mutable decoder layer stack")
        return layers

    def run_decoder_forward(self, model: nn.Module, tokens: torch.Tensor) -> object:
        base = getattr(model, "model", None)
        if not isinstance(base, nn.Module):
            raise UnsupportedModelVariant("SRC001 model has no decoder model")
        return cast(Any, base)(input_ids=tokens, use_cache=False)

    def run_full_forward(self, model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        output = cast(Any, model)(input_ids=tokens, use_cache=False)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise UnsupportedModelVariant("SRC001 causal model forward returned no logits")
        return logits

    def run_prefix(self, model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        if self.family == "opt":
            decoder = getattr(getattr(model, "model", None), "decoder", None)
            if not isinstance(decoder, nn.Module):
                raise UnsupportedModelVariant("SRC001 OPT model has no decoder")
            embeddings = cast(Any, decoder).embed_tokens(tokens.view(-1, tokens.shape[-1]))
            attention_mask = torch.ones(
                embeddings.shape[:2],
                device=embeddings.device,
                dtype=torch.long,
            )
            position_ids = (torch.cumsum(attention_mask, dim=1) * attention_mask - 1).long()
            positions = cast(Any, decoder).embed_positions(
                attention_mask,
                0,
                position_ids=position_ids,
            )
            project_in = getattr(decoder, "project_in", None)
            if isinstance(project_in, nn.Module):
                embeddings = project_in(embeddings)
            return cast(torch.Tensor, embeddings + positions.to(embeddings.device))
        get_embeddings = cast(Callable[[], nn.Module], cast(Any, model).get_input_embeddings)
        embeddings = cast(torch.Tensor, get_embeddings()(tokens))
        if self.definition.model_types[0] in {"gemma", "gemma2"}:
            config = cast(Any, model).config
            normalizer = torch.tensor(config.hidden_size**0.5, dtype=embeddings.dtype)
            embeddings = embeddings * normalizer
        return embeddings

    def run_block(self, block: nn.Module, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        result = block(inputs, **kwargs)
        return cast(torch.Tensor, result[0] if isinstance(result, tuple) else result)

    def run_suffix(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        if self.family == "opt":
            decoder = getattr(getattr(model, "model", None), "decoder", None)
            if not isinstance(decoder, nn.Module):
                raise UnsupportedModelVariant("SRC001 OPT model has no decoder")
            final_layer_norm = getattr(decoder, "final_layer_norm", None)
            if isinstance(final_layer_norm, nn.Module):
                inputs = final_layer_norm(inputs)
            project_out = getattr(decoder, "project_out", None)
            if isinstance(project_out, nn.Module):
                inputs = project_out(inputs)
            return cast(torch.Tensor, self.lm_head(model)(inputs))
        base = getattr(model, "model", model)
        norm = getattr(base, "norm", nn.Identity())
        logits = cast(torch.Tensor, self.lm_head(model)(norm(inputs)))
        if self.definition.model_types[0] in {"gemma2", "gemma3_text", "gemma3"}:
            softcap = getattr(cast(Any, model).config, "final_logit_softcapping", None)
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
        return logits

    def lm_head(self, model: nn.Module) -> nn.Module:
        head = getattr(model, "lm_head", None)
        if not isinstance(head, nn.Module):
            raise UnsupportedModelVariant("SRC001 model has no LM head")
        return head


ADAPTERS = {
    model_type: TransformersModelAdapter(definition)
    for definition in DEFINITIONS
    for model_type in definition.model_types
}


def adapter_for_config(config: dict[str, object]) -> TransformersModelAdapter:
    model_type = config.get("model_type")
    try:
        return ADAPTERS[cast(str, model_type)]
    except (KeyError, TypeError) as exc:
        raise UnsupportedModelVariant(
            f"SRC001 unsupported model_type={model_type!r}; supported={sorted(ADAPTERS)}"
        ) from exc
