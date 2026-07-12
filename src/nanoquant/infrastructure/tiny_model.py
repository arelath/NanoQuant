"""Offline deterministic causal-transformer adapter for integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from nanoquant.domain.models import BlockId, LayerId


@dataclass(frozen=True, slots=True)
class TinyModelConfig:
    vocabulary_size: int = 32
    hidden_size: int = 16
    intermediate_size: int = 32
    block_count: int = 2


class TinyAttention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(value), self.k_proj(value), self.v_proj(value)
        scores = q @ k.mT / value.shape[-1] ** 0.5
        mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        return cast(torch.Tensor, self.o_proj(torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1) @ v))


class TinyBlock(nn.Module):
    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.self_attn = TinyAttention(config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.ModuleDict(
            {
                "up_proj": nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                "down_proj": nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            }
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.self_attn(self.attn_norm(value))
        return cast(
            torch.Tensor,
            value + self.mlp["down_proj"](torch.nn.functional.gelu(self.mlp["up_proj"](self.mlp_norm(value)))),
        )


class TinyCausalTransformer(nn.Module):
    def __init__(self, config: TinyModelConfig | None = None, seed: int = 0) -> None:
        super().__init__()
        self.config = config or TinyModelConfig()
        config = self.config
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.embed = nn.Embedding(config.vocabulary_size, config.hidden_size)
            self.blocks = nn.ModuleList(TinyBlock(config) for _ in range(config.block_count))
            self.final_norm = nn.LayerNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocabulary_size, bias=False)
            self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.embed(tokens)
        for block in self.blocks:
            value = block(value)
        return cast(torch.Tensor, self.lm_head(self.final_norm(value)))


class TinyModelAdapter:
    family = "tiny"
    version = "1"

    def __init__(self, config: TinyModelConfig | None = None) -> None:
        self.config = config or TinyModelConfig()

    def decoder_block_count(self, source: object) -> int:
        return self.config.block_count

    def source_key(self, layer: LayerId, tensor_name: str = "weight") -> str:
        return f"blocks.{layer.block.index}.{layer.path}.{tensor_name}"

    def quantizable_layers(self, block: object, block_id: BlockId) -> tuple[LayerId, ...]:
        if not isinstance(block, TinyBlock):
            raise TypeError("tiny adapter requires TinyBlock")
        return tuple(
            LayerId(block_id, path)
            for path in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )

    def construct_block(self, source: object, block_id: BlockId, device: str) -> TinyBlock:
        if block_id.index >= self.config.block_count:
            raise IndexError(block_id.index)
        return TinyBlock(self.config).to(device)

    def run_prefix(self, model: object, batch: object) -> torch.Tensor:
        if not isinstance(model, TinyCausalTransformer) or not isinstance(batch, torch.Tensor):
            raise TypeError("invalid tiny prefix input")
        return cast(torch.Tensor, model.embed(batch))

    def run_block(self, block: object, inputs: object, **kwargs: object) -> torch.Tensor:
        if not isinstance(block, TinyBlock) or not isinstance(inputs, torch.Tensor):
            raise TypeError("invalid tiny block input")
        return cast(torch.Tensor, block(inputs))

    def run_suffix(self, model: object, inputs: object) -> torch.Tensor:
        if not isinstance(model, TinyCausalTransformer) or not isinstance(inputs, torch.Tensor):
            raise TypeError("invalid tiny suffix input")
        return cast(torch.Tensor, model.lm_head(model.final_norm(inputs)))
