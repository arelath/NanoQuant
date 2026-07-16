from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from nanoquant.infrastructure.model_adapters import TransformersModelAdapter
from nanoquant.infrastructure.streamed_language_model import BlockStreamedCausalLM


class _Block(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, *, position_bias: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(hidden) + position_bias)


class _Core(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int, blocks: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocabulary_size, hidden_size)
        self.layers = nn.ModuleList(_Block(hidden_size) for _ in range(blocks))
        self.norm = nn.LayerNorm(hidden_size)


class _CausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.model = _Core(17, 8, 3)
        self.lm_head = nn.Linear(8, 17, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del use_cache
        hidden = self.model.embed_tokens(input_ids)
        position_bias = torch.arange(hidden.shape[-2], dtype=hidden.dtype).view(1, -1, 1)
        if attention_mask is not None:
            position_bias = position_bias * attention_mask.unsqueeze(-1)
        for block in self.model.layers:
            hidden = block(hidden, position_bias=position_bias)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(hidden)))


class _Adapter:
    family = "gemma"

    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        return cast(Any, model).model.layers

    def run_block(self, block: nn.Module, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return cast(torch.Tensor, cast(Any, block)(inputs, **kwargs))

    def run_suffix(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, cast(Any, model).lm_head(cast(Any, model).model.norm(inputs)))

    def lm_head(self, model: nn.Module) -> nn.Module:
        return cast(nn.Module, cast(Any, model).lm_head)


def test_block_streamed_forward_matches_resident_and_returns_weights_to_host() -> None:
    torch.manual_seed(4)
    model = _CausalModel().eval()
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    expected = model(tokens, attention_mask=attention_mask).logits
    streamed = BlockStreamedCausalLM(
        model,
        cast(TransformersModelAdapter, _Adapter()),
        "cpu",
    )

    actual = streamed(tokens, attention_mask=attention_mask).logits

    assert torch.equal(actual, expected)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
