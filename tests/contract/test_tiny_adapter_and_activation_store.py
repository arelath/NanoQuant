import pytest
import torch

from nanoquant.domain.models import BlockId
from nanoquant.infrastructure.activation_store import MemoryActivationStore
from nanoquant.infrastructure.tiny_model import TinyCausalTransformer, TinyModelAdapter


def test_tiny_adapter_prefix_block_suffix_matches_full_model_and_is_deterministic() -> None:
    first = TinyCausalTransformer(seed=5)
    second = TinyCausalTransformer(seed=5)
    adapter = TinyModelAdapter(first.config)
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]])
    value = adapter.run_prefix(first, tokens)
    for index, block in enumerate(first.blocks):
        layers = adapter.quantizable_layers(block, BlockId(index))
        assert len(layers) == 6
        assert all(adapter.source_key(layer).startswith(f"blocks.{index}.") for layer in layers)
        value = adapter.run_block(block, value)
    assert torch.allclose(adapter.run_suffix(first, value), first(tokens))
    assert torch.equal(first(tokens), second(tokens))
    assert first.lm_head.weight.data_ptr() == first.embed.weight.data_ptr()


def test_pageable_activation_store_clones_leases_and_releases() -> None:
    store = MemoryActivationStore("ram")
    source = torch.arange(6).reshape(2, 3)
    store.put("inputs", source)
    source.zero_()
    with store.read("inputs") as value:
        assert torch.equal(value, torch.arange(6).reshape(2, 3))
    store.remove("inputs")
    with pytest.raises(KeyError, match="not stored"):
        with store.read("inputs"):
            pass
    store.put("again", torch.ones(1))
    store.clear()


def test_activation_store_rejects_invalid_tier_and_duplicate_key() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        MemoryActivationStore("disk")
    store = MemoryActivationStore("ram")
    store.put("x", torch.ones(1))
    with pytest.raises(ValueError, match="unique"):
        store.put("x", torch.ones(1))
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="without CUDA"):
            MemoryActivationStore("cuda")
