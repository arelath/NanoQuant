import pytest
import torch
from torch import nn

from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.infrastructure.profiling import Profiler
from nanoquant.resident_quantization import (
    _block_loss,
    _peak_device_memory_bytes,
    _release_uncompleted_decoder_blocks,
    _run_block_batched,
)


class _BlockAdapter:
    def run_block(self, block: nn.Module, value: torch.Tensor, **_metadata: object) -> torch.Tensor:
        return block(value)


def test_uncompleted_dense_decoder_blocks_are_released_from_model_shell() -> None:
    completed = nn.Linear(2, 2)
    layers = nn.ModuleList((nn.Linear(2, 2), completed, nn.Linear(2, 2)))

    released = _release_uncompleted_decoder_blocks(layers, {1})

    assert released == 2
    assert isinstance(layers[0], nn.Identity)
    assert layers[1] is completed
    assert isinstance(layers[2], nn.Identity)


def test_block_forward_does_not_retain_autograd_graphs() -> None:
    inputs = torch.randn(5, 4, requires_grad=True)
    block = nn.Linear(4, 3, bias=False)

    actual = _run_block_batched(_BlockAdapter(), block, inputs, {}, 2)

    assert not actual.requires_grad
    assert actual.grad_fn is None


def test_block_forward_micro_profile_preserves_output_and_attributes_batches() -> None:
    inputs = torch.randn(5, 4, generator=torch.Generator().manual_seed(44))
    block = nn.Linear(4, 3, bias=False)
    expected = _run_block_batched(_BlockAdapter(), block, inputs, {}, 2)
    profiler = Profiler(ProfilingConfig(level=ProfilingLevel.MICRO), run_id="block-forward")

    actual = _run_block_batched(_BlockAdapter(), block, inputs, {}, 2, recorder=profiler)

    assert torch.equal(actual, expected)
    payload = profiler.snapshot()
    phases = {phase["path"]: phase for phase in payload["phases"]}
    assert {"batch_stage", "forward", "store"} <= phases.keys()
    assert phases["batch_stage"]["count"] == phases["forward"]["count"] == phases["store"]["count"] == 3
    assert all(phase["failed_count"] == 0 for phase in phases.values())
    counters = {counter["name"]: counter for counter in payload["counters"]}
    assert counters["forward.batches"]["total"] == 3
    assert counters["forward.elements"]["total"] == actual.numel()


def test_block_loss_micro_profile_preserves_accumulation_and_attributes_work() -> None:
    inputs = torch.randn(5, 4, generator=torch.Generator().manual_seed(45))
    targets = torch.randn(5, 3, generator=torch.Generator().manual_seed(46))
    importance = torch.linspace(0.5, 1.5, 3)
    block = nn.Linear(4, 3, bias=False)
    expected = _block_loss(_BlockAdapter(), block, inputs, targets, importance, {}, 2)
    profiler = Profiler(ProfilingConfig(level=ProfilingLevel.MICRO), run_id="block-loss")

    actual = _block_loss(_BlockAdapter(), block, inputs, targets, importance, {}, 2, profiler)

    assert actual == expected
    payload = profiler.snapshot()
    phases = {phase["path"]: phase for phase in payload["phases"]}
    assert {"batch_stage", "forward", "loss", "synchronize"} <= phases.keys()
    assert phases["batch_stage"]["count"] == phases["forward"]["count"] == phases["loss"]["count"] == 3
    assert phases["synchronize"]["count"] == 1
    assert all(phase["failed_count"] == 0 for phase in phases.values())
    counters = {counter["name"]: counter for counter in payload["counters"]}
    assert counters["forward.batches"]["total"] == 3
    assert counters["forward.elements"]["total"] == targets.numel()


def test_peak_device_memory_uses_reserved_allocator_high_water(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 6_000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 9_000)

    assert _peak_device_memory_bytes("cuda:0") == 9_000
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 4_000)
    assert _peak_device_memory_bytes("cuda:0") == 6_000
    assert _peak_device_memory_bytes("cpu") == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned CUDA transfer requires a GPU")
def test_cuda_block_forward_produces_bitwise_equal_pinned_host_activations() -> None:
    inputs = torch.randn(5, 4, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(41))
    block = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        actual = _run_block_batched(_BlockAdapter(), block, inputs, {}, 2, "cpu")
        expected = torch.cat(
            [block(inputs[start : start + 2].cuda()).cpu() for start in range(0, inputs.shape[0], 2)]
        )

    assert actual.is_pinned()
    assert torch.equal(actual, expected)
    del block
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned CUDA transfer requires a GPU")
def test_prefetched_block_loss_matches_pageable_accumulation_bitwise() -> None:
    inputs = torch.randn(7, 4, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(42))
    targets = torch.randn(7, 3, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(43))
    importance = torch.linspace(0.5, 1.5, 3)
    block = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16, device="cuda")

    pageable = _block_loss(_BlockAdapter(), block, inputs, targets, importance, {}, 2)
    prefetched = _block_loss(
        _BlockAdapter(), block, inputs.pin_memory(), targets.pin_memory(), importance, {}, 2
    )

    assert prefetched == pageable
    del block
    torch.cuda.empty_cache()
