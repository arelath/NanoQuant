import pytest
import torch
from torch import nn

from nanoquant.resident_quantization import _block_loss, _run_block_batched


class _BlockAdapter:
    def run_block(self, block: nn.Module, value: torch.Tensor, **_metadata: object) -> torch.Tensor:
        return block(value)


def test_block_forward_does_not_retain_autograd_graphs() -> None:
    inputs = torch.randn(5, 4, requires_grad=True)
    block = nn.Linear(4, 3, bias=False)

    actual = _run_block_batched(_BlockAdapter(), block, inputs, {}, 2)

    assert not actual.requires_grad
    assert actual.grad_fn is None


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
