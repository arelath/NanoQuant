import pytest
import torch
from torch import nn

from nanoquant.resident_quantization import _run_block_batched


class _BlockAdapter:
    def run_block(self, block: nn.Module, value: torch.Tensor, **_metadata: object) -> torch.Tensor:
        return block(value)


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
