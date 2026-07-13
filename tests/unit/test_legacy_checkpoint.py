import pytest
import torch
from torch import nn

from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.infrastructure.legacy_checkpoint import apply_legacy_checkpoint, unpack_binary_gemv


def _pack(value: torch.Tensor) -> torch.Tensor:
    rows, columns = value.shape
    words = (columns + 31) // 32
    padded = torch.ones((rows, words * 32), dtype=torch.int8)
    padded[:, :columns] = value
    bits = ((1 - padded) // 2).to(torch.int32).reshape(rows, words, 32)
    powers = 2 ** torch.arange(32, dtype=torch.int64)
    return (bits.to(torch.int64) * powers).sum(dim=-1).to(torch.int32)


@pytest.mark.parametrize("shape", ((2, 3), (3, 32), (2, 35)))
def test_unpack_binary_gemv_roundtrips_lsb_first_rows(shape: tuple[int, int]) -> None:
    generator = torch.Generator().manual_seed(sum(shape))
    value = torch.where(torch.rand(shape, generator=generator) >= 0.5, 1, -1).to(torch.int8)

    assert torch.equal(unpack_binary_gemv(_pack(value), shape), value)


def test_unpack_binary_gemv_rejects_inconsistent_metadata() -> None:
    with pytest.raises(ValueError, match="shape differs"):
        unpack_binary_gemv(torch.zeros((2, 1), dtype=torch.int32), (3, 4))


class _TinyLegacyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 3)
        block = nn.Module()
        block.proj = nn.Linear(3, 2, bias=False)
        block.norm = nn.LayerNorm(3)
        self.model.layers = nn.ModuleList((block,))


def test_apply_legacy_checkpoint_installs_factors_embedding_and_auxiliary_parameters() -> None:
    model = _TinyLegacyModel()
    left = torch.tensor([[1, -1], [-1, 1]], dtype=torch.int8)
    right = torch.tensor([[1, -1, 1], [-1, -1, 1]], dtype=torch.int8)
    embedding_int8 = torch.tensor(
        [[1, 2, 3], [-1, 0, 1], [2, -2, 0], [1, 1, -1]],
        dtype=torch.int8,
    )
    state = {
        "model.layers.0.proj.U_packed": _pack(left),
        "model.layers.0.proj.U_shape": torch.tensor(left.shape),
        "model.layers.0.proj.V_packed": _pack(right),
        "model.layers.0.proj.V_shape": torch.tensor(right.shape),
        "model.layers.0.proj.scale_pre": torch.tensor([0.5, 1.0, 1.5]),
        "model.layers.0.proj.scale_mid": torch.tensor([2.0, 3.0]),
        "model.layers.0.proj.scale_post": torch.tensor([0.25, 0.75]),
        "model.embed_tokens.weight_int8": embedding_int8,
        "model.embed_tokens.weight_int8_scale": torch.tensor([0.5, 1.0, 1.5, 2.0]),
        "model.layers.0.norm.weight": torch.tensor([3.0, 4.0, 5.0]),
    }

    installed = apply_legacy_checkpoint(model, state)

    assert installed == ("model.layers.0.proj",)
    assert isinstance(model.model.layers[0].proj, FactorizedReferenceLinear)
    assert torch.equal(model.model.layers[0].proj.left_binary, left.float())
    assert torch.equal(model.model.layers[0].proj.right_binary, right.float())
    expected_embedding = embedding_int8.float() * state["model.embed_tokens.weight_int8_scale"].reshape(-1, 1)
    assert torch.equal(model.model.embed_tokens.weight, expected_embedding)
    assert torch.equal(model.model.layers[0].norm.weight, state["model.layers.0.norm.weight"])
