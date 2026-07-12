from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from nanoquant.application.layers import (
    BlockEditor,
    FactorizedReferenceLinear,
    FrozenReferenceLinear,
    LayerFreezer,
    TrainableFactorizedLinear,
)
from nanoquant.domain.models import BlockId, FrozenOutlierState, LayerId
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.tensor_store import LocalTensorStore


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.ModuleDict({"up_proj": nn.Linear(3, 2, bias=False)})

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mlp["up_proj"](value)


def test_freezer_persists_immutable_state_and_editor_installs_explicitly(tmp_path: Path) -> None:
    trainable = TrainableFactorizedLinear(
        torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.5, 1.5]),
        torch.tensor([2.0, 1.0]),
    )
    inputs = torch.randn(4, 3, generator=torch.Generator().manual_seed(2))
    expected = trainable(inputs).detach()
    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))
    frozen = LayerFreezer().freeze(LayerId(BlockId(0), "mlp.up_proj"), trainable, tensors)
    assert isinstance(frozen.module, FrozenReferenceLinear)
    assert torch.equal(frozen.module(inputs), expected)
    block = Block()
    BlockEditor().install_frozen_layer(block, "mlp.up_proj", frozen.module)
    assert block.mlp["up_proj"] is frozen.module
    assert torch.equal(block(inputs), expected)
    trainable.left_latent.data.zero_()
    assert torch.equal(block(inputs), expected)
    with tensors.read(frozen.state.left_binary) as persisted:
        assert torch.equal(persisted, frozen.module.left_binary)
    loaded = LayerFreezer().load(frozen.state, tensors)
    factorized = LayerFreezer().load(frozen.state, tensors, backend="factorized")
    assert torch.equal(loaded.module(inputs), expected)
    assert torch.allclose(factorized.module(inputs), expected, atol=1e-6)
    assert loaded.state == frozen.state

    outliers = tensors.put(
        "outlier-fixture",
        {
            "indices": torch.tensor([1], dtype=torch.int64),
            "values": torch.tensor([[6], [-4]], dtype=torch.int8),
            "scales": torch.tensor([[0.5], [0.25]]),
        },
    )
    state_with_outliers = replace(
        frozen.state,
        outliers=FrozenOutlierState(outliers["indices"], outliers["values"], outliers["scales"]),
    )
    loaded_with_outliers = LayerFreezer().load(state_with_outliers, tensors)
    factorized_with_outliers = LayerFreezer().load(state_with_outliers, tensors, backend="factorized")
    expected_weight = frozen.module.dense_weight().clone()
    expected_weight[:, 1] += torch.tensor([3.0, -1.0])
    assert torch.equal(loaded_with_outliers.module.dense_weight(), expected_weight)
    assert torch.allclose(
        factorized_with_outliers.module(inputs),
        torch.nn.functional.linear(inputs, expected_weight),
        atol=1e-6,
    )


def test_editor_rejects_missing_or_non_linear_targets() -> None:
    block = Block()
    frozen = FrozenReferenceLinear(torch.ones(2, 1), torch.ones(1, 3), torch.ones(3), torch.ones(1), torch.ones(2))
    with pytest.raises(KeyError, match="not found"):
        BlockEditor().install_frozen_layer(block, "missing.value", frozen)
    with pytest.raises(TypeError, match="not a replaceable"):
        BlockEditor().install_frozen_layer(block, "mlp", frozen)


def test_trainable_factorized_linear_keeps_fp32_parameters_with_bfloat16_activations() -> None:
    trainable = TrainableFactorizedLinear(
        torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
        torch.tensor([[1.0, -1.0, 1.0]], dtype=torch.float32),
        torch.ones(3, dtype=torch.float32),
        torch.ones(1, dtype=torch.float32),
        torch.ones(2, dtype=torch.float32),
    )
    inputs = torch.randn(4, 3, generator=torch.Generator().manual_seed(9), dtype=torch.bfloat16)

    output = trainable(inputs)
    output.float().square().mean().backward()

    assert output.dtype is torch.bfloat16
    assert trainable.left_latent.dtype is torch.float32
    assert trainable.left_latent.grad is not None
    assert trainable.left_latent.grad.dtype is torch.float32


@pytest.mark.parametrize(
    ("out_features", "in_features", "rank"),
    ((256, 1152, 128), (1152, 1152, 448), (6912, 1152, 1056)),
)
def test_dense_and_factorized_references_match_real_gemma_shapes(
    out_features: int, in_features: int, rank: int
) -> None:
    generator = torch.Generator().manual_seed(out_features + rank)
    left = torch.where(torch.rand(out_features, rank, generator=generator) >= 0.5, 1.0, -1.0)
    right = torch.where(torch.rand(rank, in_features, generator=generator) >= 0.5, 1.0, -1.0)
    scale_pre = torch.rand(in_features, generator=generator) * 0.02
    scale_mid = torch.rand(rank, generator=generator) * 0.02
    scale_post = torch.rand(out_features, generator=generator) * 0.02
    dense = FrozenReferenceLinear(left, right, scale_pre, scale_mid, scale_post)
    factorized = FactorizedReferenceLinear(left, right, scale_pre, scale_mid, scale_post)
    inputs = torch.randn(2, in_features, generator=generator)

    assert torch.allclose(factorized(inputs), dense(inputs), rtol=2e-5, atol=2e-6)
