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
    freeze_block_auxiliary_parameters,
    restore_block_auxiliary_parameters,
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
    expected_weight[:, 1] = torch.tensor([3.0, -1.0])
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


def test_block_auxiliary_parameters_round_trip_by_name(tmp_path: Path) -> None:
    block = Block()
    block.register_parameter("gain", nn.Parameter(torch.tensor([1.25, -0.5])))
    frozen = FrozenReferenceLinear(
        torch.ones(2, 1),
        torch.ones(1, 3),
        torch.ones(3),
        torch.ones(1),
        torch.ones(2),
    )
    BlockEditor().install_frozen_layer(block, "mlp.up_proj", frozen)
    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))

    parameters = freeze_block_auxiliary_parameters(block, tensors)
    assert tuple(name for name, _reference in parameters) == ("gain",)
    block.gain.data.zero_()
    restore_block_auxiliary_parameters(block, parameters, tensors, device="cpu")

    assert torch.equal(block.gain, torch.tensor([1.25, -0.5]))


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


def test_trainable_forward_uses_two_stage_factorized_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    trainable = TrainableFactorizedLinear(
        torch.tensor([[0.2, -0.7], [-0.4, 0.9]], dtype=torch.float32),
        torch.tensor([[0.8, -0.1, 0.5], [-0.6, 0.3, -0.2]], dtype=torch.float32),
        torch.tensor([0.75, 1.25, 0.5], dtype=torch.float32),
        torch.tensor([1.5, 0.25], dtype=torch.float32),
        torch.tensor([0.5, 2.0], dtype=torch.float32),
        bias=torch.tensor([0.25, -0.5], dtype=torch.float32),
        outlier_indices=torch.tensor([1]),
        outlier_values=torch.tensor([[0.75], [-0.25]], dtype=torch.float32),
    )
    inputs = torch.tensor([[1.5, -0.75, 0.25], [-0.5, 0.125, 2.0]], dtype=torch.bfloat16)

    def fail_dense_materialization() -> torch.Tensor:
        raise AssertionError("trainable forward materialized a dense factor product")

    monkeypatch.setattr(trainable, "dense_weight", fail_dense_materialization)
    right = torch.where(trainable.right_latent >= 0, 1.0, -1.0).to(torch.bfloat16)
    left = torch.where(trainable.left_latent >= 0, 1.0, -1.0).to(torch.bfloat16)
    masked_scale_pre = trainable.scale_pre.detach().clone()
    masked_scale_pre[trainable.outlier_indices.long()] = 0
    expected = torch.nn.functional.linear(inputs * masked_scale_pre.to(torch.bfloat16), right)
    expected = torch.nn.functional.linear(expected * trainable.scale_mid.to(torch.bfloat16), left)
    expected = expected * trainable.scale_post.to(torch.bfloat16)
    expected = expected + torch.nn.functional.linear(
        inputs.index_select(-1, trainable.outlier_indices.long()),
        trainable.outlier_values.to(torch.bfloat16),
    )
    expected = expected + trainable.bias.to(torch.bfloat16)

    actual = trainable(inputs)
    assert torch.equal(actual, expected)
    actual.float().sum().backward()
    assert trainable.left_latent.grad is not None
    assert trainable.right_latent.grad is not None
    assert trainable.scale_pre.grad is not None
    assert trainable.scale_pre.grad[trainable.outlier_indices.long()].count_nonzero() == 0


def test_freezer_zeroes_main_path_scale_for_outlier_columns(tmp_path: Path) -> None:
    trainable = TrainableFactorizedLinear(
        torch.tensor([[0.2], [-0.3]]),
        torch.tensor([[0.4, -0.5, 0.6]]),
        torch.tensor([1.0, 7.0, 2.0]),
        torch.tensor([0.5]),
        torch.tensor([2.0, 3.0]),
        outlier_indices=torch.tensor([1]),
        outlier_values=torch.tensor([[4.0], [-2.0]]),
    )
    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))

    frozen = LayerFreezer().freeze(
        LayerId(BlockId(0), "mlp.up_proj"), trainable, tensors, backend="factorized"
    )

    assert frozen.module.scale_pre[1] == 0
    with tensors.read(frozen.state.scales.pre) as persisted:
        assert persisted[1] == 0
    expected = trainable.dense_weight().detach()
    assert torch.equal(frozen.module.dense_weight(), expected)


def test_freezer_can_return_factorized_execution_backend(tmp_path: Path) -> None:
    trainable = TrainableFactorizedLinear(
        torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.5, 1.5]),
        torch.tensor([2.0, 1.0]),
    )
    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))

    frozen = LayerFreezer().freeze(
        LayerId(BlockId(0), "mlp.up_proj"),
        trainable,
        tensors,
        backend="factorized",
    )

    assert isinstance(frozen.module, FactorizedReferenceLinear)
    inputs = torch.randn(4, 3, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(frozen.module(inputs), trainable(inputs), atol=1e-6)


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
