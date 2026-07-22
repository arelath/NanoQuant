from pathlib import Path

import pytest
import torch

from nanoquant.application.covariance import (
    DenseCovarianceAccumulator,
    DenseHessianWorkspaceError,
    SplitDenseCovarianceAccumulator,
    enforce_dense_hessian_reservation,
    load_covariance_objective,
    materialize_covariance,
)
from nanoquant.domain.objectives import DenseHessianObjective
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.tensor_store import LocalTensorStore


def test_split_dense_covariance_accumulator_uses_disjoint_bounded_rows() -> None:
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    )
    accumulator = SplitDenseCovarianceAccumulator(2, 2, 2, device="cpu")
    accumulator.update(values[:3])
    accumulator.update(values[3:])
    fit_covariance, fit_mean = accumulator.fit.materialize()
    held_covariance, held_mean = accumulator.held_out.materialize()

    assert accumulator.complete
    assert torch.allclose(fit_mean, values[:2].mean(dim=0))
    assert torch.allclose(fit_covariance, values[:2].mT @ values[:2] / 2)
    assert torch.allclose(held_mean, values[2:4].mean(dim=0))
    assert torch.allclose(held_covariance, values[2:4].mT @ values[2:4] / 2)


def test_dense_covariance_accumulator_rejects_incomplete_materialization() -> None:
    accumulator = DenseCovarianceAccumulator(2, 3, device="cpu")
    accumulator.update(torch.ones(2, 2))

    with pytest.raises(ValueError, match="incomplete"):
        accumulator.materialize()


def test_block_diagonal_covariance_storage_and_execution_match_dense_blocks(tmp_path: Path) -> None:
    activations = torch.randn(3, 5, generator=torch.Generator().manual_seed(4))
    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))
    reference = materialize_covariance(activations, "block_diagonal", tensors, block_size=2)
    objective = load_covariance_objective(reference, torch.tensor([1.0, 2.0]), tensors)
    target = torch.randn(2, 5, generator=torch.Generator().manual_seed(5))
    prediction = target + 0.2
    covariance = activations.mT @ activations / activations.shape[0]
    blocked = torch.zeros_like(covariance)
    for start in range(0, 5, 2):
        blocked[start : start + 2, start : start + 2] = covariance[start : start + 2, start : start + 2]

    expected = DenseHessianObjective(blocked, torch.tensor([1.0, 2.0])).weighted_error(target, prediction)
    assert torch.allclose(objective.weighted_error(target, prediction), expected)
    assert reference.blocks is not None
    assert reference.token_count == 3


def test_full_rank_low_rank_diagonal_storage_reconstructs_dense_objective(tmp_path: Path) -> None:
    activations = torch.randn(8, 4, generator=torch.Generator().manual_seed(6))
    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))
    reference = materialize_covariance(activations, "low_rank_diagonal", tensors, rank=4)
    objective = load_covariance_objective(reference, torch.ones(3), tensors)
    target = torch.randn(3, 4, generator=torch.Generator().manual_seed(7))
    prediction = target - 0.1
    covariance = activations.mT @ activations / activations.shape[0]
    expected = DenseHessianObjective(covariance, torch.ones(3)).weighted_error(target, prediction)

    assert torch.allclose(objective.weighted_error(target, prediction), expected, atol=1e-5)
    assert reference.low_rank_factors is not None


def test_dense_hessian_requires_explicit_workspace_reservation(tmp_path: Path) -> None:
    assert enforce_dense_hessian_reservation(8, 256) == 256
    with pytest.raises(DenseHessianWorkspaceError, match=r"HES001.*requires 256.*reservation is 255"):
        enforce_dense_hessian_reservation(8, 255)
    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))
    with pytest.raises(DenseHessianWorkspaceError):
        materialize_covariance(
            torch.randn(2, 8),
            "dense",
            tensors,
            dense_workspace_bytes=255,
        )
