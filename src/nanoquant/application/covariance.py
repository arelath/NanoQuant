"""Scalable covariance approximation persistence and dense-workspace policy."""

from __future__ import annotations

import torch

from nanoquant.domain.models import CovarianceRef
from nanoquant.domain.objectives import (
    BlockDiagonalObjective,
    DenseHessianObjective,
    DiagonalObjective,
    LowRankDiagonalObjective,
)
from nanoquant.ports.tensor_store import TensorStore


class DenseCovarianceAccumulator:
    """Bounded streaming first/second moments for one activation feature axis."""

    def __init__(self, width: int, maximum_rows: int, *, device: str | torch.device) -> None:
        if width <= 0 or maximum_rows <= 0:
            raise ValueError("dense covariance dimensions and row limit must be positive")
        self.width = width
        self.maximum_rows = maximum_rows
        self.row_count = 0
        self.total = torch.zeros(width, dtype=torch.float32, device=device)
        self.gram = torch.zeros(width, width, dtype=torch.float32, device=device)

    @property
    def remaining(self) -> int:
        return self.maximum_rows - self.row_count

    def update(self, value: torch.Tensor) -> int:
        if value.shape[-1] != self.width:
            raise ValueError("activation width differs from covariance accumulator")
        rows = value.detach().reshape(-1, self.width)
        take = min(self.remaining, rows.shape[0])
        if take <= 0:
            return 0
        selected = rows[:take].to(device=self.total.device, dtype=torch.float32)
        self.total.add_(selected.sum(dim=0))
        self.gram.addmm_(selected.mT, selected)
        self.row_count += take
        return take

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.row_count != self.maximum_rows:
            raise ValueError(
                f"covariance accumulator is incomplete: {self.row_count} != {self.maximum_rows}"
            )
        return (
            (self.gram / self.row_count).detach().cpu().contiguous(),
            (self.total / self.row_count).detach().cpu().contiguous(),
        )


class SplitDenseCovarianceAccumulator:
    """Route a stream into disjoint fit and held-out covariance windows."""

    def __init__(
        self,
        width: int,
        fit_rows: int,
        held_out_rows: int,
        *,
        device: str | torch.device,
    ) -> None:
        self.fit = DenseCovarianceAccumulator(width, fit_rows, device=device)
        self.held_out = DenseCovarianceAccumulator(width, held_out_rows, device=device)

    @property
    def complete(self) -> bool:
        return self.fit.remaining == 0 and self.held_out.remaining == 0

    def update(self, value: torch.Tensor) -> None:
        rows = value.detach().reshape(-1, value.shape[-1])
        consumed = self.fit.update(rows)
        if consumed < rows.shape[0]:
            self.held_out.update(rows[consumed:])


class DenseHessianWorkspaceError(MemoryError):
    code = "HES001"


def enforce_dense_hessian_reservation(width: int, available_bytes: int, *, dtype_bytes: int = 4) -> int:
    if width <= 0 or available_bytes < 0 or dtype_bytes <= 0:
        raise ValueError("invalid dense Hessian reservation inputs")
    required = width * width * dtype_bytes
    if required > available_bytes:
        raise DenseHessianWorkspaceError(
            f"HES001 dense covariance requires {required} bytes but reservation is {available_bytes}"
        )
    return required


def _rows(activations: torch.Tensor) -> torch.Tensor:
    if activations.ndim < 2 or activations.shape[-1] == 0:
        raise ValueError("covariance activations require a non-empty feature dimension")
    return activations.detach().flatten(0, -2).float()


def materialize_covariance(
    activations: torch.Tensor,
    representation: str,
    tensors: TensorStore,
    *,
    block_size: int | None = None,
    rank: int | None = None,
    dense_workspace_bytes: int | None = None,
) -> CovarianceRef:
    rows = _rows(activations)
    covariance = rows.mT @ rows / rows.shape[0]
    diagonal = covariance.diagonal().clone()
    values: dict[str, torch.Tensor] = {"diagonal": diagonal}
    if representation == "diagonal":
        pass
    elif representation == "block_diagonal":
        if block_size is None or block_size <= 0:
            raise ValueError("block-diagonal covariance requires a positive block size")
        count = (covariance.shape[0] + block_size - 1) // block_size
        padded = torch.zeros(count, block_size, block_size, dtype=covariance.dtype)
        for index, start in enumerate(range(0, covariance.shape[0], block_size)):
            block = covariance[start : start + block_size, start : start + block_size]
            padded[index, : block.shape[0], : block.shape[1]] = block
        values["blocks"] = padded
    elif representation == "low_rank_diagonal":
        if rank is None or rank <= 0 or rank > covariance.shape[0]:
            raise ValueError("low-rank covariance rank is invalid")
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        selected = eigenvalues[-rank:].clamp_min(0)
        factors = eigenvectors[:, -rank:] * selected.sqrt().reshape(1, -1)
        residual_diagonal = (diagonal - factors.square().sum(dim=1)).clamp_min(0)
        values["diagonal"] = residual_diagonal
        values["low_rank_factors"] = factors
    elif representation == "dense":
        enforce_dense_hessian_reservation(
            covariance.shape[0],
            dense_workspace_bytes if dense_workspace_bytes is not None else covariance.numel() * 4,
        )
        values["dense"] = covariance
    else:
        raise ValueError(f"unsupported covariance representation: {representation}")
    refs = tensors.put("covariance", values)
    return CovarianceRef(
        representation,
        refs["diagonal"],
        refs.get("blocks"),
        refs.get("low_rank_factors"),
        refs.get("dense"),
        rows.shape[0],
    )


def load_covariance_objective(
    reference: CovarianceRef, output_importance: torch.Tensor, tensors: TensorStore
) -> DiagonalObjective | BlockDiagonalObjective | LowRankDiagonalObjective | DenseHessianObjective:
    with tensors.read(reference.diagonal) as value:
        diagonal = value.clone()
    if reference.representation == "diagonal":
        return DiagonalObjective(diagonal, output_importance)
    if reference.representation == "block_diagonal":
        if reference.blocks is None:
            raise ValueError("block-diagonal covariance reference has no blocks")
        with tensors.read(reference.blocks) as value:
            padded = value.clone()
        width = diagonal.numel()
        blocks = tuple(
            padded[
                index,
                : min(padded.shape[1], width - index * padded.shape[1]),
                : min(padded.shape[1], width - index * padded.shape[1]),
            ]
            for index in range(padded.shape[0])
        )
        return BlockDiagonalObjective(blocks, padded.shape[1], output_importance)
    if reference.representation == "low_rank_diagonal":
        if reference.low_rank_factors is None:
            raise ValueError("low-rank covariance reference has no factors")
        with tensors.read(reference.low_rank_factors) as value:
            factors = value.clone()
        return LowRankDiagonalObjective(diagonal, factors, output_importance)
    if reference.representation == "dense":
        if reference.dense is None:
            raise ValueError("dense covariance reference has no matrix")
        with tensors.read(reference.dense) as value:
            dense = value.clone()
        return DenseHessianObjective(dense, output_importance)
    raise ValueError(f"unsupported covariance representation: {reference.representation}")
