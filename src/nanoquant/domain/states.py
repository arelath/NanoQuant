"""Explicit trainable-to-frozen materialized state conversion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .scale_fit import reconstruct


@dataclass(slots=True)
class TrainableNanoQuantState:
    left_latent: torch.Tensor
    right_latent: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    outlier_indices: torch.Tensor | None = None
    outlier_values: torch.Tensor | None = None
    bias: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MaterializedFrozenNanoQuantState:
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    outlier_indices: torch.Tensor | None
    outlier_values: torch.Tensor | None
    bias: torch.Tensor | None

    def dense_weight(self) -> torch.Tensor:
        result = reconstruct(self.left_binary, self.right_binary, self.scale_pre, self.scale_mid, self.scale_post)
        if self.outlier_indices is not None and self.outlier_values is not None:
            result[:, self.outlier_indices.long()] += self.outlier_values.to(result.dtype)
        return result


def freeze_state(state: TrainableNanoQuantState) -> MaterializedFrozenNanoQuantState:
    if state.left_latent.ndim != 2 or state.right_latent.ndim != 2:
        raise ValueError("factor tensors must be matrices")
    if state.left_latent.shape[1] != state.right_latent.shape[0]:
        raise ValueError("factor ranks do not match")
    output, rank = state.left_latent.shape
    _, input_features = state.right_latent.shape
    if (
        state.scale_pre.numel() != input_features
        or state.scale_mid.numel() != rank
        or state.scale_post.numel() != output
    ):
        raise ValueError("scale dimensions do not match factors")
    if (state.outlier_indices is None) != (state.outlier_values is None):
        raise ValueError("outlier indices and values must be provided together")
    if state.outlier_indices is not None and state.outlier_values is not None:
        if state.outlier_values.shape != (output, state.outlier_indices.numel()):
            raise ValueError("outlier value dimensions do not match")

    def sign(value: torch.Tensor) -> torch.Tensor:
        return torch.where(value.detach() >= 0, torch.ones_like(value), -torch.ones_like(value))

    def clone(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.detach().clone().contiguous()

    return MaterializedFrozenNanoQuantState(
        sign(state.left_latent).contiguous(),
        sign(state.right_latent).contiguous(),
        state.scale_pre.detach().clone().reshape(-1),
        state.scale_mid.detach().clone().reshape(-1),
        state.scale_post.detach().clone().reshape(-1),
        clone(state.outlier_indices),
        clone(state.outlier_values),
        clone(state.bias),
    )
