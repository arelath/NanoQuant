"""Explicit trainable/frozen research linears and stage-owned block editing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from nanoquant.domain.linear_math import (
    functional_dense_reconstruction,
    functional_factorized_linear,
    mask_outlier_columns,
)
from nanoquant.domain.models import FrozenNanoQuantState, FrozenOutlierState, LayerId, ScaleState, TensorRef
from nanoquant.ports.tensor_store import TensorStore


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        return (value >= 0).to(dtype=value.dtype).mul_(2).sub_(1)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (gradient,)


class TrainableFactorizedLinear(nn.Module):
    outlier_indices: torch.Tensor | None
    outlier_values: nn.Parameter | None
    outlier_scales: torch.Tensor | None
    immutable_binary_factors: bool

    def __init__(
        self,
        left_latent: torch.Tensor,
        right_latent: torch.Tensor,
        scale_pre: torch.Tensor,
        scale_mid: torch.Tensor,
        scale_post: torch.Tensor,
        bias: torch.Tensor | None = None,
        outlier_indices: torch.Tensor | None = None,
        outlier_values: torch.Tensor | None = None,
        outlier_scales: torch.Tensor | None = None,
        *,
        immutable_binary_factors: bool = False,
    ) -> None:
        super().__init__()
        if left_latent.shape[1] != right_latent.shape[0]:
            raise ValueError("factor ranks do not match")
        self.left_latent = nn.Parameter(left_latent.detach().clone())
        self.right_latent = nn.Parameter(right_latent.detach().clone())
        self.scale_pre = nn.Parameter(scale_pre.detach().clone().reshape(-1))
        self.scale_mid = nn.Parameter(scale_mid.detach().clone().reshape(-1))
        self.scale_post = nn.Parameter(scale_post.detach().clone().reshape(-1))
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone())
        self.register_buffer("outlier_indices", None if outlier_indices is None else outlier_indices.detach().clone())
        if (outlier_indices is None) != (outlier_values is None):
            raise ValueError("outlier indices and values must be provided together")
        if outlier_values is not None and not outlier_values.is_floating_point():
            if outlier_scales is None:
                raise ValueError("quantized outlier values require scales")
            outlier_values = outlier_values.float() * outlier_scales.float()
            outlier_scales = None
        self.outlier_values = None if outlier_values is None else nn.Parameter(outlier_values.detach().clone())
        self.register_buffer("outlier_scales", None if outlier_scales is None else outlier_scales.detach().clone())
        self.immutable_binary_factors = immutable_binary_factors

    def dense_weight(self) -> torch.Tensor:
        apply_sign = cast(Any, _SignSTE).apply
        return functional_dense_reconstruction(
            cast(torch.Tensor, apply_sign(self.left_latent)),
            cast(torch.Tensor, apply_sign(self.right_latent)),
            self.scale_pre,
            self.scale_mid,
            self.scale_post,
            self.outlier_indices,
            self.outlier_values,
            self.outlier_scales,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        apply_sign = cast(Any, _SignSTE).apply
        right = (
            self.right_latent
            if self.immutable_binary_factors and not self.right_latent.requires_grad
            else cast(torch.Tensor, apply_sign(self.right_latent))
        ).to(device=value.device, dtype=value.dtype)
        left = (
            self.left_latent
            if self.immutable_binary_factors and not self.left_latent.requires_grad
            else cast(torch.Tensor, apply_sign(self.left_latent))
        ).to(device=value.device, dtype=value.dtype)
        return functional_factorized_linear(
            value,
            left,
            right,
            self.scale_pre.to(device=value.device, dtype=value.dtype),
            self.scale_mid.to(device=value.device, dtype=value.dtype),
            self.scale_post.to(device=value.device, dtype=value.dtype),
            None if self.bias is None else self.bias.to(device=value.device, dtype=value.dtype),
            self.outlier_indices,
            self.outlier_values,
            self.outlier_scales,
        )


class FrozenReferenceLinear(nn.Module):
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    bias: torch.Tensor | None
    outlier_indices: torch.Tensor | None
    outlier_values: torch.Tensor | None
    outlier_scales: torch.Tensor | None
    _cached_dense_weight: torch.Tensor | None

    def __init__(
        self,
        left_binary: torch.Tensor,
        right_binary: torch.Tensor,
        scale_pre: torch.Tensor,
        scale_mid: torch.Tensor,
        scale_post: torch.Tensor,
        bias: torch.Tensor | None = None,
        outlier_indices: torch.Tensor | None = None,
        outlier_values: torch.Tensor | None = None,
        outlier_scales: torch.Tensor | None = None,
        *,
        cache_dense_weight: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("left_binary", left_binary.detach().clone())
        self.register_buffer("right_binary", right_binary.detach().clone())
        self.register_buffer("scale_pre", scale_pre.detach().clone().reshape(-1))
        self.register_buffer("scale_mid", scale_mid.detach().clone().reshape(-1))
        self.register_buffer("scale_post", scale_post.detach().clone().reshape(-1))
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.register_buffer("outlier_indices", None if outlier_indices is None else outlier_indices.detach().clone())
        self.register_buffer("outlier_values", None if outlier_values is None else outlier_values.detach().clone())
        self.register_buffer("outlier_scales", None if outlier_scales is None else outlier_scales.detach().clone())
        if (outlier_indices is None) != (outlier_values is None):
            raise ValueError("outlier indices and values must be provided together")
        if outlier_scales is not None and outlier_values is None:
            raise ValueError("outlier scales require outlier values")
        self.register_buffer(
            "_cached_dense_weight",
            self._materialize_dense_weight() if cache_dense_weight else None,
            persistent=False,
        )

    def _materialize_dense_weight(self) -> torch.Tensor:
        return functional_dense_reconstruction(
            self.left_binary,
            self.right_binary,
            self.scale_pre,
            self.scale_mid,
            self.scale_post,
            self.outlier_indices,
            self.outlier_values,
            self.outlier_scales,
        )

    def dense_weight(self) -> torch.Tensor:
        return self._materialize_dense_weight() if self._cached_dense_weight is None else self._cached_dense_weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.dense_weight(), self.bias)


class FactorizedReferenceLinear(FrozenReferenceLinear):
    def __init__(
        self,
        left_binary: torch.Tensor,
        right_binary: torch.Tensor,
        scale_pre: torch.Tensor,
        scale_mid: torch.Tensor,
        scale_post: torch.Tensor,
        bias: torch.Tensor | None = None,
        outlier_indices: torch.Tensor | None = None,
        outlier_values: torch.Tensor | None = None,
        outlier_scales: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            left_binary,
            right_binary,
            scale_pre,
            scale_mid,
            scale_post,
            bias,
            outlier_indices,
            outlier_values,
            outlier_scales,
            cache_dense_weight=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return functional_factorized_linear(
            value,
            self.left_binary,
            self.right_binary,
            self.scale_pre,
            self.scale_mid,
            self.scale_post,
            self.bias,
            self.outlier_indices,
            self.outlier_values,
            self.outlier_scales,
            scale_left_before_linear=True,
        )


@dataclass(frozen=True, slots=True)
class FrozenLayer:
    state: FrozenNanoQuantState
    module: FrozenReferenceLinear


class LayerFreezer:
    def freeze(
        self,
        layer: LayerId,
        trainable: TrainableFactorizedLinear,
        tensors: TensorStore,
        logical_format: str = "nanoquant-v1",
        outliers: FrozenOutlierState | None = None,
        backend: str = "dense",
    ) -> FrozenLayer:
        left = torch.where(trainable.left_latent.detach() >= 0, 1.0, -1.0)
        right = torch.where(trainable.right_latent.detach() >= 0, 1.0, -1.0)
        values: dict[str, torch.Tensor] = {
            "left_binary": left,
            "right_binary": right,
            "scale_pre": mask_outlier_columns(
                trainable.scale_pre.detach(), trainable.outlier_indices
            ),
            "scale_mid": trainable.scale_mid.detach(),
            "scale_post": trainable.scale_post.detach(),
        }
        if trainable.bias is not None:
            values["bias"] = trainable.bias.detach()
        if trainable.outlier_indices is not None and trainable.outlier_values is not None:
            values["outlier_indices"] = trainable.outlier_indices.detach()
            values["outlier_values"] = trainable.outlier_values.detach()
            if trainable.outlier_scales is not None:
                values["outlier_scales"] = trainable.outlier_scales.detach()
        refs = tensors.put("frozen-layer", values)
        scales = ScaleState(refs["scale_pre"], refs["scale_mid"], refs["scale_post"])
        if "outlier_indices" in refs:
            outliers = FrozenOutlierState(
                refs["outlier_indices"],
                refs["outlier_values"],
                refs.get("outlier_scales"),
            )
        state = FrozenNanoQuantState(
            layer,
            left.shape[1],
            refs["left_binary"],
            refs["right_binary"],
            scales,
            outliers,
            refs.get("bias"),
            logical_format,
        )
        return self.load(
            state,
            tensors,
            device=str(trainable.left_latent.device),
            dtype=trainable.left_latent.dtype,
            backend=backend,
        )

    def load(
        self,
        state: FrozenNanoQuantState,
        tensors: TensorStore,
        *,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
        backend: str = "dense",
    ) -> FrozenLayer:
        if backend not in {"dense", "factorized"}:
            raise ValueError(f"unsupported frozen reference backend: {backend}")
        if state.scales.mid is None:
            raise ValueError("frozen NanoQuant state is missing its mid scale")
        with (
            tensors.read(state.left_binary, device) as left,
            tensors.read(state.right_binary, device) as right,
            tensors.read(state.scales.pre, device) as scale_pre,
            tensors.read(state.scales.mid, device) as scale_mid,
            tensors.read(state.scales.post, device) as scale_post,
        ):
            bias = None
            if state.bias is not None:
                with tensors.read(state.bias, device) as value:
                    bias = value.clone()
            outlier_indices = None
            outlier_values = None
            outlier_scales = None
            if state.outliers is not None:
                with (
                    tensors.read(state.outliers.indices, device) as indices,
                    tensors.read(state.outliers.values, device) as values,
                ):
                    outlier_indices = indices.clone()
                    outlier_values = values.clone()
                if state.outliers.scales is not None:
                    with tensors.read(state.outliers.scales, device) as scales:
                        outlier_scales = scales.clone()
            module_type = FrozenReferenceLinear if backend == "dense" else FactorizedReferenceLinear
            module = module_type(
                left,
                right,
                scale_pre,
                scale_mid,
                scale_post,
                bias,
                outlier_indices,
                outlier_values,
                outlier_scales,
            )
        if dtype is not None:
            module = module.to(dtype=dtype)
        return FrozenLayer(state, module)


class BlockEditor:
    def _replace(self, block: nn.Module, path: str, replacement: nn.Module) -> None:
        parts = path.split(".")
        if not parts or any(not part for part in parts):
            raise ValueError("invalid module path")
        parent: nn.Module = block
        for part in parts[:-1]:
            child = parent[part] if isinstance(parent, nn.ModuleDict) else getattr(parent, part, None)
            if not isinstance(child, nn.Module):
                raise KeyError(f"module path not found: {path}")
            parent = child
        name = parts[-1]
        existing = parent[name] if isinstance(parent, nn.ModuleDict) and name in parent else getattr(parent, name, None)
        if not isinstance(existing, (nn.Linear, TrainableFactorizedLinear, FrozenReferenceLinear)):
            raise TypeError(f"target is not a replaceable linear: {path}")
        if isinstance(parent, nn.ModuleDict):
            parent[name] = replacement
        else:
            setattr(parent, name, replacement)

    def install_trainable_layer(self, block: nn.Module, path: str, trainable: TrainableFactorizedLinear) -> None:
        self._replace(block, path, trainable)

    def install_frozen_layer(self, block: nn.Module, path: str, frozen: FrozenReferenceLinear) -> None:
        self._replace(block, path, frozen)


def freeze_block_auxiliary_parameters(
    block: nn.Module, tensors: TensorStore
) -> tuple[tuple[str, TensorRef], ...]:
    """Persist the named parameters left after every quantized linear is frozen."""
    values = {name: parameter.detach() for name, parameter in block.named_parameters()}
    if not values:
        return ()
    references = tensors.put("frozen-block-auxiliary", values)
    return tuple((name, references[name]) for name in sorted(references))


def restore_block_auxiliary_parameters(
    block: nn.Module,
    parameters: tuple[tuple[str, TensorRef], ...],
    tensors: TensorStore,
    *,
    device: str,
) -> None:
    """Restore durable block-local parameters by their stable module names."""
    available = dict(block.named_parameters())
    expected = {name for name, _reference in parameters}
    missing = sorted(expected - set(available))
    if missing:
        raise ValueError(f"frozen block auxiliary parameters are absent from the model: {missing}")
    with torch.no_grad():
        for name, reference in parameters:
            target = available[name]
            with tensors.read(reference, device) as value:
                if value.shape != target.shape:
                    raise ValueError(f"frozen block auxiliary parameter shape differs: {name}")
                target.copy_(value.to(dtype=target.dtype))
