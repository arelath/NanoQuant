"""Explicit trainable/frozen research linears and stage-owned block editing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from nanoquant.domain.models import FrozenNanoQuantState, FrozenOutlierState, LayerId, ScaleState
from nanoquant.ports.tensor_store import TensorStore


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (gradient,)


class TrainableFactorizedLinear(nn.Module):
    outlier_indices: torch.Tensor | None
    outlier_values: nn.Parameter | None
    outlier_scales: torch.Tensor | None

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

    def dense_weight(self) -> torch.Tensor:
        apply_sign = cast(Any, _SignSTE).apply
        left = cast(torch.Tensor, apply_sign(self.left_latent)) * self.scale_post.reshape(-1, 1)
        right = (
            cast(torch.Tensor, apply_sign(self.right_latent))
            * self.scale_mid.reshape(-1, 1)
            * self.scale_pre.reshape(1, -1)
        )
        result = left @ right
        if self.outlier_indices is not None and self.outlier_values is not None:
            outlier_values: torch.Tensor = self.outlier_values
            if self.outlier_scales is not None:
                outlier_values = outlier_values.float() * self.outlier_scales.float()
            result = result.clone()
            result[:, self.outlier_indices.long()] += outlier_values.to(result.dtype)
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.dense_weight(), self.bias)


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

    def dense_weight(self) -> torch.Tensor:
        result = (self.left_binary * self.scale_post.reshape(-1, 1)) @ (
            self.right_binary * self.scale_mid.reshape(-1, 1) * self.scale_pre.reshape(1, -1)
        )
        if self.outlier_indices is not None and self.outlier_values is not None:
            values = self.outlier_values
            if self.outlier_scales is not None:
                values = values.float() * self.outlier_scales.float()
            result = result.clone()
            result[:, self.outlier_indices.long()] += values.to(result.dtype)
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.dense_weight(), self.bias)


class FactorizedReferenceLinear(FrozenReferenceLinear):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        latent = torch.nn.functional.linear(value * self.scale_pre, self.right_binary)
        output = torch.nn.functional.linear(latent * self.scale_mid, self.left_binary * self.scale_post.reshape(-1, 1))
        if self.outlier_indices is not None and self.outlier_values is not None:
            outlier_values = self.outlier_values
            if self.outlier_scales is not None:
                outlier_values = outlier_values.float() * self.outlier_scales.float()
            output = output + torch.nn.functional.linear(
                value.index_select(-1, self.outlier_indices.long()), outlier_values.to(value.dtype)
            )
        if self.bias is not None:
            output = output + self.bias
        return output


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
    ) -> FrozenLayer:
        left = torch.where(trainable.left_latent.detach() >= 0, 1.0, -1.0)
        right = torch.where(trainable.right_latent.detach() >= 0, 1.0, -1.0)
        values: dict[str, torch.Tensor] = {
            "left_binary": left,
            "right_binary": right,
            "scale_pre": trainable.scale_pre.detach(),
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
