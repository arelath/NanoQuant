"""Explicit trainable/frozen research linears and stage-owned block editing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from nanoquant.domain.models import FrozenNanoQuantState, LayerId, ScaleState
from nanoquant.ports.tensor_store import TensorStore


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (gradient,)


class TrainableFactorizedLinear(nn.Module):
    def __init__(
        self,
        left_latent: torch.Tensor,
        right_latent: torch.Tensor,
        scale_pre: torch.Tensor,
        scale_mid: torch.Tensor,
        scale_post: torch.Tensor,
        bias: torch.Tensor | None = None,
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

    def dense_weight(self) -> torch.Tensor:
        apply_sign = cast(Any, _SignSTE).apply
        left = cast(torch.Tensor, apply_sign(self.left_latent)) * self.scale_post.reshape(-1, 1)
        right = (
            cast(torch.Tensor, apply_sign(self.right_latent))
            * self.scale_mid.reshape(-1, 1)
            * self.scale_pre.reshape(1, -1)
        )
        return left @ right

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.dense_weight(), self.bias)


class FrozenReferenceLinear(nn.Module):
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        left_binary: torch.Tensor,
        right_binary: torch.Tensor,
        scale_pre: torch.Tensor,
        scale_mid: torch.Tensor,
        scale_post: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("left_binary", left_binary.detach().clone())
        self.register_buffer("right_binary", right_binary.detach().clone())
        self.register_buffer("scale_pre", scale_pre.detach().clone().reshape(-1))
        self.register_buffer("scale_mid", scale_mid.detach().clone().reshape(-1))
        self.register_buffer("scale_post", scale_post.detach().clone().reshape(-1))
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    def dense_weight(self) -> torch.Tensor:
        return (self.left_binary * self.scale_post.reshape(-1, 1)) @ (
            self.right_binary * self.scale_mid.reshape(-1, 1) * self.scale_pre.reshape(1, -1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.dense_weight(), self.bias)


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
    ) -> FrozenLayer:
        left = torch.where(trainable.left_latent.detach() >= 0, 1.0, -1.0)
        right = torch.where(trainable.right_latent.detach() >= 0, 1.0, -1.0)
        values = {
            "left_binary": left,
            "right_binary": right,
            "scale_pre": trainable.scale_pre.detach(),
            "scale_mid": trainable.scale_mid.detach(),
            "scale_post": trainable.scale_post.detach(),
        }
        if trainable.bias is not None:
            values["bias"] = trainable.bias.detach()
        refs = tensors.put("frozen-layer", values)
        scales = ScaleState(refs["scale_pre"], refs["scale_mid"], refs["scale_post"])
        state = FrozenNanoQuantState(
            layer,
            left.shape[1],
            refs["left_binary"],
            refs["right_binary"],
            scales,
            None,
            refs.get("bias"),
            logical_format,
        )
        module = FrozenReferenceLinear(
            left, right, trainable.scale_pre, trainable.scale_mid, trainable.scale_post, trainable.bias
        )
        return FrozenLayer(state, module)

    def load(
        self,
        state: FrozenNanoQuantState,
        tensors: TensorStore,
        *,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> FrozenLayer:
        if state.outliers is not None:
            raise NotImplementedError("frozen reference loading does not yet support outliers")
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
            module = FrozenReferenceLinear(left, right, scale_pre, scale_mid, scale_post, bias)
        if dtype is not None:
            module = module.to(dtype=dtype)
        return FrozenLayer(state, module)


class BlockEditor:
    def install_frozen_layer(self, block: nn.Module, path: str, frozen: FrozenReferenceLinear) -> None:
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
        if not isinstance(existing, nn.Linear) and not isinstance(existing, TrainableFactorizedLinear):
            raise TypeError(f"target is not a replaceable linear: {path}")
        if isinstance(parent, nn.ModuleDict):
            parent[name] = frozen
        else:
            setattr(parent, name, frozen)
