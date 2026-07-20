"""Explicit trainable/frozen research linears and stage-owned block editing."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from nanoquant.domain.linear_math import (
    functional_dense_reconstruction,
    functional_factorized_linear,
    mask_outlier_columns,
)
from nanoquant.domain.models import (
    BlockId,
    FrozenNanoQuantState,
    FrozenOutlierState,
    FrozenSharedInputGroupState,
    LayerId,
    ScaleState,
    SharedInputMemberSlice,
    TensorRef,
)
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
    patch_left: nn.Parameter | None
    patch_right: nn.Parameter | None
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
        patch_left: torch.Tensor | None = None,
        patch_right: torch.Tensor | None = None,
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
        if (patch_left is None) != (patch_right is None):
            raise ValueError("low-rank patch tensors must be provided together")
        if patch_left is not None and patch_right is not None:
            if patch_left.ndim != 2 or patch_right.ndim != 2:
                raise ValueError("low-rank patch tensors must be matrices")
            if patch_left.shape[0] != left_latent.shape[0] or patch_right.shape[1] != right_latent.shape[1]:
                raise ValueError("low-rank patch dimensions differ from the linear")
            if patch_left.shape[1] != patch_right.shape[0]:
                raise ValueError("low-rank patch ranks do not match")
        self.patch_left = None if patch_left is None else nn.Parameter(patch_left.detach().clone())
        self.patch_right = None if patch_right is None else nn.Parameter(patch_right.detach().clone())
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
            self.patch_left,
            self.patch_right,
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
            self.patch_left,
            self.patch_right,
        )


class TrainableSharedInputFactorGroup(TrainableFactorizedLinear):
    """One parameter owner for a row-stacked collection of projections."""


class SharedInputProjectionView(nn.Module):
    """Parameter-free logical projection backed by a registered group owner."""

    def __init__(self, owner: nn.Module, row_start: int, row_end: int, in_features: int) -> None:
        super().__init__()
        if row_start < 0 or row_end <= row_start or in_features <= 0:
            raise ValueError("invalid shared-input projection slice")
        object.__setattr__(self, "_owner_reference", weakref.ref(owner))
        self.row_start = row_start
        self.row_end = row_end
        self.in_features = in_features
        self.out_features = row_end - row_start

    @property
    def owner(self) -> nn.Module:
        reference = cast(weakref.ReferenceType[nn.Module], self._owner_reference)
        owner = reference()
        if not isinstance(owner, nn.Module):
            raise RuntimeError("shared-input group owner is no longer available")
        return owner

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.owner(value)
        if not isinstance(output, torch.Tensor) or output.shape[-1] < self.row_end:
            raise RuntimeError("shared-input group output does not cover its member slice")
        return output[..., self.row_start : self.row_end]


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
    patch_left: torch.Tensor | None
    patch_right: torch.Tensor | None
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
        patch_left: torch.Tensor | None = None,
        patch_right: torch.Tensor | None = None,
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
        if (patch_left is None) != (patch_right is None):
            raise ValueError("low-rank patch tensors must be provided together")
        self.register_buffer("patch_left", None if patch_left is None else patch_left.detach().clone())
        self.register_buffer("patch_right", None if patch_right is None else patch_right.detach().clone())
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
            self.patch_left,
            self.patch_right,
        )

    def dense_weight(self) -> torch.Tensor:
        return self._materialize_dense_weight() if self._cached_dense_weight is None else self._cached_dense_weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.dense_weight(), self.bias)


class DenseWeightReferenceLinear(FrozenReferenceLinear):
    """Dense replay module that does not retain the factors used to reconstruct its weight."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        nn.Module.__init__(self)
        # The reconstruction result is already a new immutable tensor. Register it
        # directly so loading a full model never holds a second dense clone.
        self.register_buffer("_cached_dense_weight", weight.detach(), persistent=False)
        self.register_buffer("bias", None if bias is None else bias.detach())

    def dense_weight(self) -> torch.Tensor:
        if self._cached_dense_weight is None:
            raise AssertionError("compact dense reference is missing its materialized weight")
        return self._cached_dense_weight

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
        patch_left: torch.Tensor | None = None,
        patch_right: torch.Tensor | None = None,
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
            patch_left,
            patch_right,
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
            self.patch_left,
            self.patch_right,
            scale_left_before_linear=True,
        )


@dataclass(frozen=True, slots=True)
class FrozenLayer:
    state: FrozenNanoQuantState
    module: FrozenReferenceLinear


@dataclass(frozen=True, slots=True)
class FrozenSharedInputGroup:
    state: FrozenSharedInputGroupState
    owner: FrozenReferenceLinear
    views: tuple[tuple[LayerId, SharedInputProjectionView], ...]


class LayerFreezer:
    def freeze(
        self,
        layer: LayerId,
        trainable: TrainableFactorizedLinear,
        tensors: TensorStore,
        logical_format: str = "nanoquant-v1",
        outliers: FrozenOutlierState | None = None,
        backend: str = "dense",
        bias_storage_dtype: torch.dtype | None = None,
        patch_storage_dtype: torch.dtype | None = None,
    ) -> FrozenLayer:
        left = torch.where(trainable.left_latent.detach() >= 0, 1.0, -1.0)
        right = torch.where(trainable.right_latent.detach() >= 0, 1.0, -1.0)
        values: dict[str, torch.Tensor] = {
            "left_binary": left,
            "right_binary": right,
            "scale_pre": mask_outlier_columns(trainable.scale_pre.detach(), trainable.outlier_indices),
            "scale_mid": trainable.scale_mid.detach(),
            "scale_post": trainable.scale_post.detach(),
        }
        if trainable.bias is not None:
            values["bias"] = trainable.bias.detach().to(bias_storage_dtype or trainable.bias.dtype)
        if trainable.patch_left is not None and trainable.patch_right is not None:
            values["patch_left"] = trainable.patch_left.detach().to(
                patch_storage_dtype or trainable.patch_left.dtype
            )
            values["patch_right"] = trainable.patch_right.detach().to(
                patch_storage_dtype or trainable.patch_right.dtype
            )
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
            refs.get("patch_left"),
            refs.get("patch_right"),
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
        compact_dense: bool = False,
    ) -> FrozenLayer:
        if backend not in {"dense", "factorized"}:
            raise ValueError(f"unsupported frozen reference backend: {backend}")
        if compact_dense and backend != "dense":
            raise ValueError("compact dense storage requires the dense backend")
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
            patch_left = patch_right = None
            if state.patch_left is not None and state.patch_right is not None:
                with (
                    tensors.read(state.patch_left, device) as left_value,
                    tensors.read(state.patch_right, device) as right_value,
                ):
                    patch_left = left_value.clone()
                    patch_right = right_value.clone()
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
            module: FrozenReferenceLinear
            if compact_dense:
                module = DenseWeightReferenceLinear(
                    functional_dense_reconstruction(
                        left,
                        right,
                        scale_pre,
                        scale_mid,
                        scale_post,
                        outlier_indices,
                        outlier_values,
                        outlier_scales,
                        patch_left,
                        patch_right,
                    ),
                    bias,
                )
            else:
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
                    patch_left,
                    patch_right,
                )
        if dtype is not None:
            module = module.to(dtype=dtype)
        return FrozenLayer(state, module)


class SharedInputGroupFreezer:
    def freeze(
        self,
        block: LayerId | tuple[LayerId, ...],
        name: str,
        member_widths: tuple[int, ...],
        trainable: TrainableSharedInputFactorGroup,
        tensors: TensorStore,
        logical_format: str = "nanoquant-v1",
        backend: str = "factorized",
        bias_storage_dtype: torch.dtype | None = None,
    ) -> FrozenSharedInputGroup:
        members = (block,) if isinstance(block, LayerId) else block
        if len(members) != len(member_widths) or len(members) < 2:
            raise ValueError("shared-input freeze requires aligned members and widths")
        if len({member.block for member in members}) != 1:
            raise ValueError("shared-input freeze members must belong to one block")
        if sum(member_widths) != trainable.left_latent.shape[0]:
            raise ValueError("shared-input member widths do not cover the stacked output")
        left = torch.where(trainable.left_latent.detach() >= 0, 1.0, -1.0)
        right = torch.where(trainable.right_latent.detach() >= 0, 1.0, -1.0)
        values: dict[str, torch.Tensor] = {
            "left_binary": left,
            "right_binary": right,
            "scale_pre": mask_outlier_columns(trainable.scale_pre.detach(), trainable.outlier_indices),
            "scale_mid": trainable.scale_mid.detach(),
            "scale_post": trainable.scale_post.detach(),
        }
        if trainable.bias is not None:
            values["bias"] = trainable.bias.detach().to(bias_storage_dtype or trainable.bias.dtype)
        if trainable.outlier_indices is not None and trainable.outlier_values is not None:
            values["outlier_indices"] = trainable.outlier_indices.detach()
            values["outlier_values"] = trainable.outlier_values.detach()
            if trainable.outlier_scales is not None:
                values["outlier_scales"] = trainable.outlier_scales.detach()
        refs = tensors.put("frozen-shared-input-group", values)
        outliers = (
            None
            if "outlier_indices" not in refs
            else FrozenOutlierState(refs["outlier_indices"], refs["outlier_values"], refs.get("outlier_scales"))
        )
        cursor = 0
        slices = []
        for member, width in zip(members, member_widths, strict=True):
            slices.append(SharedInputMemberSlice(member, cursor, cursor + width))
            cursor += width
        state = FrozenSharedInputGroupState(
            members[0].block,
            name,
            tuple(slices),
            left.shape[1],
            refs["left_binary"],
            refs["right_binary"],
            ScaleState(refs["scale_pre"], refs["scale_mid"], refs["scale_post"]),
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
        state: FrozenSharedInputGroupState,
        tensors: TensorStore,
        *,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
        backend: str = "factorized",
    ) -> FrozenSharedInputGroup:
        if backend not in {"dense", "factorized"}:
            raise ValueError(f"unsupported shared-input backend: {backend}")
        if state.scales.mid is None:
            raise ValueError("frozen shared-input group is missing its mid scale")
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
            indices = values = scales = None
            if state.outliers is not None:
                with (
                    tensors.read(state.outliers.indices, device) as value_indices,
                    tensors.read(state.outliers.values, device) as value_values,
                ):
                    indices = value_indices.clone()
                    values = value_values.clone()
                if state.outliers.scales is not None:
                    with tensors.read(state.outliers.scales, device) as value_scales:
                        scales = value_scales.clone()
            module_type = FrozenReferenceLinear if backend == "dense" else FactorizedReferenceLinear
            owner = module_type(left, right, scale_pre, scale_mid, scale_post, bias, indices, values, scales)
        if dtype is not None:
            owner = owner.to(dtype=dtype)
        views = tuple(
            (
                member.layer,
                SharedInputProjectionView(owner, member.row_start, member.row_end, owner.scale_pre.numel()),
            )
            for member in state.members
        )
        return FrozenSharedInputGroup(state, owner, views)


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
        if not isinstance(
            existing,
            (nn.Linear, TrainableFactorizedLinear, FrozenReferenceLinear, SharedInputProjectionView),
        ):
            raise TypeError(f"target is not a replaceable linear: {path}")
        if isinstance(parent, nn.ModuleDict):
            parent[name] = replacement
        else:
            setattr(parent, name, replacement)

    def install_trainable_layer(self, block: nn.Module, path: str, trainable: TrainableFactorizedLinear) -> None:
        self._replace(block, path, trainable)

    def install_frozen_layer(self, block: nn.Module, path: str, frozen: FrozenReferenceLinear) -> None:
        self._replace(block, path, frozen)

    @staticmethod
    def _group_key(name: str) -> str:
        key = name.replace(".", "__")
        if not key or key.startswith("_"):
            raise ValueError(f"invalid shared-input group name: {name!r}")
        return key

    def _install_group(
        self,
        block: nn.Module,
        name: str,
        owner: nn.Module,
        views: tuple[tuple[LayerId, SharedInputProjectionView], ...],
    ) -> None:
        registry = getattr(block, "_nanoquant_shared_input_groups", None)
        if registry is None:
            registry = nn.ModuleDict()
            block.add_module("_nanoquant_shared_input_groups", registry)
        if not isinstance(registry, nn.ModuleDict):
            raise TypeError("shared-input group registry has an incompatible type")
        key = self._group_key(name)
        if key in registry:
            del registry[key]
        registry[key] = owner
        for layer, view in views:
            self._replace(block, layer.path, view)

    def install_trainable_group(
        self,
        block: nn.Module,
        name: str,
        members: tuple[LayerId, ...],
        member_widths: tuple[int, ...],
        owner: TrainableSharedInputFactorGroup,
    ) -> None:
        if len(members) != len(member_widths):
            raise ValueError("shared-input trainable members and widths differ")
        cursor = 0
        views = []
        for member, width in zip(members, member_widths, strict=True):
            views.append((member, SharedInputProjectionView(owner, cursor, cursor + width, owner.scale_pre.numel())))
            cursor += width
        if cursor != owner.scale_post.numel():
            raise ValueError("shared-input trainable slices do not cover the group output")
        self._install_group(block, name, owner, tuple(views))

    def install_frozen_group(
        self,
        block: nn.Module,
        frozen: FrozenSharedInputGroup,
    ) -> None:
        self._install_group(block, frozen.state.name, frozen.owner, frozen.views)

    def install_runtime_group(
        self,
        block: nn.Module,
        name: str,
        block_index: int,
        members: tuple[tuple[str, int, int], ...],
        owner: FrozenReferenceLinear,
    ) -> None:
        views = tuple(
            (
                LayerId(BlockId(block_index), path),
                SharedInputProjectionView(owner, row_start, row_end, owner.scale_pre.numel()),
            )
            for path, row_start, row_end in members
        )
        self._install_group(block, name, owner, views)


def freeze_block_auxiliary_parameters(block: nn.Module, tensors: TensorStore) -> tuple[tuple[str, TensorRef], ...]:
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
