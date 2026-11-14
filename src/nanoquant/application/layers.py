"""Explicit trainable/frozen research linears and stage-owned block editing."""

from __future__ import annotations

import math
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nanoquant.domain.linear_math import (
    functional_dense_reconstruction,
    functional_factorized_linear,
    mask_outlier_columns,
)
from nanoquant.domain.models import (
    BlockId,
    FrozenNanoQuantState,
    FrozenOutlierState,
    FrozenResidualFactorState,
    FrozenSharedInputGroupState,
    LayerId,
    ProductCodebookFactorState,
    ScaleState,
    SharedInputMemberSlice,
    TensorRef,
    TrainableFactors,
)
from nanoquant.domain.outliers import quantize_int8_columns
from nanoquant.domain.packing import pack_sign_bits, unpack_sign_bits
from nanoquant.domain.sign_word_codebook import ProductSignCodebook, decode_product_codebook
from nanoquant.ports.tensor_store import TensorStore


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        return (value >= 0).to(dtype=value.dtype).mul_(2).sub_(1)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (gradient,)


class TrainableFactorizedLinear(nn.Module):
    residual_stages: nn.ModuleList
    coupled_shared_input_scales = False
    outlier_indices: torch.Tensor | None
    outlier_values: nn.Parameter | None
    outlier_scales: torch.Tensor | None
    patch_left: nn.Parameter | None
    patch_right: nn.Parameter | None
    immutable_binary_factors: bool
    product_codebook_free_rows: int | None
    outlier_storage_dtype: torch.dtype | None

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
        product_codebook_free_rows: int | None = None,
        residual_stages: tuple[TrainableFactorizedLinear, ...] = (),
    ) -> None:
        super().__init__()
        if left_latent.shape[1] != right_latent.shape[0]:
            raise ValueError("factor ranks do not match")
        self.left_latent = nn.Parameter(left_latent.detach().clone())
        self.right_latent = nn.Parameter(right_latent.detach().clone())
        if product_codebook_free_rows is not None and not (
            0 <= product_codebook_free_rows < right_latent.shape[0]
        ):
            raise ValueError("product-codebook free rows must leave a coded suffix")
        self.product_codebook_free_rows = product_codebook_free_rows
        self.register_buffer(
            "_fixed_right_suffix",
            (
                None
                if product_codebook_free_rows is None
                else right_latent.detach()[product_codebook_free_rows:].clone()
            ),
        )
        self.scale_pre = nn.Parameter(scale_pre.detach().clone().reshape(-1))
        self.scale_mid = nn.Parameter(scale_mid.detach().clone().reshape(-1))
        self.scale_post = nn.Parameter(scale_post.detach().clone().reshape(-1))
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone())
        self.register_buffer("outlier_indices", None if outlier_indices is None else outlier_indices.detach().clone())
        if (outlier_indices is None) != (outlier_values is None):
            raise ValueError("outlier indices and values must be provided together")
        self.outlier_storage_dtype = None if outlier_values is None else outlier_values.dtype
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
        self._packed_factor_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self.checkpoint_fixed_signs = True
        if any(stage.residual_stages or stage.bias is not None or stage.outlier_indices is not None
               or stage.patch_left is not None or stage.product_codebook_free_rows is not None
               for stage in residual_stages):
            raise ValueError("residual stages must contain only independent factor terms")
        if any(stage.left_latent.shape[0] != left_latent.shape[0]
               or stage.right_latent.shape[1] != right_latent.shape[1] for stage in residual_stages):
            raise ValueError("residual stage dimensions differ from the owning linear")
        self.residual_stages = nn.ModuleList(residual_stages)
        for stage in self.residual_terms:
            stage.coupled_shared_input_scales = self.coupled_shared_input_scales

    @property
    def residual_terms(self) -> tuple[TrainableFactorizedLinear, ...]:
        return tuple(cast(TrainableFactorizedLinear, stage) for stage in self.residual_stages)

    def effective_right_latent(self) -> torch.Tensor:
        """Return the trainable free prefix joined to the immutable coded suffix."""

        free_rows = self.product_codebook_free_rows
        if free_rows is None:
            return self.right_latent
        if self._fixed_right_suffix is None:
            raise AssertionError("product-codebook module is missing its fixed suffix")
        return torch.cat(
            (self.right_latent[:free_rows], cast(torch.Tensor, self._fixed_right_suffix)),
            dim=0,
        )

    def dense_weight(self) -> torch.Tensor:
        apply_sign = cast(Any, _SignSTE).apply
        left, right = (
            self._decode_fixed_signs(self.scale_pre.dtype)
            if self._packed_factor_shapes is not None
            else (cast(torch.Tensor, apply_sign(self.left_latent)),
                  cast(torch.Tensor, apply_sign(self.effective_right_latent())))
        )
        weight = functional_dense_reconstruction(
            left,
            right,
            self.scale_pre,
            self.scale_mid,
            self.scale_post,
            self.outlier_indices,
            self.outlier_values,
            self.outlier_scales,
            self.patch_left,
            self.patch_right,
        )
        for stage in self.residual_terms:
            weight = weight + stage.dense_weight()
        return weight

    def pack_fixed_signs(self) -> None:
        """Use transient decoded signs for scale-only training, without changing parameter names for scales."""
        if not self.immutable_binary_factors or self.product_codebook_free_rows is not None:
            raise ValueError("compact training requires immutable plain binary factors")
        if self._packed_factor_shapes is None:
            left, left_shape = pack_sign_bits(self.left_latent)
            right, right_shape = pack_sign_bits(self.right_latent)
            del self.left_latent, self.right_latent
            self.register_buffer("left_latent", left, persistent=False)
            self.register_buffer("right_latent", right, persistent=False)
            self._packed_factor_shapes = left_shape, right_shape

    def fixed_sign_residency_bytes(self) -> int:
        """Return the extra memory needed to keep this term's packed signs decoded."""
        shapes = self._packed_factor_shapes
        if shapes is None:
            return 0
        decoded = sum(math.prod(shape) for shape in shapes) * self.scale_pre.element_size()
        return decoded - self.left_latent.numel() - self.right_latent.numel()

    def unpack_fixed_signs(self) -> None:
        """Keep decoded signs resident, trading memory for the decode in every forward."""
        if self._packed_factor_shapes is not None:
            left, right = self._decode_fixed_signs(self.scale_pre.dtype)
            self.register_buffer("left_latent", left, persistent=False)
            self.register_buffer("right_latent", right, persistent=False)
            self._packed_factor_shapes = None

    def _decode_fixed_signs(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        shapes = self._packed_factor_shapes
        if shapes is None:
            raise ValueError("no packed sign factors")
        return (unpack_sign_bits(self.left_latent, shapes[0], dtype=dtype),
                unpack_sign_bits(self.right_latent, shapes[1], dtype=dtype))

    def export_signs(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._packed_factor_shapes is not None:
            return self._decode_fixed_signs(torch.float32)
        return (torch.where(self.left_latent.detach() >= 0, 1.0, -1.0),
                torch.where(self.effective_right_latent().detach() >= 0, 1.0, -1.0))

    def _forward_term(self, value: torch.Tensor) -> torch.Tensor:
        if self._packed_factor_shapes is not None:
            left, right = self._decode_fixed_signs(value.dtype)
        else:
            left, right = self._materialize_forward_signs(value)
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

    def _materialize_forward_signs(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        apply_sign = cast(Any, _SignSTE).apply
        right = (
            self.effective_right_latent()
            if self.immutable_binary_factors and not self.right_latent.requires_grad
            else cast(torch.Tensor, apply_sign(self.effective_right_latent()))
        ).to(device=value.device, dtype=value.dtype)
        left = (
            self.left_latent
            if self.immutable_binary_factors and not self.left_latent.requires_grad
            else cast(torch.Tensor, apply_sign(self.left_latent))
        ).to(device=value.device, dtype=value.dtype)
        return left, right

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Autograd would otherwise retain each decoded matrix until backward.
        output = (
            cast(torch.Tensor, checkpoint(self._forward_term, value, use_reentrant=False, preserve_rng_state=False))
            if self._packed_factor_shapes is not None and self.checkpoint_fixed_signs and torch.is_grad_enabled()
            else self._forward_term(value)
        )
        for stage in self.residual_terms:
            output = output + stage(value)
        return output


class TrainableSharedInputFactorGroup(TrainableFactorizedLinear):
    """One parameter owner for a row-stacked collection of projections."""

    coupled_shared_input_scales = True


class SharedGroupEvaluation:
    """One owner evaluation shared by every member view that receives the same input.

    Members of a group are called back to back with one input tensor. Without
    sharing, each member re-evaluates the whole stacked owner and autograd
    back-propagates through it once per member. Sharing leaves every forward
    value bitwise unchanged; parameter gradients remain the same sum but are
    accumulated in one backward pass, so they differ in floating-point rounding.
    """

    def __init__(self, members: int) -> None:
        if members <= 0:
            raise ValueError("a shared group evaluation needs at least one member")
        self.members = members
        self._input: torch.Tensor | None = None
        self._input_version = 0
        self._gradient_mode = False
        self._output: torch.Tensor | None = None
        self._remaining = 0

    def __call__(self, owner: nn.Module, value: torch.Tensor) -> torch.Tensor:
        output = self._output
        if (
            output is not None
            and self._input is value
            and self._input_version == value._version
            and self._gradient_mode == torch.is_grad_enabled()
        ):
            self._remaining -= 1
        else:
            output = owner(value)
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("shared-input group owner did not return a tensor")
            # Holding the input keeps its identity from being reused by another tensor.
            self._input, self._input_version = value, value._version
            self._gradient_mode = torch.is_grad_enabled()
            self._output = output
            self._remaining = self.members - 1
        if self._remaining <= 0:
            # The last member has its slice; keep neither activations nor an autograd graph alive.
            self._input = self._output = None
        return output


class SharedInputProjectionView(nn.Module):
    """Parameter-free logical projection backed by a registered group owner."""

    def __init__(
        self,
        owner: nn.Module,
        row_start: int,
        row_end: int,
        in_features: int,
        evaluation: SharedGroupEvaluation | None = None,
    ) -> None:
        super().__init__()
        if row_start < 0 or row_end <= row_start or in_features <= 0:
            raise ValueError("invalid shared-input projection slice")
        object.__setattr__(self, "_owner_reference", weakref.ref(owner))
        self._evaluation = evaluation
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
        output = self.owner(value) if self._evaluation is None else self._evaluation(self.owner, value)
        if not isinstance(output, torch.Tensor) or output.shape[-1] < self.row_end:
            raise RuntimeError("shared-input group output does not cover its member slice")
        return output[..., self.row_start : self.row_end]


class FrozenReferenceLinear(nn.Module):
    residual_stages: nn.ModuleList
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
        residual_stages: tuple[FrozenReferenceLinear, ...] = (),
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
        self.residual_stages = nn.ModuleList(residual_stages)
        self.register_buffer(
            "_cached_dense_weight",
            self._materialize_dense_weight() if cache_dense_weight else None,
            persistent=False,
        )

    @property
    def residual_terms(self) -> tuple[FrozenReferenceLinear, ...]:
        return tuple(cast(FrozenReferenceLinear, stage) for stage in self.residual_stages)

    def _materialize_dense_weight(self) -> torch.Tensor:
        weight = functional_dense_reconstruction(
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
        for stage in self.residual_terms:
            weight = weight + stage.dense_weight()
        return weight

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
        *,
        residual_stages: tuple[FrozenReferenceLinear, ...] = (),
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
            residual_stages=residual_stages,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = functional_factorized_linear(
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
        for stage in self.residual_terms:
            output = output + stage(value)
        return output


@dataclass(frozen=True, slots=True)
class FrozenLayer:
    state: FrozenNanoQuantState
    module: FrozenReferenceLinear


@dataclass(frozen=True, slots=True)
class FrozenSharedInputGroup:
    state: FrozenSharedInputGroupState
    owner: FrozenReferenceLinear
    views: tuple[tuple[LayerId, SharedInputProjectionView], ...]


def _freeze_residual_stages(
    trainable: TrainableFactorizedLinear, tensors: TensorStore,
) -> tuple[FrozenResidualFactorState, ...]:
    stages = []
    for stage in trainable.residual_terms:
        left, right = stage.export_signs()
        refs = tensors.put("frozen-residual-stage", {
            "left_binary": left,
            "right_binary": right,
            "scale_pre": stage.scale_pre.detach(),
            "scale_mid": stage.scale_mid.detach(),
            "scale_post": stage.scale_post.detach(),
        })
        stages.append(FrozenResidualFactorState(
            left.shape[1], refs["left_binary"], refs["right_binary"],
            ScaleState(refs["scale_pre"], refs["scale_mid"], refs["scale_post"]),
        ))
    return tuple(stages)


def load_residual_stage(
    state: FrozenResidualFactorState, tensors: TensorStore, *, device: str,
    trainable: bool = False, backend: str = "factorized",
) -> TrainableFactorizedLinear | FrozenReferenceLinear:
    if state.scales.mid is None:
        raise ValueError("residual stage is missing its component scales")
    with ExitStack() as stack:
        values = [stack.enter_context(tensors.read(ref, device)) for ref in (
            state.left_binary, state.right_binary, state.scales.pre, state.scales.mid, state.scales.post,
        )]
        if trainable:
            return TrainableFactorizedLinear(*values, immutable_binary_factors=True)
        module_type = FrozenReferenceLinear if backend == "dense" else FactorizedReferenceLinear
        return module_type(*values)


def thaw_residual_stages(module: FrozenReferenceLinear) -> tuple[TrainableFactorizedLinear, ...]:
    return tuple(TrainableFactorizedLinear(
        stage.left_binary, stage.right_binary, stage.scale_pre, stage.scale_mid, stage.scale_post,
        immutable_binary_factors=True,
    ) for stage in module.residual_terms)


def materialize_residual_modules(
    factors: TrainableFactors, tensors: TensorStore, *, device: str, tune_binary: bool,
) -> tuple[TrainableFactorizedLinear, ...]:
    modules = []
    for stage in factors.residual_stages:
        if stage.scales.mid is None or stage.residual_stages:
            raise ValueError("expected a flat inventory of independently scaled residual factors")
        with ExitStack() as stack:
            values = [stack.enter_context(tensors.read(ref, device)) for ref in (
                stage.left_latent if tune_binary else stage.left_binary,
                stage.right_latent if tune_binary else stage.right_binary,
                stage.scales.pre, stage.scales.mid, stage.scales.post,
            )]
            modules.append(TrainableFactorizedLinear(*values))
    return tuple(modules)


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
        product_codebook: ProductCodebookFactorState | None = None,
    ) -> FrozenLayer:
        left, right = trainable.export_signs()
        if product_codebook is not None:
            if trainable.product_codebook_free_rows != product_codebook.right_free_rows:
                raise ValueError("trainable and persisted product-codebook free rows differ")
            with (
                tensors.read(product_codebook.right_indices, str(right.device)) as right_indices,
                tensors.read(product_codebook.first_table, str(right.device)) as first_table,
                tensors.read(product_codebook.second_table, str(right.device)) as second_table,
            ):
                decoded = decode_product_codebook(
                    right_indices,
                    ProductSignCodebook(16, first_table, second_table),
                    right.shape[1],
                ).to(device=right.device, dtype=right.dtype)
            if not torch.equal(right[product_codebook.right_free_rows :], decoded):
                raise ValueError("frozen right-factor suffix differs from its product-codebook payload")
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
            outlier_values = trainable.outlier_values.detach()
            outlier_scales = trainable.outlier_scales
            if trainable.outlier_storage_dtype is torch.int8:
                outlier_values, outlier_scales = quantize_int8_columns(outlier_values)
                outlier_scales = outlier_scales.to(trainable.scale_pre.dtype)
            values["outlier_values"] = outlier_values
            if outlier_scales is not None:
                values["outlier_scales"] = outlier_scales.detach()
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
            "nanoquant-residual-v2" if trainable.residual_stages else logical_format,
            refs.get("patch_left"),
            refs.get("patch_right"),
            product_codebook,
            _freeze_residual_stages(trainable, tensors),
        )
        return self.load(
            state,
            tensors,
            device=str(trainable.scale_pre.device),
            dtype=(trainable.scale_pre.dtype if trainable._packed_factor_shapes is not None
                   else trainable.left_latent.dtype),
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
        residual_stages = tuple(cast(FrozenReferenceLinear, load_residual_stage(
            stage, tensors, device=device, backend=backend,
        )) for stage in state.residual_stages)
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
                weight = functional_dense_reconstruction(
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
                    )
                for stage in residual_stages:
                    weight = weight + stage.dense_weight()
                module = DenseWeightReferenceLinear(weight, bias)
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
                    residual_stages=residual_stages,
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
        left, right = trainable.export_signs()
        if sum(member_widths) != left.shape[0]:
            raise ValueError("shared-input member widths do not cover the stacked output")
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
            outlier_values = trainable.outlier_values.detach()
            outlier_scales = trainable.outlier_scales
            if trainable.outlier_storage_dtype is torch.int8:
                outlier_values, outlier_scales = quantize_int8_columns(outlier_values)
                outlier_scales = outlier_scales.to(trainable.scale_pre.dtype)
            values["outlier_values"] = outlier_values
            if outlier_scales is not None:
                values["outlier_scales"] = outlier_scales.detach()
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
            "nanoquant-residual-v2" if trainable.residual_stages else logical_format,
            _freeze_residual_stages(trainable, tensors),
        )
        return self.load(
            state,
            tensors,
            device=str(trainable.scale_pre.device),
            dtype=(trainable.scale_pre.dtype if trainable._packed_factor_shapes is not None
                   else trainable.left_latent.dtype),
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
            owner = module_type(
                left, right, scale_pre, scale_mid, scale_post, bias, indices, values, scales,
                residual_stages=tuple(cast(FrozenReferenceLinear, load_residual_stage(
                    stage, tensors, device=device, backend=backend,
                )) for stage in state.residual_stages),
            )
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
        *,
        share_member_evaluations: bool = False,
    ) -> None:
        """Install member views of one owner.

        ``share_member_evaluations`` evaluates the owner once per shared input
        instead of once per member. It is opt-in because it changes gradient
        rounding, which block tuning's resumable commits must not see.
        """
        if len(members) != len(member_widths):
            raise ValueError("shared-input trainable members and widths differ")
        evaluation = SharedGroupEvaluation(len(members)) if share_member_evaluations else None
        cursor = 0
        views = []
        for member, width in zip(members, member_widths, strict=True):
            view = SharedInputProjectionView(owner, cursor, cursor + width, owner.scale_pre.numel(), evaluation)
            views.append((member, view))
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
        *,
        in_features: int,
    ) -> None:
        if in_features <= 0:
            raise ValueError("shared-input runtime group input width must be positive")
        views = tuple(
            (
                LayerId(BlockId(block_index), path),
                SharedInputProjectionView(owner, row_start, row_end, in_features),
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
