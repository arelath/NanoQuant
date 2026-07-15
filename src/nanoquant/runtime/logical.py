"""Validated backend-independent logical NanoQuant layer state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nanoquant.runtime.backend import QuantizedLinearSpec


def canonical_torch_dtype(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported runtime tensor dtype: {dtype}") from error


@dataclass(frozen=True, slots=True)
class LogicalLayerState:
    spec: QuantizedLinearSpec
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    bias: torch.Tensor | None = None
    outlier_indices: torch.Tensor | None = None
    outlier_values: torch.Tensor | None = None
    outlier_scales: torch.Tensor | None = None

    def __post_init__(self) -> None:
        spec = self.spec
        expected_shapes = {
            "left_binary": (spec.out_features, spec.rank),
            "right_binary": (spec.rank, spec.in_features),
            "scale_pre": (spec.in_features,),
            "scale_mid": (spec.rank,),
            "scale_post": (spec.out_features,),
        }
        values = {
            "left_binary": self.left_binary,
            "right_binary": self.right_binary,
            "scale_pre": self.scale_pre,
            "scale_mid": self.scale_mid,
            "scale_post": self.scale_post,
        }
        for name, expected in expected_shapes.items():
            value = values[name]
            if tuple(value.shape) != expected:
                raise ValueError(f"runtime tensor {name} has shape {tuple(value.shape)}, expected {expected}")
            if not value.is_contiguous():
                raise ValueError(f"runtime tensor {name} must be contiguous")
        if canonical_torch_dtype(self.left_binary.dtype) != spec.factor_dtype:
            raise ValueError("runtime left factor dtype differs from its specification")
        if self.right_binary.dtype != self.left_binary.dtype:
            raise ValueError("runtime factor dtypes differ")
        if any(canonical_torch_dtype(value.dtype) != spec.scale_dtype for value in values.values() if value.ndim == 1):
            raise ValueError("runtime scale dtype differs from its specification")
        scales = (self.scale_pre, self.scale_mid, self.scale_post)
        if any(not bool(torch.all(torch.isfinite(value))) for value in scales):
            raise ValueError("runtime scale contains a non-finite value")
        if not bool(torch.all((self.left_binary == 1) | (self.left_binary == -1))):
            raise ValueError("runtime left factor contains a value other than -1 or +1")
        if not bool(torch.all((self.right_binary == 1) | (self.right_binary == -1))):
            raise ValueError("runtime right factor contains a value other than -1 or +1")
        if (self.bias is None) != (not spec.has_bias):
            raise ValueError("runtime bias presence differs from its specification")
        if self.bias is not None and tuple(self.bias.shape) != (spec.out_features,):
            raise ValueError("runtime bias shape differs from the output dimension")
        if self.bias is not None and canonical_torch_dtype(self.bias.dtype) != spec.scale_dtype:
            raise ValueError("runtime bias dtype differs from the scale dtype")
        if self.bias is not None and not bool(torch.all(torch.isfinite(self.bias))):
            raise ValueError("runtime bias contains a non-finite value")
        has_outliers = self.outlier_indices is not None or self.outlier_values is not None
        if has_outliers != (spec.outlier_count > 0):
            raise ValueError("runtime outlier presence differs from its specification")
        if (self.outlier_indices is None) != (self.outlier_values is None):
            raise ValueError("runtime outlier indices and values must be provided together")
        if self.outlier_indices is not None and self.outlier_values is not None:
            if tuple(self.outlier_indices.shape) != (spec.outlier_count,):
                raise ValueError("runtime outlier index shape differs from its specification")
            if tuple(self.outlier_values.shape) != (spec.out_features, spec.outlier_count):
                raise ValueError("runtime outlier value shape differs from its specification")
            if self.outlier_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("runtime outlier indices must be int32 or int64")
            indexes = self.outlier_indices.to(dtype=torch.int64)
            if bool(torch.any(indexes < 0)) or bool(torch.any(indexes >= spec.in_features)):
                raise ValueError("runtime outlier index is outside the input dimension")
            if not bool(torch.all(indexes[1:] > indexes[:-1])):
                raise ValueError("runtime outlier indices must be strictly increasing")
            if canonical_torch_dtype(self.outlier_values.dtype) != spec.outlier_value_dtype:
                raise ValueError("runtime outlier value dtype differs from its specification")
            if self.outlier_values.is_floating_point() and not bool(torch.all(torch.isfinite(self.outlier_values))):
                raise ValueError("runtime outlier value contains a non-finite value")
        if (self.outlier_scales is not None) != spec.has_outlier_scales:
            raise ValueError("runtime outlier scale presence differs from its specification")
        if self.outlier_scales is not None and tuple(self.outlier_scales.shape) not in {
            (),
            (spec.outlier_count,),
            (spec.out_features, 1),
            (spec.out_features, spec.outlier_count),
        }:
            raise ValueError("runtime outlier scale shape is not broadcastable to outlier values")
        if self.outlier_scales is not None and canonical_torch_dtype(self.outlier_scales.dtype) != spec.scale_dtype:
            raise ValueError("runtime outlier scale dtype differs from the scale dtype")
        if self.outlier_scales is not None and not bool(torch.all(torch.isfinite(self.outlier_scales))):
            raise ValueError("runtime outlier scale contains a non-finite value")
