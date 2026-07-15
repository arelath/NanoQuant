"""Versioned CUDA backend for the llama.cpp-compatible packed NanoQuant layout."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch

from nanoquant.runtime.backend import (
    BackendCapabilities,
    DeviceLike,
    PreparedLayer,
    QuantizedLinearSpec,
    RuntimeLayerState,
    SupportResult,
    WorkloadSpec,
    evaluate_capabilities,
)
from nanoquant.runtime.logical import canonical_torch_dtype
from nanoquant.runtime.packed import PACKED_LAYOUT_VERSION, PackedLayerState

CUDA_PACKED_BACKEND_VERSION = "1"
CUDA_PACKED_REFERENCE_SHA256 = "5c87336c2b6b8fb33805c6ee6a8752d4bd364beed63fd4cca03c2b36be966619"
_FLOAT_DTYPES = ("float16", "bfloat16", "float32")


@dataclass(frozen=True, slots=True)
class _CudaPackedPayload:
    device: torch.device
    left_words: torch.Tensor
    right_words: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    bias: torch.Tensor | None
    outlier_indices: torch.Tensor | None
    outlier_values: torch.Tensor | None
    outlier_scales: torch.Tensor | None


def _copy_optional(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if value is None else value.to(device=device).contiguous()


class CudaPackedBackend:
    """Triton port of the pinned two-stage llama.cpp NanoQuant CUDA operation."""

    name = "cuda-packed-triton"
    version = CUDA_PACKED_BACKEND_VERSION
    packed_layout = PACKED_LAYOUT_VERSION
    reference_cuda_sha256 = CUDA_PACKED_REFERENCE_SHA256

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            logical_formats=("nanoquant-v1",),
            device_types=("cuda",),
            input_dtypes=_FLOAT_DTYPES,
            factor_dtypes=_FLOAT_DTYPES,
            scale_dtypes=_FLOAT_DTYPES,
            outlier_value_dtypes=(*_FLOAT_DTYPES, "int8"),
            workload_kinds=("prefill", "decode"),
            supports_bias=True,
            supports_outliers=True,
            supports_deterministic=True,
        )

    def supports(self, op: QuantizedLinearSpec, workload: WorkloadSpec) -> SupportResult:
        result = evaluate_capabilities(self.capabilities(), op, workload)
        if not result.supported:
            return result
        if importlib.util.find_spec("triton") is None:
            return SupportResult.rejected("NQ-INF-CUDA-KERNEL", "Triton is not installed")
        if not torch.cuda.is_available():
            return SupportResult.rejected("NQ-INF-DEVICE-UNAVAILABLE", "CUDA is not available in this runtime")
        capability = torch.cuda.get_device_capability()
        needs_bfloat16 = "bfloat16" in (
            workload.input_dtype,
            op.scale_dtype,
            op.outlier_value_dtype,
        )
        minimum = (8, 0) if needs_bfloat16 else (7, 0)
        if capability < minimum:
            return SupportResult.rejected(
                "NQ-INF-CUDA-CAPABILITY",
                f"CUDA compute capability {capability[0]}.{capability[1]} is below "
                f"{minimum[0]}.{minimum[1]}",
            )
        return result

    def prepare(self, state: RuntimeLayerState, device: DeviceLike) -> PreparedLayer:
        if not isinstance(state, PackedLayerState):
            raise TypeError("CUDA packed backend requires PackedLayerState")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("CUDA packed backend preparation requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this runtime")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        payload = _CudaPackedPayload(
            target,
            state.left_words.to(device=target).contiguous(),
            state.right_words.to(device=target).contiguous(),
            state.scale_pre.to(device=target).contiguous(),
            state.scale_mid.to(device=target).contiguous(),
            state.scale_post.to(device=target).contiguous(),
            _copy_optional(state.bias, target),
            _copy_optional(state.outlier_indices, target),
            _copy_optional(state.outlier_values, target),
            _copy_optional(state.outlier_scales, target),
        )
        return PreparedLayer(self.name, self.version, state.spec, payload)

    def linear(self, value: torch.Tensor, layer: PreparedLayer) -> torch.Tensor:
        if layer.backend_name != self.name or layer.backend_version != self.version:
            raise ValueError("prepared runtime layer belongs to a different backend")
        if not isinstance(layer.payload, _CudaPackedPayload):
            raise TypeError("prepared runtime layer has an invalid CUDA packed payload")
        payload = layer.payload
        if value.device != payload.device:
            raise ValueError("CUDA packed input and prepared layer must use the same device")
        if canonical_torch_dtype(value.dtype) not in _FLOAT_DTYPES:
            raise ValueError(f"CUDA packed input dtype is unsupported: {value.dtype}")
        if value.ndim == 0 or value.shape[-1] != layer.spec.in_features:
            raise ValueError("CUDA packed input feature dimension differs from the layer")
        if value.numel() == 0:
            raise ValueError("CUDA packed input must contain at least one token")
        if not value.is_contiguous():
            raise ValueError("CUDA packed input must be contiguous")
        if value.requires_grad:
            raise ValueError("CUDA packed runtime does not support autograd")
        from nanoquant.runtime.cuda_kernels import launch_packed_linear

        return launch_packed_linear(
            value,
            payload.left_words,
            payload.right_words,
            payload.scale_pre,
            payload.scale_mid,
            payload.scale_post,
            payload.bias,
            payload.outlier_indices,
            payload.outlier_values,
            payload.outlier_scales,
        )
