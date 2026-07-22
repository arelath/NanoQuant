"""Self-contained dense and factorized PyTorch runtime reference backends."""

from __future__ import annotations

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
from nanoquant.runtime.logical import LogicalLayerState
from nanoquant.runtime.packed import PackedLayerState

_FLOAT_DTYPES = ("float16", "bfloat16", "float32")


def _reference_capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        logical_formats=("nanoquant-v1",),
        device_types=("cpu", "cuda"),
        input_dtypes=_FLOAT_DTYPES,
        factor_dtypes=_FLOAT_DTYPES,
        scale_dtypes=_FLOAT_DTYPES,
        outlier_value_dtypes=(*_FLOAT_DTYPES, "int8", "int16"),
        workload_kinds=("prefill", "decode"),
        supports_bias=True,
        supports_outliers=True,
        supports_deterministic=True,
        patch_value_dtypes=_FLOAT_DTYPES,
    )


def _reference_support(op: QuantizedLinearSpec, workload: WorkloadSpec) -> SupportResult:
    result = evaluate_capabilities(_reference_capabilities(), op, workload)
    if not result.supported:
        return result
    if workload.device_type == "cuda" and not torch.cuda.is_available():
        return SupportResult.rejected("NQ-INF-DEVICE-UNAVAILABLE", "CUDA is not available in this runtime")
    return result


@dataclass(frozen=True, slots=True)
class _ReferencePayload:
    left: torch.Tensor
    right: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    bias: torch.Tensor | None
    outlier_indices: torch.Tensor | None
    outlier_values: torch.Tensor | None
    outlier_scales: torch.Tensor | None
    patch_left: torch.Tensor | None
    patch_right: torch.Tensor | None


def _prepare_payload(state: RuntimeLayerState, device: DeviceLike) -> _ReferencePayload:
    if not isinstance(state, LogicalLayerState):
        raise TypeError("PyTorch reference backends require LogicalLayerState")
    target = torch.device(device)
    return _ReferencePayload(
        state.left_binary.to(target),
        state.right_binary.to(target),
        state.scale_pre.to(target),
        state.scale_mid.to(target),
        state.scale_post.to(target),
        None if state.bias is None else state.bias.to(target),
        None if state.outlier_indices is None else state.outlier_indices.to(target),
        None if state.outlier_values is None else state.outlier_values.to(target),
        None if state.outlier_scales is None else state.outlier_scales.to(target),
        None if state.patch_left is None else state.patch_left.to(target),
        None if state.patch_right is None else state.patch_right.to(target),
    )


def _payload(layer: PreparedLayer, backend_name: str, backend_version: str) -> _ReferencePayload:
    if layer.backend_name != backend_name or layer.backend_version != backend_version:
        raise ValueError("prepared runtime layer belongs to a different backend")
    if not isinstance(layer.payload, _ReferencePayload):
        raise TypeError("prepared runtime layer has an invalid reference payload")
    return layer.payload


def _outlier_values(payload: _ReferencePayload) -> torch.Tensor | None:
    if payload.outlier_values is None:
        return None
    if payload.outlier_scales is None:
        return payload.outlier_values
    return payload.outlier_values.float() * payload.outlier_scales.float()


def _masked_scale_pre(payload: _ReferencePayload) -> torch.Tensor:
    if payload.outlier_indices is None or payload.outlier_indices.numel() == 0:
        return payload.scale_pre
    mask = torch.ones_like(payload.scale_pre)
    mask.index_fill_(0, payload.outlier_indices.long(), 0)
    return payload.scale_pre * mask


class DenseReferenceBackend:
    name = "torch-dense-reference"
    version = "1"

    def capabilities(self) -> BackendCapabilities:
        return _reference_capabilities()

    def supports(self, op: QuantizedLinearSpec, workload: WorkloadSpec) -> SupportResult:
        return _reference_support(op, workload)

    def prepare(self, state: RuntimeLayerState, device: DeviceLike) -> PreparedLayer:
        return PreparedLayer(self.name, self.version, state.spec, _prepare_payload(state, device))

    def linear(self, value: torch.Tensor, layer: PreparedLayer) -> torch.Tensor:
        payload = _payload(layer, self.name, self.version)
        pre = _masked_scale_pre(payload)
        weight = (payload.left * payload.scale_post.reshape(-1, 1)) @ (
            payload.right * payload.scale_mid.reshape(-1, 1) * pre.reshape(1, -1)
        )
        outliers = _outlier_values(payload)
        if payload.outlier_indices is not None and outliers is not None:
            weight = weight.clone()
            weight[:, payload.outlier_indices.long()] += outliers.to(weight.dtype)
        if payload.patch_left is not None and payload.patch_right is not None:
            weight = weight + payload.patch_left.float() @ payload.patch_right.float()
        return torch.nn.functional.linear(
            value,
            weight.to(device=value.device, dtype=value.dtype),
            None if payload.bias is None else payload.bias.to(device=value.device, dtype=value.dtype),
        )


class FactorizedReferenceBackend:
    name = "torch-factorized-reference"
    version = "1"

    def capabilities(self) -> BackendCapabilities:
        return _reference_capabilities()

    def supports(self, op: QuantizedLinearSpec, workload: WorkloadSpec) -> SupportResult:
        return _reference_support(op, workload)

    def prepare(self, state: RuntimeLayerState, device: DeviceLike) -> PreparedLayer:
        return PreparedLayer(self.name, self.version, state.spec, _prepare_payload(state, device))

    def linear(self, value: torch.Tensor, layer: PreparedLayer) -> torch.Tensor:
        payload = _payload(layer, self.name, self.version)
        pre = _masked_scale_pre(payload).to(device=value.device, dtype=value.dtype)
        right = payload.right.to(device=value.device, dtype=value.dtype)
        left = payload.left.to(device=value.device, dtype=value.dtype)
        latent = torch.nn.functional.linear(value * pre, right)
        output = torch.nn.functional.linear(
            latent * payload.scale_mid.to(device=value.device, dtype=value.dtype),
            left * payload.scale_post.to(device=value.device, dtype=value.dtype).reshape(-1, 1),
        )
        outliers = _outlier_values(payload)
        if payload.outlier_indices is not None and outliers is not None:
            output = output + torch.nn.functional.linear(
                value.index_select(-1, payload.outlier_indices.long()),
                outliers.to(device=value.device, dtype=value.dtype),
            )
        if payload.patch_left is not None and payload.patch_right is not None:
            patch_latent = torch.nn.functional.linear(
                value,
                payload.patch_right.to(device=value.device, dtype=value.dtype),
            )
            output = output + torch.nn.functional.linear(
                patch_latent,
                payload.patch_left.to(device=value.device, dtype=value.dtype),
            )
        if payload.bias is not None:
            output = output + payload.bias.to(device=value.device, dtype=value.dtype)
        return output


class PackedReferenceBackend:
    """Correctness backend that unpacks once during preparation, never per linear call."""

    name = "torch-packed-reference"
    version = "1"

    def capabilities(self) -> BackendCapabilities:
        return _reference_capabilities()

    def supports(self, op: QuantizedLinearSpec, workload: WorkloadSpec) -> SupportResult:
        return _reference_support(op, workload)

    def prepare(self, state: RuntimeLayerState, device: DeviceLike) -> PreparedLayer:
        if not isinstance(state, PackedLayerState):
            raise TypeError("packed reference backend requires PackedLayerState")
        return PreparedLayer(
            self.name,
            self.version,
            state.spec,
            _prepare_payload(state.to_logical(), device),
        )

    def linear(self, value: torch.Tensor, layer: PreparedLayer) -> torch.Tensor:
        payload = _payload(layer, self.name, self.version)
        pre = _masked_scale_pre(payload).to(device=value.device, dtype=value.dtype)
        right = payload.right.to(device=value.device, dtype=value.dtype)
        left = payload.left.to(device=value.device, dtype=value.dtype)
        latent = torch.nn.functional.linear(value * pre, right)
        output = torch.nn.functional.linear(
            latent * payload.scale_mid.to(device=value.device, dtype=value.dtype),
            left * payload.scale_post.to(device=value.device, dtype=value.dtype).reshape(-1, 1),
        )
        outliers = _outlier_values(payload)
        if payload.outlier_indices is not None and outliers is not None:
            output = output + torch.nn.functional.linear(
                value.index_select(-1, payload.outlier_indices.long()),
                outliers.to(device=value.device, dtype=value.dtype),
            )
        if payload.patch_left is not None and payload.patch_right is not None:
            patch_latent = torch.nn.functional.linear(
                value,
                payload.patch_right.to(device=value.device, dtype=value.dtype),
            )
            output = output + torch.nn.functional.linear(
                patch_latent,
                payload.patch_left.to(device=value.device, dtype=value.dtype),
            )
        if payload.bias is not None:
            output = output + payload.bias.to(device=value.device, dtype=value.dtype)
        return output
