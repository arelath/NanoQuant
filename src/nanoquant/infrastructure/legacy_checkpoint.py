"""Read-only adapter for retained legacy packed NanoQuant checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from nanoquant.application.layers import FactorizedReferenceLinear, FrozenReferenceLinear


def unpack_binary_gemv(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Unpack the legacy row-major LSB-first int32 binary representation."""
    rows, columns = shape
    words = (columns + 31) // 32
    if packed.ndim != 2 or tuple(packed.shape) != (rows, words):
        raise ValueError(f"packed binary shape differs: expected {(rows, words)}, got {tuple(packed.shape)}")
    powers = torch.arange(32, dtype=torch.int32, device=packed.device)
    bits = (packed.unsqueeze(2) >> powers) & 1
    unpacked = bits.reshape(rows, words * 32)[:, :columns].to(torch.int8)
    return 1 - 2 * unpacked


def _shape(value: torch.Tensor) -> tuple[int, int]:
    if value.numel() != 2:
        raise ValueError("legacy binary shape metadata must contain two dimensions")
    return cast(tuple[int, int], tuple(int(item) for item in value.reshape(-1).tolist()))


def _install(model: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, name = path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    if not isinstance(getattr(parent, name, None), nn.Linear):
        raise TypeError(f"legacy checkpoint target is not a linear module: {path}")
    setattr(parent, name, module)


def apply_legacy_checkpoint(
    model: nn.Module,
    state: Mapping[str, Any],
    *,
    backend: str = "factorized",
) -> tuple[str, ...]:
    """Apply packed factors, outliers, embeddings, and auxiliary parameters to a base model."""
    if backend not in {"factorized", "dense"}:
        raise ValueError(f"unsupported legacy checkpoint backend: {backend}")
    prefixes = sorted(key.removesuffix(".U_packed") for key in state if key.endswith(".U_packed"))
    if not prefixes:
        raise ValueError("legacy checkpoint contains no packed NanoQuant layers")
    module_type = FactorizedReferenceLinear if backend == "factorized" else FrozenReferenceLinear
    for prefix in prefixes:
        required = (
            "U_packed",
            "U_shape",
            "V_packed",
            "V_shape",
            "scale_pre",
            "scale_mid",
            "scale_post",
        )
        missing = [name for name in required if f"{prefix}.{name}" not in state]
        if missing:
            raise ValueError(f"legacy checkpoint layer is incomplete: {prefix}: {missing}")
        factor_dtype = state[f"{prefix}.scale_pre"].dtype
        left = unpack_binary_gemv(
            state[f"{prefix}.U_packed"],
            _shape(state[f"{prefix}.U_shape"]),
        ).to(factor_dtype)
        right = unpack_binary_gemv(
            state[f"{prefix}.V_packed"],
            _shape(state[f"{prefix}.V_shape"]),
        ).to(factor_dtype)
        indices = state.get(f"{prefix}.salient_idx")
        values = state.get(f"{prefix}.salient_weight")
        scales = state.get(f"{prefix}.salient_scale")
        bias = state.get(f"{prefix}.bias")
        _install(
            model,
            prefix,
            module_type(
                left,
                right,
                state[f"{prefix}.scale_pre"].reshape(-1),
                state[f"{prefix}.scale_mid"].reshape(-1),
                state[f"{prefix}.scale_post"].reshape(-1),
                bias=bias if isinstance(bias, torch.Tensor) else None,
                outlier_indices=indices.long() if isinstance(indices, torch.Tensor) else None,
                outlier_values=values if isinstance(values, torch.Tensor) else None,
                outlier_scales=scales if isinstance(scales, torch.Tensor) else None,
            ),
        )

    embedding = model.get_submodule("model.embed_tokens")
    weight = getattr(embedding, "weight", None)
    quantized = state.get("model.embed_tokens.weight_int8")
    scale = state.get("model.embed_tokens.weight_int8_scale")
    if (
        not isinstance(weight, nn.Parameter)
        or not isinstance(quantized, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
    ):
        raise ValueError("legacy checkpoint is missing its row-wise int8 embedding")
    row_scale = scale.reshape(-1, 1) if scale.ndim == 1 else scale
    dequantized = quantized.float() * row_scale.float()
    if dequantized.shape != weight.shape:
        raise ValueError("legacy embedding shape differs from the base model")

    parameters = dict(model.named_parameters())
    with torch.no_grad():
        weight.copy_(dequantized.to(dtype=weight.dtype))
        for name, parameter in parameters.items():
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or name == "model.embed_tokens.weight":
                continue
            if value.shape != parameter.shape:
                raise ValueError(f"legacy auxiliary parameter shape differs: {name}")
            parameter.copy_(value.to(dtype=parameter.dtype))
    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()
    return tuple(prefixes)
