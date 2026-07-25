"""Pure factorized-linear algebra shared by training, fitting, and replay."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch


def chunk_slices(length: int, chunk_size: int) -> Iterator[slice]:
    """Yield contiguous bounded slices covering ``range(length)`` exactly once."""

    if length < 0:
        raise ValueError("chunked length must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def chunked_reduce(
    tensor: torch.Tensor,
    chunk_size: int,
    reduction_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Sum scalar reductions over bounded first-dimension chunks."""

    if tensor.ndim == 0:
        raise ValueError("chunked reduction requires a tensor with a leading dimension")
    total = torch.zeros((), device=tensor.device, dtype=dtype or tensor.dtype)
    for item_slice in chunk_slices(tensor.shape[0], chunk_size):
        reduced = reduction_fn(tensor[item_slice])
        if reduced.numel() != 1:
            raise ValueError("chunk reduction function must return a scalar tensor")
        total = total + reduced.to(device=total.device, dtype=total.dtype)
    return total


def parse_torch_dtype(name: str) -> torch.dtype:
    """Parse a canonical research-pipeline dtype name."""

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"unsupported torch dtype: {name}") from error


def mask_outlier_columns(scale_pre: torch.Tensor, indices: torch.Tensor | None) -> torch.Tensor:
    if indices is None or indices.numel() == 0:
        return scale_pre
    mask = torch.ones_like(scale_pre)
    mask.index_fill_(0, indices.to(device=scale_pre.device, dtype=torch.long), 0)
    return scale_pre * mask


def materialize_outlier_values(
    values: torch.Tensor,
    scales: torch.Tensor | None,
) -> torch.Tensor:
    return values if scales is None else values.float() * scales.float()


def functional_dense_reconstruction(
    left: torch.Tensor,
    right: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    outlier_indices: torch.Tensor | None = None,
    outlier_values: torch.Tensor | None = None,
    outlier_scales: torch.Tensor | None = None,
    patch_left: torch.Tensor | None = None,
    patch_right: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the scaled low-rank weight plus optional exact columns."""

    pre = mask_outlier_columns(scale_pre, outlier_indices)
    result = (left * scale_post.reshape(-1, 1)) @ (
        right * scale_mid.reshape(-1, 1) * pre.reshape(1, -1)
    )
    if outlier_indices is not None and outlier_values is not None:
        values = materialize_outlier_values(outlier_values, outlier_scales)
        result = result.clone()
        result[:, outlier_indices.long()] += values.to(result.dtype)
    if (patch_left is None) != (patch_right is None):
        raise ValueError("low-rank patch tensors must be paired")
    if patch_left is not None and patch_right is not None:
        result = result + patch_left.to(result.dtype) @ patch_right.to(result.dtype)
    return result


def functional_factorized_linear(
    value: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
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
    scale_left_before_linear: bool = False,
) -> torch.Tensor:
    """Apply the factorized linear without materializing its dense weight."""

    pre = mask_outlier_columns(scale_pre, outlier_indices)
    latent = torch.nn.functional.linear(value * pre, right)
    output = torch.nn.functional.linear(
        latent * scale_mid,
        left * scale_post.reshape(-1, 1) if scale_left_before_linear else left,
    )
    if not scale_left_before_linear:
        output = output * scale_post
    if outlier_indices is not None and outlier_values is not None:
        values = materialize_outlier_values(outlier_values, outlier_scales)
        output = output + torch.nn.functional.linear(
            value.index_select(-1, outlier_indices.long()),
            values.to(device=value.device, dtype=value.dtype),
        )
    if (patch_left is None) != (patch_right is None):
        raise ValueError("low-rank patch tensors must be paired")
    if patch_left is not None and patch_right is not None:
        patch_latent = torch.nn.functional.linear(
            value,
            patch_right.to(device=value.device, dtype=value.dtype),
        )
        output = output + torch.nn.functional.linear(
            patch_latent,
            patch_left.to(device=value.device, dtype=value.dtype),
        )
    if bias is not None:
        output = output + bias
    return output
