"""Pure factorized-linear algebra shared by training, fitting, and replay."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RescaledFactorizedTerms:
    """Existing factorized payload terms after a dense row/column rescale."""

    scale_pre: torch.Tensor
    scale_post: torch.Tensor
    outlier_values: torch.Tensor | None
    patch_left: torch.Tensor | None
    patch_right: torch.Tensor | None


def rescale_factorized_terms(
    scale_pre: torch.Tensor,
    scale_post: torch.Tensor,
    *,
    input_multiplier: torch.Tensor | None = None,
    output_multiplier: torch.Tensor | None = None,
    outlier_indices: torch.Tensor | None = None,
    outlier_values: torch.Tensor | None = None,
    patch_left: torch.Tensor | None = None,
    patch_right: torch.Tensor | None = None,
) -> RescaledFactorizedTerms:
    """Encode ``diag(output) @ W @ diag(input)`` without adding payload terms.

    The low-rank body is rescaled through its existing pre/post vectors. Exact
    outlier columns and optional correction patches must be rescaled as well;
    changing only the low-rank scales would represent a different dense weight.
    Results retain each payload tensor's original dtype and shape.
    """

    if scale_pre.ndim != 1 or scale_post.ndim != 1:
        raise ValueError("factorized pre/post scales must be vectors")
    if input_multiplier is None:
        input_multiplier = torch.ones_like(scale_pre)
    if output_multiplier is None:
        output_multiplier = torch.ones_like(scale_post)
    if input_multiplier.shape != scale_pre.shape or output_multiplier.shape != scale_post.shape:
        raise ValueError("factorized rescale multipliers differ from the linear dimensions")
    if not torch.isfinite(input_multiplier).all() or not torch.isfinite(output_multiplier).all():
        raise ValueError("factorized rescale multipliers must be finite")
    if (outlier_indices is None) != (outlier_values is None):
        raise ValueError("factorized outlier indices and values must be paired")
    if (patch_left is None) != (patch_right is None):
        raise ValueError("factorized patch tensors must be paired")

    def _scaled(value: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
        return (value.float() * multiplier.to(device=value.device, dtype=torch.float32)).to(value.dtype)

    scaled_outliers = None
    if outlier_indices is not None and outlier_values is not None:
        if not outlier_values.is_floating_point():
            raise ValueError("factorized rescale requires floating-point outlier values")
        if outlier_values.shape != (scale_post.numel(), outlier_indices.numel()):
            raise ValueError("factorized outlier values differ from the linear dimensions")
        selected_input = input_multiplier.index_select(
            0,
            outlier_indices.to(device=input_multiplier.device, dtype=torch.long),
        )
        outer = output_multiplier.reshape(-1, 1) * selected_input.reshape(1, -1)
        scaled_outliers = _scaled(outlier_values, outer)

    scaled_patch_left = scaled_patch_right = None
    if patch_left is not None and patch_right is not None:
        if (
            patch_left.ndim != 2
            or patch_right.ndim != 2
            or patch_left.shape[0] != scale_post.numel()
            or patch_right.shape[1] != scale_pre.numel()
            or patch_left.shape[1] != patch_right.shape[0]
        ):
            raise ValueError("factorized patch tensors differ from the linear dimensions")
        scaled_patch_left = _scaled(patch_left, output_multiplier.reshape(-1, 1))
        scaled_patch_right = _scaled(patch_right, input_multiplier.reshape(1, -1))

    return RescaledFactorizedTerms(
        _scaled(scale_pre, input_multiplier),
        _scaled(scale_post, output_multiplier),
        scaled_outliers,
        scaled_patch_left,
        scaled_patch_right,
    )


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
