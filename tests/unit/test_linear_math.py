from __future__ import annotations

import pytest
import torch

from nanoquant.domain.linear_math import (
    functional_dense_reconstruction,
    rescale_factorized_terms,
)


def test_factorized_rescale_covers_low_rank_outliers_and_patch() -> None:
    left = torch.tensor([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    right = torch.tensor([[1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, -1.0]])
    scale_pre = torch.tensor([0.5, 0.0, 1.5, 2.0])
    scale_mid = torch.tensor([0.75, 1.25])
    scale_post = torch.tensor([1.0, 1.5, 0.5])
    outlier_indices = torch.tensor([1])
    outlier_values = torch.tensor([[2.0], [3.0], [4.0]])
    patch_left = torch.tensor([[0.5], [1.0], [-0.5]])
    patch_right = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    input_multiplier = torch.tensor([2.0, 3.0, 0.5, 1.25])
    output_multiplier = torch.tensor([0.5, 2.0, 1.5])
    baseline = functional_dense_reconstruction(
        left,
        right,
        scale_pre,
        scale_mid,
        scale_post,
        outlier_indices,
        outlier_values,
        None,
        patch_left,
        patch_right,
    )

    scaled = rescale_factorized_terms(
        scale_pre,
        scale_post,
        input_multiplier=input_multiplier,
        output_multiplier=output_multiplier,
        outlier_indices=outlier_indices,
        outlier_values=outlier_values,
        patch_left=patch_left,
        patch_right=patch_right,
    )
    reconstructed = functional_dense_reconstruction(
        left,
        right,
        scaled.scale_pre,
        scale_mid,
        scaled.scale_post,
        outlier_indices,
        scaled.outlier_values,
        None,
        scaled.patch_left,
        scaled.patch_right,
    )

    expected = output_multiplier.reshape(-1, 1) * baseline * input_multiplier.reshape(1, -1)
    torch.testing.assert_close(reconstructed, expected)
    assert scaled.scale_pre.dtype == scale_pre.dtype
    assert scaled.outlier_values is not None
    assert scaled.outlier_values.shape == outlier_values.shape


def test_factorized_rescale_rejects_quantized_outlier_values() -> None:
    with pytest.raises(ValueError, match="floating-point outlier"):
        rescale_factorized_terms(
            torch.ones(2),
            torch.ones(3),
            outlier_indices=torch.tensor([0]),
            outlier_values=torch.ones((3, 1), dtype=torch.int8),
        )
