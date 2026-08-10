from __future__ import annotations

import torch

from tools.probe_int8_outlier_expansion_kl import (
    _rate_matched_int8_patch,
    _sidecar_bits,
    _stored_int8_columns,
)


def test_stored_int8_columns_round_scales_to_declared_dtype() -> None:
    values = torch.tensor([[0.0, -2.0], [1.0, 0.5], [-0.25, 3.0]])

    restored = _stored_int8_columns(values, scale_dtype=torch.bfloat16)

    assert restored.shape == values.shape
    assert torch.isfinite(restored).all()
    assert torch.allclose(restored.abs().amax(dim=0), values.abs().amax(dim=0), atol=0.02)


def test_rate_matched_patch_quantizes_existing_and_adds_best_residual_columns() -> None:
    source = torch.tensor(
        [
            [1.0, 0.2, 4.0, 0.1, 2.0, 0.3],
            [-1.0, 0.1, -3.0, 0.2, -2.0, 0.4],
            [0.5, -0.2, 2.0, -0.1, 1.5, 0.2],
            [-0.5, -0.1, -1.0, -0.2, -1.0, 0.1],
        ]
    )
    existing_indices = torch.tensor([0, 1])
    existing_values = source[:, existing_indices].clone()
    compressed = source.clone()
    compressed[:, 2] = 0
    compressed[:, 4] = 0

    patch = _rate_matched_int8_patch(
        source,
        compressed,
        existing_indices,
        existing_values,
        scale_bits=1,
        scale_dtype=torch.float32,
    )

    # Two four-row BF16 columns fund three INT8 columns when metadata is one
    # scale bit plus three index bits, leaving one added residual column.
    assert patch.existing_count == 2
    assert patch.expanded_count == 3
    assert patch.additional_indices.tolist() == [2]
    assert patch.int8_sidecar_bits <= patch.bf16_sidecar_bits
    assert torch.sum((source - patch.expanded_weight).square()) < torch.sum(
        (source - patch.same_count_weight).square()
    )


def test_down_projection_rate_matches_thirteen_int8_columns_to_seven_bf16() -> None:
    bf16 = _sidecar_bits(
        out_features=1152,
        in_features=6912,
        count=7,
        value_bits=16,
        scale_bits=0,
    )
    int8 = _sidecar_bits(
        out_features=1152,
        in_features=6912,
        count=13,
        value_bits=8,
        scale_bits=16,
    )
    fourteen = _sidecar_bits(
        out_features=1152,
        in_features=6912,
        count=14,
        value_bits=8,
        scale_bits=16,
    )

    assert int8 <= bf16 < fourteen
    assert bf16 == 129_115
    assert int8 == 120_185


def test_weighted_selection_can_choose_a_different_residual_column() -> None:
    source = torch.tensor(
        [
            [1.0, 0.0, 4.0, 0.0],
            [-1.0, 0.0, -4.0, 2.0],
            [0.5, 0.0, 2.0, 0.0],
            [-0.5, 0.0, -2.0, 2.0],
        ]
    )
    compressed = torch.zeros_like(source)
    existing_indices = torch.tensor([0, 1])
    existing_values = source[:, :2]

    raw = _rate_matched_int8_patch(
        source,
        compressed,
        existing_indices,
        existing_values,
        scale_bits=1,
        scale_dtype=torch.float32,
    )
    weighted = _rate_matched_int8_patch(
        source,
        compressed,
        existing_indices,
        existing_values,
        scale_bits=1,
        scale_dtype=torch.float32,
        input_importance=torch.tensor([1.0, 1.0, 0.01, 100.0]),
        output_importance=torch.ones(4),
    )

    assert raw.additional_indices.tolist() == [2]
    assert weighted.additional_indices.tolist() == [3]
