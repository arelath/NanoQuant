from __future__ import annotations

import torch

from nanoquant.domain.outliers import reconstruct_with_outliers
from nanoquant.domain.planning import outlier_bit_cost
from nanoquant.domain.sign_word_codebook import corrected_asymmetric_codebook_bit_cost
from tools.probe_sign_word_codebook import (
    _combine_candidate_bit_cost,
    _fixed_outlier_bit_cost,
    _prepare_candidate_outliers,
    _prepare_fixed_outliers,
    _replace_exact_columns,
    _restore_fixed_outliers,
    build_parser,
)


def test_candidate_outliers_remove_and_exactly_restore_fisher_columns() -> None:
    weight = torch.tensor(
        [
            [1.0, 3.0, 3.0, 4.0],
            [2.0, 1.0, 1.0, 2.0],
        ]
    )
    input_importance = torch.tensor([1.0, 1.0, 0.1, 2.0])
    output_importance = torch.ones(2)

    residual, indices, values = _prepare_candidate_outliers(
        weight,
        input_importance,
        output_importance,
        2,
    )

    assert indices.tolist() == [1, 3]
    assert torch.count_nonzero(residual[:, indices]) == 0
    assert torch.equal(
        reconstruct_with_outliers(residual, indices, values),
        weight,
    )


def test_candidate_outlier_counts_are_parsed_as_a_tuple() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--calibration-state",
            "calibration",
            "--output",
            "results.json",
            "--candidate-outlier-columns",
            "0,1,2",
        ]
    )

    assert args.candidate_outlier_columns == (0, 1, 2)


def test_fixed_outlier_indices_are_parsed_and_remove_exact_columns() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--calibration-state",
            "calibration",
            "--output",
            "results.json",
            "--fixed-outlier-indices",
            "1,3",
        ]
    )
    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    residual, indices, values = _prepare_fixed_outliers(
        weight,
        args.fixed_outlier_indices,
    )

    assert args.fixed_outlier_indices == (1, 3)
    assert indices.tolist() == [1, 3]
    assert torch.count_nonzero(residual[:, indices]) == 0
    assert torch.equal(values, weight[:, indices])


def test_fixed_outlier_columns_round_trip_as_rows_after_transpose() -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    residual, indices, values = _prepare_fixed_outliers(
        weight,
        (0, 2),
        axis=0,
    )
    restored = _restore_fixed_outliers(
        residual,
        indices,
        values,
        axis=0,
    )

    assert torch.count_nonzero(residual[indices, :]) == 0
    assert torch.equal(restored, weight)


def test_transposed_fixed_outlier_cost_uses_source_output_extent() -> None:
    transposed_weight = torch.zeros((3, 4))

    cost = _fixed_outlier_bit_cost(transposed_weight, 2, axis=0)

    assert cost.outlier_value_bits == 4 * 2 * 16
    assert cost.outlier_index_bits == 2 * 2


def test_extended_binary_search_outer_passes_are_parsed() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--calibration-state",
            "calibration",
            "--output",
            "results.json",
            "--binary-search",
            "--binary-search-control-outer-passes",
            "16",
            "--binary-search-tabu-outer-passes",
            "24",
        ]
    )

    assert args.binary_search
    assert args.binary_search_control_outer_passes == 16
    assert args.binary_search_tabu_outer_passes == 24


def test_codebook_and_outlier_costs_are_combined_without_losing_components() -> None:
    factor_cost = corrected_asymmetric_codebook_bit_cost(
        8,
        32,
        4,
        left_index_width=None,
        right_index_width=2,
        right_flip_bits=5,
    )
    columns = outlier_bit_cost(8, 2, value_bits=16, index_bits=5)

    combined, total = _combine_candidate_bit_cost(factor_cost, columns)

    assert combined["outlier_value_bits"] == 256
    assert combined["outlier_index_bits"] == 10
    assert total == factor_cost.total + 266


def test_exact_residual_columns_replace_instead_of_add_to_base() -> None:
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    base = torch.tensor([[0.5, 8.0], [2.5, 9.0]])

    patched = _replace_exact_columns(base, source, torch.tensor([1]))

    assert torch.equal(patched[:, 0], base[:, 0])
    assert torch.equal(patched[:, 1], source[:, 1])
