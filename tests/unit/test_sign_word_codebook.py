from __future__ import annotations

import torch

from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.sign_word_codebook import (
    BankedFullSignCodebook,
    FullSignCodebook,
    ProductSignCodebook,
    _assign_corrected_flat_words,
    apply_single_word_flip,
    apply_word_flips,
    asymmetric_sign_word_codebook_bit_cost,
    corrected_asymmetric_codebook_bit_cost,
    decode_product_codebook,
    decode_sign_codebook,
    factorize_sign_word_codebook_admm,
    maximum_asymmetric_codebook_rank_for_budget,
    maximum_codebook_rank_for_budget,
    maximum_corrected_asymmetric_rank_for_budget,
    maximum_mixed_right_free_rows_for_budget,
    mixed_right_corrected_codebook_bit_cost,
    sign_word_codebook_bit_cost,
)


def test_codebook_rank_stays_within_free_word_budget() -> None:
    baseline = factor_bit_cost(1152, 6912, 970, scale_bits=16).total
    rank8 = maximum_codebook_rank_for_budget(1152, 6912, baseline, index_width=8)
    rank12 = maximum_codebook_rank_for_budget(1152, 6912, baseline, index_width=12)

    assert rank8 > rank12 > 2 * 970
    assert sign_word_codebook_bit_cost(1152, 6912, rank8, index_width=8).total <= baseline
    assert sign_word_codebook_bit_cost(1152, 6912, rank8 + 32, index_width=8).total > baseline
    assert sign_word_codebook_bit_cost(1152, 6912, rank12, index_width=12).total <= baseline


def test_right_only_codebook_charges_free_left_words() -> None:
    baseline = factor_bit_cost(1152, 6912, 970, scale_bits=16).total
    rank = maximum_asymmetric_codebook_rank_for_budget(
        1152,
        6912,
        baseline,
        left_index_width=None,
        right_index_width=12,
    )

    assert rank == 2048
    assert asymmetric_sign_word_codebook_bit_cost(
        1152,
        6912,
        rank,
        left_index_width=None,
        right_index_width=12,
    ).total <= baseline


def test_single_flip_codebook_charges_positions_and_decodes() -> None:
    baseline = factor_bit_cost(1152, 6912, 970, scale_bits=16).total
    rank = maximum_corrected_asymmetric_rank_for_budget(
        1152,
        6912,
        baseline,
        left_index_width=None,
        right_index_width=12,
        right_flip_bits=5,
    )
    cost = corrected_asymmetric_codebook_bit_cost(
        1152,
        6912,
        rank,
        left_index_width=None,
        right_index_width=12,
        right_flip_bits=5,
    )
    decoded = apply_single_word_flip(
        torch.ones((2, 35)),
        torch.tensor([[3, 2], [7, 0]], dtype=torch.int8),
    )

    assert rank == 1568
    assert cost.total <= baseline
    assert decoded[0, 3] == -1
    assert decoded[0, 34] == -1
    assert int((decoded == -1).sum()) == 4


def test_mixed_right_budget_funds_an_aligned_free_prefix() -> None:
    baseline = factor_bit_cost(1152, 6912, 970, scale_bits=16).total
    free_rows = maximum_mixed_right_free_rows_for_budget(
        1152,
        6912,
        1408,
        baseline,
        right_index_width=10,
        right_flip_bits=9,
    )
    cost = mixed_right_corrected_codebook_bit_cost(
        1152,
        6912,
        1408,
        right_free_rows=free_rows,
        right_index_width=10,
        right_flip_bits=9,
    )

    assert free_rows == 128
    assert cost.total <= baseline


def test_banked_mixed_right_cost_charges_every_implicit_table() -> None:
    single = mixed_right_corrected_codebook_bit_cost(
        1152,
        6912,
        1344,
        right_free_rows=224,
        right_index_width=10,
        right_flip_bits=9,
    )
    banked = mixed_right_corrected_codebook_bit_cost(
        1152,
        6912,
        1344,
        right_free_rows=224,
        right_index_width=10,
        right_flip_bits=9,
        right_codebook_count=2,
    )

    assert banked.total - single.total == (1 << 10) * 32


def test_tiered_mixed_right_cost_charges_corrections_only_on_selected_rows() -> None:
    all_corrected = mixed_right_corrected_codebook_bit_cost(
        1152,
        6912,
        1344,
        right_free_rows=256,
        right_index_width=10,
        right_flip_bits=9,
        right_codebook_count=4,
    )
    tiered = mixed_right_corrected_codebook_bit_cost(
        1152,
        6912,
        1344,
        right_free_rows=256,
        right_index_width=10,
        right_flip_bits=9,
        right_codebook_count=4,
        right_corrected_rows=272,
    )

    assert all_corrected.total - tiered.total == (1088 - 272) * 216 * 9


def test_multiple_flip_positions_use_combinatorial_widths() -> None:
    baseline = factor_bit_cost(1152, 6912, 970, scale_bits=16).total
    rank2 = maximum_corrected_asymmetric_rank_for_budget(
        1152,
        6912,
        baseline,
        left_index_width=None,
        right_index_width=12,
        right_flip_bits=9,
    )
    rank3 = maximum_corrected_asymmetric_rank_for_budget(
        1152,
        6912,
        baseline,
        left_index_width=None,
        right_index_width=12,
        right_flip_bits=13,
    )
    decoded = apply_word_flips(
        torch.ones((1, 32)),
        torch.tensor([[[2, 19]]], dtype=torch.int8),
    )

    assert rank2 == 1344
    assert rank3 == 1152
    assert decoded[0, 2] == -1
    assert decoded[0, 19] == -1
    assert int((decoded == -1).sum()) == 2


def test_joint_assignment_can_prefer_a_farther_base_codeword() -> None:
    values = torch.ones((1, 32))
    values[0, 0] = 10
    table = torch.ones((2, 32))
    table[1, 0] = -1

    assignments, positions, _ = _assign_corrected_flat_words(
        values,
        table,
        flips_per_word=1,
        update=False,
        batch_words=1,
    )

    assert int(assignments[0]) == 1
    assert int(positions[0, 0]) == 0


def test_corrected_assignment_shortlist_controls_offline_search_depth() -> None:
    values = torch.ones((1, 32))
    values[0, 0] = 10
    table = torch.ones((2, 32))
    table[1, 0] = -1

    nearest, _positions, _ = _assign_corrected_flat_words(
        values,
        table,
        flips_per_word=1,
        update=False,
        batch_words=1,
        candidate_count=1,
    )
    searched, _positions, _ = _assign_corrected_flat_words(
        values,
        table,
        flips_per_word=1,
        update=False,
        batch_words=1,
        candidate_count=2,
    )

    assert int(nearest[0]) == 0
    assert int(searched[0]) == 1


def test_product_codebook_decodes_two_half_indices() -> None:
    first = torch.ones((4, 16))
    second = torch.ones((4, 16))
    first[1, 3] = -1
    second[2, 5] = -1
    codebook = ProductSignCodebook(4, first, second)
    indices = torch.tensor([[1 | (2 << 2)]], dtype=torch.int32)

    decoded = decode_product_codebook(indices, codebook, 27)

    assert decoded.shape == (1, 27)
    assert decoded[0, 3] == -1
    assert decoded[0, 16 + 5] == -1
    assert int((decoded == -1).sum()) == 2


def test_full_codebook_decodes_arbitrary_word() -> None:
    entries = torch.ones((4, 32))
    entries[3, (2, 19)] = -1
    codebook = FullSignCodebook(2, entries)

    decoded = decode_sign_codebook(
        torch.tensor([[3]], dtype=torch.int32),
        codebook,
        24,
    )

    assert decoded.shape == (1, 24)
    assert decoded[0, 2] == -1
    assert decoded[0, 19] == -1
    assert int((decoded == -1).sum()) == 2


def test_banked_full_codebook_uses_the_table_implied_by_word_position() -> None:
    entries = torch.ones((2, 2, 32))
    entries[0, 1, 3] = -1
    entries[1, 1, 7] = -1
    codebook = BankedFullSignCodebook(1, entries)

    decoded = decode_sign_codebook(
        torch.tensor([[1, 1]], dtype=torch.int32),
        codebook,
        64,
    )

    assert decoded[0, 3] == -1
    assert decoded[0, 32 + 7] == -1
    assert int((decoded == -1).sum()) == 2


def test_banked_full_codebook_can_select_tables_by_component_row() -> None:
    entries = torch.ones((2, 2, 32))
    entries[0, 1, 3] = -1
    entries[1, 1, 7] = -1
    codebook = BankedFullSignCodebook(1, entries, "row")

    decoded = decode_sign_codebook(
        torch.tensor([[1], [1]], dtype=torch.int32),
        codebook,
        32,
    )

    assert decoded[0, 3] == -1
    assert decoded[1, 7] == -1
    assert int((decoded == -1).sum()) == 2


def test_codebook_admm_exports_only_decodable_words() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn((5, 32), generator=generator)
    result = factorize_sign_word_codebook_admm(
        weight,
        torch.ones(32),
        torch.ones(5),
        32,
        torch.Generator().manual_seed(11),
        index_bits=4,
        outer_iterations=2,
        inner_iterations=2,
        convergence_check_interval=1,
        codebook_update_interval=1,
        assignment_batch_words=8,
    )

    decoded_left = decode_product_codebook(
        result.left_indices,
        result.left_codebook,
        result.factors.left_binary.shape[1],
    )
    decoded_right = decode_product_codebook(
        result.right_indices,
        result.right_codebook,
        result.factors.right_binary.shape[1],
    )
    assert torch.equal(decoded_left, result.factors.left_binary)
    assert torch.equal(decoded_right, result.factors.right_binary)
    assert torch.isfinite(result.factors.reconstruction).all()


def test_codebook_admm_can_leave_left_factor_free() -> None:
    generator = torch.Generator().manual_seed(41)
    result = factorize_sign_word_codebook_admm(
        torch.randn((4, 32), generator=generator),
        torch.ones(32),
        torch.ones(4),
        4,
        torch.Generator().manual_seed(43),
        index_bits=2,
        outer_iterations=2,
        inner_iterations=2,
        codebook_update_interval=1,
        codebook_mode="full",
        constrain_left=False,
    )

    assert result.left_codebook is None
    assert result.left_indices is None
    assert result.right_codebook is not None
    assert result.right_indices is not None
    assert torch.isfinite(result.factors.reconstruction).all()


def test_full_codebook_admm_accepts_an_odd_index_width() -> None:
    generator = torch.Generator().manual_seed(44)
    result = factorize_sign_word_codebook_admm(
        torch.randn((4, 32), generator=generator),
        torch.ones(32),
        torch.ones(4),
        4,
        torch.Generator().manual_seed(45),
        index_bits=3,
        outer_iterations=2,
        inner_iterations=2,
        codebook_update_interval=1,
        codebook_mode="full",
        constrain_left=False,
    )

    assert isinstance(result.right_codebook, FullSignCodebook)
    assert result.right_codebook.index_bits == 3
    assert torch.isfinite(result.factors.reconstruction).all()


def test_full_codebook_admm_exports_implicit_right_word_banks() -> None:
    generator = torch.Generator().manual_seed(46)
    result = factorize_sign_word_codebook_admm(
        torch.randn((4, 64), generator=generator),
        torch.ones(64),
        torch.ones(4),
        4,
        torch.Generator().manual_seed(47),
        index_bits=2,
        outer_iterations=2,
        inner_iterations=2,
        codebook_update_interval=1,
        codebook_mode="full",
        constrain_left=False,
        right_flips_per_word=1,
        right_codebook_banks=2,
    )

    assert isinstance(result.right_codebook, BankedFullSignCodebook)
    assert result.right_codebook.bank_count == 2
    assert result.right_indices is not None
    assert result.right_flip_positions is not None
    decoded = decode_sign_codebook(
        result.right_indices,
        result.right_codebook,
        result.factors.right_binary.shape[1],
    )
    corrected = apply_single_word_flip(decoded, result.right_flip_positions)
    assert torch.equal(corrected, result.factors.right_binary)


def test_full_codebook_admm_exports_implicit_right_row_banks() -> None:
    generator = torch.Generator().manual_seed(48)
    result = factorize_sign_word_codebook_admm(
        torch.randn((4, 64), generator=generator),
        torch.ones(64),
        torch.ones(4),
        4,
        torch.Generator().manual_seed(49),
        index_bits=2,
        outer_iterations=2,
        inner_iterations=2,
        codebook_update_interval=1,
        codebook_mode="full",
        constrain_left=False,
        right_flips_per_word=1,
        right_codebook_banks=2,
        right_codebook_bank_axis="row",
    )

    assert isinstance(result.right_codebook, BankedFullSignCodebook)
    assert result.right_codebook.bank_axis == "row"
    assert result.right_indices is not None
    assert result.right_flip_positions is not None
    decoded = decode_sign_codebook(
        result.right_indices,
        result.right_codebook,
        result.factors.right_binary.shape[1],
    )
    corrected = apply_single_word_flip(decoded, result.right_flip_positions)
    assert torch.equal(corrected, result.factors.right_binary)


def test_full_codebook_admm_can_correct_only_a_row_bank_prefix() -> None:
    generator = torch.Generator().manual_seed(50)
    result = factorize_sign_word_codebook_admm(
        torch.randn((4, 64), generator=generator),
        torch.ones(64),
        torch.ones(4),
        4,
        torch.Generator().manual_seed(51),
        index_bits=2,
        outer_iterations=2,
        inner_iterations=2,
        codebook_update_interval=1,
        codebook_mode="full",
        constrain_left=False,
        right_flips_per_word=1,
        right_codebook_banks=4,
        right_codebook_bank_axis="row",
        right_corrected_codebook_banks=1,
    )

    assert isinstance(result.right_codebook, BankedFullSignCodebook)
    assert result.right_flip_positions is not None
    assert result.right_flip_positions.shape[0] == 1
    assert torch.isfinite(result.factors.reconstruction).all()


def test_codebook_admm_exports_single_flip_corrections() -> None:
    generator = torch.Generator().manual_seed(47)
    result = factorize_sign_word_codebook_admm(
        torch.randn((4, 32), generator=generator),
        torch.ones(32),
        torch.ones(4),
        4,
        torch.Generator().manual_seed(53),
        index_bits=2,
        outer_iterations=2,
        inner_iterations=2,
        codebook_update_interval=1,
        codebook_mode="full",
        constrain_left=False,
        right_flips_per_word=1,
    )

    assert result.right_codebook is not None
    assert result.right_indices is not None
    assert result.right_flip_positions is not None
    decoded = decode_sign_codebook(
        result.right_indices,
        result.right_codebook,
        result.factors.right_binary.shape[1],
    )
    corrected = apply_single_word_flip(decoded, result.right_flip_positions)
    assert torch.equal(corrected, result.factors.right_binary)


def test_codebook_admm_exports_a_free_right_prefix() -> None:
    generator = torch.Generator().manual_seed(59)
    result = factorize_sign_word_codebook_admm(
        torch.randn((4, 32), generator=generator),
        torch.ones(32),
        torch.ones(4),
        4,
        torch.Generator().manual_seed(61),
        index_bits=2,
        outer_iterations=2,
        inner_iterations=2,
        codebook_update_interval=1,
        codebook_mode="full",
        constrain_left=False,
        right_flips_per_word=1,
        right_free_rows=1,
    )

    assert result.right_free_rows == 1
    assert result.right_codebook is not None
    assert result.right_indices is not None
    assert result.right_flip_positions is not None
    coded = decode_sign_codebook(
        result.right_indices,
        result.right_codebook,
        result.factors.right_binary.shape[1],
    )
    corrected = apply_single_word_flip(coded, result.right_flip_positions)
    assert torch.equal(corrected, result.factors.right_binary[1:])
