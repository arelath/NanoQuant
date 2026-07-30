from __future__ import annotations

import torch

from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.sign_word_codebook import (
    FullSignCodebook,
    ProductSignCodebook,
    decode_product_codebook,
    decode_sign_codebook,
    factorize_sign_word_codebook_admm,
    maximum_codebook_rank_for_budget,
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
