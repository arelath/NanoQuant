from __future__ import annotations

import itertools

import pytest
import torch

from nanoquant.domain.codebook_payload_search import (
    SignWordPayloadSearchConfig,
    SignWordPayloadSearchResult,
    _best_corrected_payload_candidates,
    _best_product_payload_candidates,
    refine_sign_word_payloads,
)
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.domain.sign_word_codebook import (
    FullSignCodebook,
    ProductSignCodebook,
    apply_word_flips,
    decode_product_codebook,
    decode_sign_codebook,
)


def _config() -> SignWordPayloadSearchConfig:
    return SignWordPayloadSearchConfig(
        enabled=True,
        outer_passes=2,
        max_words_per_pass=8,
        scale_passes=8,
        candidate_batch_words=4,
        table_chunk_size=2,
    )


def test_corrected_payload_candidate_matches_brute_force() -> None:
    generator = torch.Generator().manual_seed(17)
    linear = torch.randn((3, 32), generator=generator)
    quadratic = torch.rand((3, 32), generator=generator) + 0.1
    current = torch.randint(0, 2, (3, 32), generator=generator).float().mul_(2).sub_(1)
    table = torch.randint(0, 2, (4, 32), generator=generator).float().mul_(2).sub_(1)

    costs, indices, positions = _best_corrected_payload_candidates(
        linear,
        quadratic,
        current,
        table,
        corrections_per_word=2,
        table_chunk_size=2,
    )

    for row in range(current.shape[0]):
        brute: list[tuple[float, int, tuple[int, int]]] = []
        for index in range(table.shape[0]):
            for pair in itertools.combinations(range(32), 2):
                candidate = table[index].clone()
                candidate[list(pair)] *= -1
                delta = candidate - current[row]
                change = float(
                    (-2 * linear[row] * delta + quadratic[row] * delta.square()).sum()
                )
                brute.append((change, index, pair))
        expected = min(brute)
        assert float(costs[row]) == pytest.approx(expected[0], abs=2e-5)
        assert int(indices[row]) == expected[1]
        assert tuple(sorted(positions[row].tolist())) == expected[2]


def test_product_payload_candidate_matches_brute_force() -> None:
    generator = torch.Generator().manual_seed(23)
    linear = torch.randn((3, 32), generator=generator)
    quadratic = torch.rand((3, 32), generator=generator) + 0.1
    current = torch.randint(0, 2, (3, 32), generator=generator).float().mul_(2).sub_(1)
    first = torch.randint(0, 2, (4, 16), generator=generator).float().mul_(2).sub_(1)
    second = torch.randint(0, 2, (4, 16), generator=generator).float().mul_(2).sub_(1)
    codebook = ProductSignCodebook(4, first, second)

    costs, indices = _best_product_payload_candidates(
        linear,
        quadratic,
        current,
        codebook,
        table_chunk_size=2,
    )

    for row in range(current.shape[0]):
        brute: list[tuple[float, int]] = []
        for second_index in range(second.shape[0]):
            for first_index in range(first.shape[0]):
                candidate = torch.cat((first[first_index], second[second_index]))
                delta = candidate - current[row]
                change = float(
                    (-2 * linear[row] * delta + quadratic[row] * delta.square()).sum()
                )
                combined = first_index | (second_index << 2)
                brute.append((change, combined))
        expected = min(brute)
        assert float(costs[row]) == pytest.approx(expected[0], abs=2e-5)
        assert int(indices[row]) == expected[1]


def test_free_word_payload_search_is_monotonic() -> None:
    left = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    true_right = torch.stack(
        (torch.arange(32).remainder(2).float().mul(2).sub(1), torch.ones(32))
    )
    initial_right = true_right.clone()
    initial_right[0, :8] *= -1
    initial_right[1, 8:16] *= -1
    target = reconstruct(
        left,
        true_right,
        torch.ones(32),
        torch.tensor([0.7, 1.3]),
        torch.tensor([1.1, 0.8]),
    )

    result = refine_sign_word_payloads(
        target,
        left,
        initial_right,
        torch.ones(32),
        torch.ones(2),
        torch.ones(2),
        torch.ones(32),
        torch.ones(2),
        free_rows=2,
        codebook=None,
        right_indices=None,
        right_flip_positions=None,
        config=_config(),
    )

    assert result.after_error <= result.before_error
    assert result.accepted_words > 0
    assert result.sign_updates > 0


def test_corrected_payload_search_stays_decodable_and_improves() -> None:
    alternating = torch.arange(32).remainder(2).float().mul(2).sub(1)
    table = torch.stack((torch.ones(32), alternating))
    codebook = FullSignCodebook(1, table)
    initial_indices = torch.tensor([[0], [1]], dtype=torch.int32)
    initial_positions = torch.tensor([[[0, 1]], [[4, 5]]], dtype=torch.int8)
    desired_indices = torch.tensor([[1], [0]], dtype=torch.int32)
    desired_positions = torch.tensor([[[2, 3]], [[6, 7]]], dtype=torch.int8)
    initial_right = apply_word_flips(
        decode_sign_codebook(initial_indices, codebook, 32),
        initial_positions,
    )
    desired_right = apply_word_flips(
        decode_sign_codebook(desired_indices, codebook, 32),
        desired_positions,
    )
    left = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
    target = reconstruct(
        left,
        desired_right,
        torch.ones(32),
        torch.tensor([0.8, 1.2]),
        torch.tensor([1.1, 0.9, 1.3]),
    )

    result = refine_sign_word_payloads(
        target,
        left,
        initial_right,
        torch.ones(32),
        torch.ones(2),
        torch.ones(3),
        torch.ones(32),
        torch.ones(3),
        free_rows=0,
        codebook=codebook,
        right_indices=initial_indices,
        right_flip_positions=initial_positions,
        config=_config(),
    )

    assert result.after_error < result.before_error
    assert result.right_indices is not None
    assert result.right_flip_positions is not None
    decoded = apply_word_flips(
        decode_sign_codebook(result.right_indices, codebook, 32),
        result.right_flip_positions,
    )
    torch.testing.assert_close(decoded, result.right_binary)


def test_product_payload_search_stays_decodable_and_improves() -> None:
    alternating = torch.arange(16).remainder(2).float().mul(2).sub(1)
    first = torch.stack((torch.ones(16), -torch.ones(16), alternating, -alternating))
    second = torch.stack((alternating, -alternating, torch.ones(16), -torch.ones(16)))
    codebook = ProductSignCodebook(4, first, second)
    initial_indices = torch.tensor([[0], [5]], dtype=torch.int32)
    desired_indices = torch.tensor([[10], [15]], dtype=torch.int32)
    initial_right = decode_product_codebook(initial_indices, codebook, 32)
    desired_right = decode_product_codebook(desired_indices, codebook, 32)
    left = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
    target = reconstruct(
        left,
        desired_right,
        torch.ones(32),
        torch.tensor([0.8, 1.2]),
        torch.tensor([1.1, 0.9, 1.3]),
    )

    result = refine_sign_word_payloads(
        target,
        left,
        initial_right,
        torch.ones(32),
        torch.ones(2),
        torch.ones(3),
        torch.ones(32),
        torch.ones(3),
        free_rows=0,
        codebook=codebook,
        right_indices=initial_indices,
        right_flip_positions=None,
        config=_config(),
    )

    assert result.after_error < result.before_error
    assert result.right_indices is not None
    assert result.right_flip_positions is None
    decoded = decode_product_codebook(result.right_indices, codebook, 32)
    torch.testing.assert_close(decoded, result.right_binary)


def test_corrected_payload_search_rejects_mismatched_metadata() -> None:
    codebook = FullSignCodebook(1, torch.stack((torch.ones(32), -torch.ones(32))))

    with pytest.raises(ValueError, match="does not decode"):
        refine_sign_word_payloads(
            torch.zeros((2, 32)),
            torch.ones((2, 2)),
            torch.ones((2, 32)),
            torch.ones(32),
            torch.ones(2),
            torch.ones(2),
            torch.ones(32),
            torch.ones(2),
            free_rows=0,
            codebook=codebook,
            right_indices=torch.ones((2, 1), dtype=torch.int32),
            right_flip_positions=torch.tensor(
                [[[0, 1]], [[0, 1]]], dtype=torch.int8
            ),
            config=_config(),
        )


def _functional_fixture(
    held_out_inputs: torch.Tensor,
) -> tuple[SignWordPayloadSearchConfig, SignWordPayloadSearchResult]:
    initial_right = torch.ones((1, 32))
    initial_right[0, 1] = -1
    target = initial_right.clone()
    target[0, 0] = -0.5
    target[0, 1] = 0
    fit_inputs = torch.zeros((1, 32))
    fit_inputs[0, :2] = torch.tensor([1.0, -1.0])
    config = SignWordPayloadSearchConfig(
        enabled=True,
        outer_passes=1,
        max_words_per_pass=1,
        scale_passes=0,
        candidate_batch_words=1,
        table_chunk_size=1,
        functional_candidate_words_per_pass=1,
    )
    result = refine_sign_word_payloads(
        target,
        torch.ones((1, 1)),
        initial_right,
        torch.ones(32),
        torch.ones(1),
        torch.ones(1),
        torch.ones(32),
        torch.ones(1),
        free_rows=1,
        codebook=None,
        right_indices=None,
        right_flip_positions=None,
        config=config,
        functional_fit_inputs=fit_inputs,
        functional_held_out_inputs=held_out_inputs,
    )
    return config, result


def test_functional_payload_gate_rejects_held_out_regression() -> None:
    held_out_inputs = torch.zeros((1, 32))
    held_out_inputs[0, :2] = 1

    _config_value, result = _functional_fixture(held_out_inputs)

    assert result.accepted_words == 0
    assert result.functional_candidates_ranked == 1
    assert result.functional_fit_error_after == result.functional_fit_error_before
    assert (
        result.functional_held_out_error_after
        == result.functional_held_out_error_before
    )


def test_functional_payload_gate_accepts_disjoint_improvement() -> None:
    held_out_inputs = torch.zeros((1, 32))
    held_out_inputs[0, :2] = torch.tensor([1.0, -1.0])

    _config_value, result = _functional_fixture(held_out_inputs)

    assert result.accepted_words == 1
    assert result.functional_fit_error_after < result.functional_fit_error_before
    assert (
        result.functional_held_out_error_after
        < result.functional_held_out_error_before
    )


def test_functional_product_table_bit_search_updates_shared_decode() -> None:
    first = torch.ones((2, 16))
    second = torch.ones((2, 16))
    first[0, 0] = -1
    codebook = ProductSignCodebook(2, first, second)
    indices = torch.zeros((1, 1), dtype=torch.int32)
    initial_right = decode_product_codebook(indices, codebook, 32)
    target = torch.ones((1, 32))
    functional_inputs = torch.zeros((1, 32))
    functional_inputs[0, 0] = 1

    result = refine_sign_word_payloads(
        target,
        torch.ones((1, 1)),
        initial_right,
        torch.ones(32),
        torch.ones(1),
        torch.ones(1),
        torch.ones(32),
        torch.ones(1),
        free_rows=0,
        codebook=codebook,
        right_indices=indices,
        right_flip_positions=None,
        config=SignWordPayloadSearchConfig(
            enabled=True,
            outer_passes=0,
            max_words_per_pass=0,
            scale_passes=0,
            candidate_batch_words=1,
            table_chunk_size=1,
            functional_table_bit_passes=1,
            functional_table_bit_candidates_per_pass=2,
        ),
        functional_fit_inputs=functional_inputs,
        functional_held_out_inputs=functional_inputs,
    )

    assert isinstance(result.right_codebook, ProductSignCodebook)
    assert result.accepted_table_bit_flips == 1
    assert result.table_bit_sign_updates == 1
    assert result.after_error < result.before_error
    decoded = decode_product_codebook(
        indices,
        result.right_codebook,
        32,
    )
    torch.testing.assert_close(decoded, result.right_binary)
    assert result.functional_fit_error_after == 0
    assert result.functional_held_out_error_after == 0
