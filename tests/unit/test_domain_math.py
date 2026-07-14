import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from nanoquant.domain.metrics import (
    LEGACY_IMPORTANCE_FLOOR,
    per_element_squared_error,
    raw_squared_error,
    reconstruction_metrics,
    weighted_squared_error,
)
from nanoquant.domain.models import ArtifactRef, AttemptSummary, BitCost
from nanoquant.domain.objectives import DiagonalObjective, regularize_covariance, regularized_cholesky, unwhiten, whiten
from nanoquant.domain.outliers import quantize_int8_columns, reconstruct_with_outliers, remove_columns
from nanoquant.domain.planning import decide_retry, effective_bpw, factor_bit_cost, uniform_rank
from nanoquant.domain.seeds import logical_seed


def test_reconstruction_metrics_known_values() -> None:
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    prediction = target + 1
    inputs = torch.tensor([1.0, 2.0])
    outputs = torch.tensor([3.0, 4.0])
    assert raw_squared_error(target, prediction).item() == 4
    assert per_element_squared_error(target, prediction).item() == 1
    assert weighted_squared_error(target, prediction, inputs, outputs).item() == 21
    metrics = reconstruction_metrics(target, prediction, inputs, outputs)
    assert metrics.export_weighted_error == 21
    assert metrics.raw_normalized_error == pytest.approx(4 / 30)


def test_diagonal_objective_matches_legacy_importance_floors() -> None:
    target = torch.zeros((2, 2))
    prediction = torch.ones((2, 2))
    inputs = torch.tensor([0.0, 2.0])
    outputs = torch.tensor([0.0, 3.0])
    expected = (LEGACY_IMPORTANCE_FLOOR + 2.0) * (LEGACY_IMPORTANCE_FLOOR + 3.0)
    objective = DiagonalObjective(inputs, outputs)

    assert objective.weighted_error(target, prediction).item() == pytest.approx(expected)
    transformed = objective.transform_for_factorizer(torch.ones_like(target))
    assert transformed[0, 0].item() == pytest.approx(objective.epsilon**2)
    assert transformed[1, 1].item() == pytest.approx((2.0 * 3.0) ** 0.5)


def test_diagonal_objective_rejects_mismatched_importance_shapes() -> None:
    objective = DiagonalObjective(torch.ones(3), torch.ones(2))
    with pytest.raises(ValueError, match="importance vector lengths"):
        objective.transform_for_factorizer(torch.ones((2, 2)))


def test_whitening_round_trip_and_regularization() -> None:
    covariance = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    factor = regularized_cholesky(regularize_covariance(covariance, 0.01, 0.1, 0.2))
    weight = torch.randn(3, 2, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(unwhiten(whiten(weight, factor), factor), weight, atol=1e-5)


def test_outlier_operations_do_not_mutate_callers_and_int8_is_bounded() -> None:
    weight = torch.tensor([[1.0, 200.0], [2.0, -100.0]])
    original = weight.clone()
    residual, values = remove_columns(weight, torch.tensor([1]))
    assert torch.equal(weight, original)
    assert torch.equal(reconstruct_with_outliers(residual, torch.tensor([1]), values), weight)
    quantized, scale = quantize_int8_columns(values)
    assert quantized.dtype == torch.int8
    assert torch.allclose(quantized.float() * scale, values, atol=float(scale.max()))


def test_bit_accounting_and_uniform_budget() -> None:
    cost = factor_bit_cost(64, 32, 16, scale_bits=16, rank_alignment=32)
    assert cost.binary_factor_bits == 1536
    assert cost.padding_bits == 1536
    assert effective_bpw(cost, 64 * 32) == cost.total / (64 * 32)
    assert uniform_rank(64 * 32, ((64, 32),), 1.0, multiple=4) == 20


def test_retry_is_budgeted_and_attempt_limited() -> None:
    attempt = AttemptSummary(0, 32, ArtifactRef("attempt", "x", 1), 0.7, 0.1, BitCost(), 2.0, False, "")
    retry = decide_retry(
        (attempt,),
        maximum_attempts=2,
        rank_increase_fraction=0.25,
        rank_multiple=8,
        hard_rank_cap=64,
        available_extra_bits=1000,
        cost_per_rank=10,
        weighted_threshold=0.5,
        raw_threshold=None,
    )
    assert retry.action == "retry" and retry.next_rank == 40 and retry.projected_extra_bits == 80
    accepted = decide_retry(
        (attempt, attempt),
        maximum_attempts=2,
        rank_increase_fraction=0.25,
        rank_multiple=8,
        hard_rank_cap=64,
        available_extra_bits=1000,
        cost_per_rank=10,
        weighted_threshold=0.5,
        raw_threshold=None,
    )
    assert accepted.action == "accept_best"


@given(st.integers(), st.text(max_size=20), st.integers(min_value=0, max_value=100))
def test_logical_seeds_are_deterministic(seed: int, layer: str, attempt: int) -> None:
    value = logical_seed(seed, "factorize", 2, layer, attempt)
    assert value == logical_seed(seed, "factorize", 2, layer, attempt)
    assert 0 <= value <= 0x7FFF_FFFF_FFFF_FFFF


@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=64))
def test_bit_cost_addition_is_monotonic(binary: int, scale: int) -> None:
    left = BitCost(binary_factor_bits=binary)
    right = BitCost(scale_bits=scale)
    assert (left + right).total >= left.total
    assert (left + right).total >= right.total
