import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from nanoquant.domain.factorization import SCHEDULES, factorize_admm
from nanoquant.domain.metrics import weighted_squared_error
from nanoquant.domain.models import ArtifactRef, AttemptSummary, BitCost
from nanoquant.domain.outliers import residual_probe_scores, store_outlier_values
from nanoquant.domain.packing import pack_sign_bits, unpack_sign_bits
from nanoquant.domain.planning import allocate_rank_budget, decide_retry
from nanoquant.domain.scale_fit import fit_scales, reconstruct
from nanoquant.domain.states import TrainableNanoQuantState, freeze_state


def test_admm_is_deterministic_uses_generator_and_does_not_mutate_inputs() -> None:
    weight = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0], [0.5, 1.0, -1.0]])
    original = weight.clone()
    importance = torch.ones(3)
    first = factorize_admm(
        weight,
        importance,
        importance,
        2,
        torch.Generator().manual_seed(7),
        outer_iterations=4,
        inner_iterations=2,
        convergence_check_interval=2,
    )
    second = factorize_admm(
        weight,
        importance,
        importance,
        2,
        torch.Generator().manual_seed(7),
        outer_iterations=4,
        inner_iterations=2,
        convergence_check_interval=2,
    )
    assert torch.equal(weight, original)
    assert torch.equal(first.left_binary, second.left_binary)
    assert torch.equal(first.left_latent, second.left_latent)
    assert torch.equal(first.right_latent, second.right_latent)
    assert torch.allclose(first.reconstruction, second.reconstruction)
    assert set(torch.unique(first.left_binary).tolist()) <= {-1.0, 1.0}
    assert first.iterations_completed == 4
    assert [point.iteration for point in first.trace] == [1, 2, 4]


def test_admm_rejects_invalid_schedule_and_dimensions() -> None:
    with pytest.raises(ValueError, match="unknown penalty"):
        factorize_admm(
            torch.eye(2),
            torch.ones(2),
            torch.ones(2),
            1,
            torch.Generator(),
            outer_iterations=1,
            penalty_schedule="missing",
        )
    with pytest.raises(ValueError, match="rank"):
        factorize_admm(torch.eye(2), torch.ones(2), torch.ones(2), 3, torch.Generator())
    assert all(0 <= function(0.3) <= 1 for function in SCHEDULES.values())


def test_admm_preserves_legacy_factor_dtype_and_signed_svid_scales() -> None:
    weight = torch.tensor(
        [[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0], [0.5, 1.0, -1.0]],
        dtype=torch.bfloat16,
    )
    result = factorize_admm(
        weight,
        torch.ones(3),
        torch.ones(3),
        2,
        torch.Generator().manual_seed(7),
        outer_iterations=4,
        inner_iterations=2,
    )

    exported = (
        result.left_binary * result.scale_post.reshape(-1, 1)
    ) @ (
        result.right_binary
        * result.scale_mid.reshape(-1, 1)
        * result.scale_pre.reshape(1, -1)
    )
    assert result.left_latent.dtype is torch.float32
    assert result.right_latent.dtype is torch.float32
    assert all(
        value.dtype is torch.bfloat16
        for value in (
            result.left_binary,
            result.right_binary,
            result.scale_pre,
            result.scale_mid,
            result.scale_post,
        )
    )
    assert (result.scale_pre < 0).any() and (result.scale_post < 0).any()
    assert torch.equal(result.reconstruction, exported)


def test_scale_fit_never_worsens_weighted_objective_and_protects_columns() -> None:
    generator = torch.Generator().manual_seed(2)
    target = torch.randn(5, 4, generator=generator)
    left = torch.sign(torch.randn(5, 2, generator=generator))
    right = torch.sign(torch.randn(2, 4, generator=generator))
    result = fit_scales(
        target,
        left,
        right,
        torch.ones(4),
        torch.ones(2),
        torch.ones(5),
        torch.ones(4),
        torch.ones(5),
        alternating_passes=2,
        protected_columns=torch.tensor([1]),
    )
    assert result.after_error <= result.before_error + 1e-6
    if result.accepted:
        assert result.scale_pre[1] == 0


def test_freezing_validates_shapes_clones_and_reconstructs_independently() -> None:
    left = torch.tensor([[1.2, -0.2], [-3.0, 4.0]])
    state = TrainableNanoQuantState(
        left, torch.tensor([[1.0, -1.0], [-1.0, 1.0]]), torch.ones(2), torch.ones(2), torch.ones(2)
    )
    frozen = freeze_state(state)
    expected = reconstruct(
        torch.sign(state.left_latent),
        torch.sign(state.right_latent),
        state.scale_pre,
        state.scale_mid,
        state.scale_post,
    )
    state.left_latent.zero_()
    assert torch.equal(frozen.dense_weight(), expected)
    with pytest.raises(ValueError, match="ranks"):
        freeze_state(
            TrainableNanoQuantState(torch.ones(2, 3), torch.ones(2, 2), torch.ones(2), torch.ones(2), torch.ones(2))
        )


def test_residual_probe_isolates_factorizer_and_storage_dtypes() -> None:
    calls = []

    def probe(weight: torch.Tensor, rank: int, generator: torch.Generator) -> torch.Tensor:
        calls.append((weight.clone(), rank, generator.initial_seed()))
        return torch.zeros_like(weight)

    weight = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    scores = residual_probe_scores(weight, 1, torch.ones(2), torch.ones(2), probe, torch.Generator().manual_seed(11))
    assert torch.equal(scores, torch.tensor([10.0, 5.0]))
    assert calls[0][1:] == (1, 11)
    for dtype, expected in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("int8", torch.int8)):
        stored, scale = store_outlier_values(weight, dtype)
        assert stored.dtype is expected
        assert (scale is not None) is (dtype == "int8")


@given(st.integers(min_value=0, max_value=5000))
def test_utility_allocator_never_exceeds_budget(extra_bits: int) -> None:
    dimensions = ((8, 4), (6, 6))
    budget = 24 + extra_bits
    ranks = allocate_rank_budget(dimensions, (2.0, 1.0), budget, multiple=1, floor_ranks=(1, 1), ceiling_ranks=(4, 4))
    spent = sum(rank * (output + inputs) for rank, (output, inputs) in zip(ranks, dimensions, strict=True))
    assert spent <= budget


def test_scale_fit_weight_matches_reported_objective() -> None:
    target = torch.eye(2)
    result = fit_scales(
        target,
        torch.ones(2, 1),
        torch.ones(1, 2),
        torch.ones(2),
        torch.ones(1),
        torch.ones(2),
        torch.ones(2),
        torch.ones(2),
    )
    measured = float(weighted_squared_error(target, result.reconstruction, torch.ones(2), torch.ones(2)))
    assert measured == pytest.approx(result.after_error)


@given(st.lists(st.booleans(), min_size=1, max_size=257))
def test_pack_independent_sign_reconstruction(values: list[bool]) -> None:
    signs = torch.tensor([1.0 if value else -1.0 for value in values])
    packed, shape = pack_sign_bits(signs)
    assert torch.equal(unpack_sign_bits(packed, shape), signs)


@given(st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False))
def test_reconstruction_is_invariant_to_reciprocal_outer_scaling(multiplier: float) -> None:
    left = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    right = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
    baseline = reconstruct(left, right, torch.tensor([2.0, 3.0]), torch.tensor([0.5, 1.5]), torch.tensor([1.0, 4.0]))
    rescaled = reconstruct(
        left,
        right,
        torch.tensor([2.0, 3.0]) / multiplier,
        torch.tensor([0.5, 1.5]),
        torch.tensor([1.0, 4.0]) * multiplier,
    )
    assert torch.allclose(baseline, rescaled, rtol=2e-5, atol=2e-5)


@given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
def test_retry_budget_decision_is_monotonic(first_budget: int, extra_budget: int) -> None:
    attempt = AttemptSummary(0, 8, ArtifactRef("attempt", "x", 1), 2.0, 0.0, BitCost(), 2.0, False, "")
    low = decide_retry(
        (attempt,),
        maximum_attempts=2,
        rank_increase_fraction=0.5,
        rank_multiple=1,
        hard_rank_cap=16,
        available_extra_bits=first_budget,
        cost_per_rank=10,
        weighted_threshold=1.0,
        raw_threshold=None,
    )
    high = decide_retry(
        (attempt,),
        maximum_attempts=2,
        rank_increase_fraction=0.5,
        rank_multiple=1,
        hard_rank_cap=16,
        available_extra_bits=first_budget + extra_budget,
        cost_per_rank=10,
        weighted_threshold=1.0,
        raw_threshold=None,
    )
    if low.action == "retry":
        assert high.action == "retry"
