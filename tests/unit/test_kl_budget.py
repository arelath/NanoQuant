from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    KlBudgetProfile,
    KlBudgetProvenance,
    KlBudgetRequest,
    KlBudgetWorkflow,
    KlSequenceResult,
    causal_kl_nll_from_logits,
    causal_kl_nll_per_sequence_from_logits,
    interaction_corrected_unit_kl_anchors,
    kl_calibrated_sensitivities,
    load_kl_budget_profile,
    measured_unit_kl_anchors,
    paired_bootstrap_kl_delta,
    validate_kl_budget_profile,
)
from nanoquant.config.codec import to_dict
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
)
from nanoquant.infrastructure.kl_teacher_cache import (
    commit_active_kl_teacher_cache,
    load_active_kl_teacher_cache,
)


def _provenance() -> KlBudgetProvenance:
    return KlBudgetProvenance("model", "revision", "recipe", "dataset", "slice", "run")


def _arm(
    name: str,
    nll: float,
    kl: float,
    *,
    normalized_squared_error: float | None = None,
    token_count: int = 10,
) -> KlBudgetArmResult:
    return KlBudgetArmResult(
        name,
        nll,
        kl,
        token_count,
        normalized_squared_error,
        (KlSequenceResult(nll, kl, token_count),),
    )


def test_chunked_causal_kl_matches_direct_reduction() -> None:
    teacher = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.5, 1.5, -0.5], [1.0, 0.0, 0.5], [0.0, 2.0, 1.0]]]
    )
    student = teacher + torch.tensor(
        [[[0.1, -0.2, 0.0], [0.3, -0.1, 0.2], [-0.2, 0.4, 0.0], [0.0, 0.0, 0.0]]]
    )
    tokens = torch.tensor([[0, 1, 2, 1]])

    nll, kl, count = causal_kl_nll_from_logits(teacher, student, tokens, token_chunk_size=1)
    teacher_log_probs = torch.log_softmax(teacher[:, :-1], dim=-1)
    student_log_probs = torch.log_softmax(student[:, :-1], dim=-1)
    expected_kl = (
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1).mean()
    expected_nll = -student_log_probs.gather(2, tokens[:, 1:].unsqueeze(-1)).mean()

    assert count == 3
    assert kl == pytest.approx(float(expected_kl), rel=1e-6)
    assert nll == pytest.approx(float(expected_nll), rel=1e-6)


def test_per_sequence_kl_reduces_to_the_batch_result() -> None:
    torch.manual_seed(7)
    teacher = torch.randn(3, 5, 11)
    student = torch.randn(3, 5, 11)
    tokens = torch.randint(0, 11, (3, 5))

    aggregate = causal_kl_nll_from_logits(teacher, student, tokens, token_chunk_size=3)
    sequences = causal_kl_nll_per_sequence_from_logits(teacher, student, tokens, token_chunk_size=3)
    count = sum(item.token_count for item in sequences)

    assert count == aggregate[2]
    assert sum(item.negative_log_likelihood * item.token_count for item in sequences) / count == pytest.approx(
        aggregate[0]
    )
    assert sum(item.kl_nats_per_token * item.token_count for item in sequences) / count == pytest.approx(aggregate[1])


def test_kl_budget_workflow_resumes_and_checkpoints_each_missing_arm() -> None:
    provenance = _provenance()
    first = _arm("full", 2.0, 0.5, normalized_squared_error=3.0)
    resume = KlBudgetProfile(2, provenance, 1.5, (first,), False)
    calls: list[str] = []
    checkpoints: list[KlBudgetProfile] = []

    def evaluate(arm: str) -> KlBudgetArmResult:
        calls.append(arm)
        return _arm(arm, 2.0, 0.25, normalized_squared_error=2.0)

    result = KlBudgetWorkflow().run(
        KlBudgetRequest(provenance, ("full", "type:up", "block:0")),
        evaluate,
        baseline_negative_log_likelihood=1.5,
        resume=resume,
        checkpoint=checkpoints.append,
    )

    assert calls == ["type:up", "block:0"]
    assert [len(profile.arms) for profile in checkpoints] == [2, 3]
    assert result.complete


def test_kl_calibrated_sensitivities_use_exact_and_type_by_block_fallback() -> None:
    profile = KlBudgetProfile(
        2,
        _provenance(),
        1.0,
        (
            _arm("unit:0:up", 2.0, 0.8, normalized_squared_error=4.0),
            _arm("type:up", 2.0, 3.0),
            _arm("type:down", 2.0, 1.0),
            _arm("block:0", 2.0, 2.0),
            _arm("block:1", 2.0, 6.0),
        ),
        True,
    )

    values = dict(kl_calibrated_sensitivities(profile, ("0:up", "1:down")))

    assert values["0:up"] == pytest.approx(0.2)
    assert values["1:down"] == pytest.approx((1 / 4) * (6 / 8))


def test_kl_calibrated_sensitivities_can_force_type_by_block_granularity() -> None:
    profile = KlBudgetProfile(
        2,
        _provenance(),
        1.0,
        (
            _arm("unit:0:up", 2.0, 0.8, normalized_squared_error=4.0),
            _arm("type:up", 2.0, 3.0),
            _arm("block:0", 2.0, 2.0),
        ),
        True,
    )

    values = dict(
        kl_calibrated_sensitivities(
            profile,
            ("0:up",),
            use_exact_unit_arms=False,
        )
    )

    assert values["0:up"] == pytest.approx(1.0)


def test_measured_unit_kl_anchors_use_physical_arm_kl_without_error_denominator() -> None:
    profile = KlBudgetProfile(
        2,
        _provenance(),
        1.0,
        (
            _arm("unit:0:attn_qkv", 2.0, 0.8, normalized_squared_error=4.0),
            _arm("type:attn_qkv", 2.0, 3.0),
            _arm("block:0", 2.0, 2.0),
        ),
        True,
    )

    values = dict(measured_unit_kl_anchors(profile, ("0:attn_qkv",)))

    assert values == {"0:attn_qkv": pytest.approx(0.8)}


def test_measured_unit_kl_anchors_fail_without_exact_physical_arm() -> None:
    profile = KlBudgetProfile(
        2,
        _provenance(),
        1.0,
        (_arm("type:up", 2.0, 3.0), _arm("block:0", 2.0, 2.0)),
        True,
    )

    with pytest.raises(ValueError, match="no exact physical-unit arm"):
        measured_unit_kl_anchors(profile, ("0:up",))


def _interaction_profile() -> KlBudgetProfile:
    return KlBudgetProfile(
        2,
        _provenance(),
        1.0,
        (
            _arm("unit:0:attn_qkv", 2.0, 0.8),
            _arm("unit:1:attn_qkv", 2.0, 0.4),
            _arm("unit:0:up", 2.0, 0.6),
            _arm("unit:1:up", 2.0, 0.6),
            _arm("type:attn_qkv", 2.0, 3.0),
            _arm("type:up", 2.0, 0.6),
            _arm("block:0", 2.0, 2.0),
            _arm("block:1", 2.0, 2.0),
        ),
        True,
    )


def test_interaction_corrected_anchors_rescale_each_type_to_its_joint_arm() -> None:
    unit_ids = ("0:attn_qkv", "1:attn_qkv", "0:up", "1:up")
    values = dict(interaction_corrected_unit_kl_anchors(_interaction_profile(), unit_ids))

    # Each type's anchors sum to its measured joint type-arm KL (attn_qkv=3.0, up=0.6),
    # replacing the additive-across-units assumption.
    assert values["0:attn_qkv"] + values["1:attn_qkv"] == pytest.approx(3.0)
    assert values["0:up"] + values["1:up"] == pytest.approx(0.6)
    # Within-type shares are preserved: 0:attn_qkv carries twice 1:attn_qkv, as in the raw arms.
    assert values["0:attn_qkv"] == pytest.approx(2.0)
    assert values["1:attn_qkv"] == pytest.approx(1.0)


def test_interaction_corrected_anchors_reweight_types_non_uniformly() -> None:
    unit_ids = ("0:attn_qkv", "1:attn_qkv", "0:up", "1:up")
    raw = dict(measured_unit_kl_anchors(_interaction_profile(), unit_ids))
    corrected = dict(interaction_corrected_unit_kl_anchors(_interaction_profile(), unit_ids))

    # The correction is a non-uniform reweighting across types (super-additive attn_qkv
    # scaled up, sub-additive up scaled down), not a global rescale the geometric-mean
    # normalized allocator would ignore.
    attn_ratio = corrected["0:attn_qkv"] / raw["0:attn_qkv"]
    up_ratio = corrected["0:up"] / raw["0:up"]
    assert attn_ratio == pytest.approx(2.5)
    assert up_ratio == pytest.approx(0.5)
    assert attn_ratio != pytest.approx(up_ratio)


def test_interaction_corrected_anchors_fail_without_type_arm() -> None:
    profile = KlBudgetProfile(
        2,
        _provenance(),
        1.0,
        (_arm("unit:0:up", 2.0, 0.6), _arm("block:0", 2.0, 2.0)),
        True,
    )

    with pytest.raises(ValueError, match="requires a type arm"):
        interaction_corrected_unit_kl_anchors(profile, ("0:up",))


def test_profile_key_fails_closed_when_path_content_is_stale(tmp_path: Path) -> None:
    profile = KlBudgetProfile(2, _provenance(), 1.0, (), True)
    path = tmp_path / "profile.json"
    atomic_write_json(path, to_dict(profile))
    loaded = load_kl_budget_profile(path)

    with pytest.raises(ValueError, match="key differs"):
        validate_kl_budget_profile(
            loaded,
            model_source="model",
            model_revision="revision",
            expected_profile_key="sha256:stale",
        )


def test_evaluator_v2_profile_fails_closed_instead_of_reinterpreting_error_units(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    atomic_write_json(path, {"schema_version": 1})

    with pytest.raises(ValueError, match="unsupported KL budget profile schema"):
        load_kl_budget_profile(path)


def test_paired_bootstrap_uses_ordered_sequence_deltas() -> None:
    before_sequences = tuple(KlSequenceResult(1.0, value, 10) for value in (0.9, 1.0, 1.1, 1.2))
    after_sequences = tuple(KlSequenceResult(1.0, value - 0.2, 10) for value in (0.9, 1.0, 1.1, 1.2))
    before = KlBudgetArmResult("full", 1.0, 1.05, 40, None, before_sequences)
    after = KlBudgetArmResult("full", 1.0, 0.85, 40, None, after_sequences)

    interval = paired_bootstrap_kl_delta(before, after, resamples=100, seed=3)

    assert interval.point_delta == pytest.approx(-0.2)
    assert interval.lower_delta == pytest.approx(-0.2)
    assert interval.upper_delta == pytest.approx(-0.2)


class _ToyDecoderBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)


class _ToyBackbone(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyDecoderBlock(width)])


class _ToyCausalModel(nn.Module):
    def __init__(self, vocabulary: int, width: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocabulary, width)
        self.model = _ToyBackbone(width)
        self.head = nn.Linear(width, vocabulary, bias=False)

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool) -> SimpleNamespace:
        del use_cache
        hidden = self.model.layers[0].proj(self.embed(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


def test_dense_splice_cpu_cache_and_on_the_fly_modes_match_and_restore_weights() -> None:
    torch.manual_seed(13)
    cached_model = _ToyCausalModel(11, 5)
    on_the_fly_model = _ToyCausalModel(11, 5)
    on_the_fly_model.load_state_dict(cached_model.state_dict())
    layer = LayerId(BlockId(0), "proj")
    clean_weight = cached_model.model.layers[0].proj.weight.detach().clone()
    reconstruction = clean_weight + torch.eye(5) * 0.2
    reconstruction_set = SpliceReconstructionSet(
        (SpliceReconstruction(layer, reconstruction, None, 2.0),),
        (("0:proj", (layer,)),),
        (("0:proj", 2.0),),
    )
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    cached = DenseKlSpliceEvaluator(
        cached_model,
        reconstruction_set,
        tokens,
        device="cpu",
        batch_size=1,
        token_chunk_size=2,
        teacher_cache_mode="cpu",
    )
    on_the_fly = DenseKlSpliceEvaluator(
        on_the_fly_model,
        reconstruction_set,
        tokens,
        device="cpu",
        batch_size=1,
        token_chunk_size=2,
        teacher_cache_mode="on_the_fly",
    )

    cached_result = cached("full")
    on_the_fly_result = on_the_fly("full")

    assert cached.baseline_negative_log_likelihood == pytest.approx(
        on_the_fly.baseline_negative_log_likelihood,
        rel=1e-6,
    )
    assert cached_result.negative_log_likelihood == on_the_fly_result.negative_log_likelihood
    assert cached_result.kl_nats_per_token == on_the_fly_result.kl_nats_per_token
    assert len(cached_result.sequences) == tokens.shape[0]
    assert torch.equal(cached_model.model.layers[0].proj.weight, clean_weight)
    assert torch.equal(on_the_fly_model.model.layers[0].proj.weight, clean_weight)


def test_dense_splice_persistent_teacher_cache_round_trips_and_fails_closed(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    first_model = _ToyCausalModel(11, 5)
    second_model = _ToyCausalModel(11, 5)
    second_model.load_state_dict(first_model.state_dict())
    layer = LayerId(BlockId(0), "proj")
    clean_weight = first_model.model.layers[0].proj.weight.detach().clone()
    reconstructions = SpliceReconstructionSet(
        (SpliceReconstruction(layer, clean_weight + torch.eye(5) * 0.1, None, 1.0),),
        (("0:proj", (layer,)),),
        (("0:proj", 1.0),),
    )
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    first = DenseKlSpliceEvaluator(
        first_model,
        reconstructions,
        tokens,
        device="cpu",
        batch_size=1,
    )
    baseline_nll, batches = first.teacher_cache_state()
    committed = commit_active_kl_teacher_cache(
        tmp_path / "cache",
        "sha256:teacher",
        baseline_nll,
        batches,
    )
    loaded = load_active_kl_teacher_cache(tmp_path / "cache", "sha256:teacher")
    assert loaded is not None
    assert loaded.reference == committed.reference
    assert loaded.tensor_bytes == sum(value.numel() * value.element_size() for value in batches)

    second = DenseKlSpliceEvaluator(
        second_model,
        reconstructions,
        tokens,
        device="cpu",
        batch_size=1,
    )
    second.install_teacher_cache(
        loaded.baseline_negative_log_likelihood,
        loaded.batches,
    )
    assert second("full") == first("full")

    with pytest.raises(ValueError, match="identity differs"):
        load_active_kl_teacher_cache(tmp_path / "cache", "sha256:stale")
