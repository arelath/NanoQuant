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
    causal_kl_nll_from_logits,
    kl_calibrated_sensitivities,
    load_kl_budget_profile,
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


def test_kl_budget_workflow_resumes_and_checkpoints_each_missing_arm() -> None:
    provenance = _provenance()
    first = KlBudgetArmResult("full", 2.0, 0.5, 10, 3.0)
    resume = KlBudgetProfile(1, provenance, 1.5, (first,), False)
    calls: list[str] = []
    checkpoints: list[KlBudgetProfile] = []

    def evaluate(arm: str) -> KlBudgetArmResult:
        calls.append(arm)
        return KlBudgetArmResult(arm, 2.0, 0.25, 10, 2.0)

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
        1,
        _provenance(),
        1.0,
        (
            KlBudgetArmResult("unit:0:up", 2.0, 0.8, 10, 2.0),
            KlBudgetArmResult("type:up", 2.0, 3.0, 10),
            KlBudgetArmResult("type:down", 2.0, 1.0, 10),
            KlBudgetArmResult("block:0", 2.0, 2.0, 10),
            KlBudgetArmResult("block:1", 2.0, 6.0, 10),
        ),
        True,
    )

    values = dict(kl_calibrated_sensitivities(profile, ("0:up", "1:down")))

    assert values["0:up"] == pytest.approx(0.2)
    assert values["1:down"] == pytest.approx((1 / 4) * (6 / 8))


def test_kl_calibrated_sensitivities_can_force_type_by_block_granularity() -> None:
    profile = KlBudgetProfile(
        1,
        _provenance(),
        1.0,
        (
            KlBudgetArmResult("unit:0:up", 2.0, 0.8, 10, 2.0),
            KlBudgetArmResult("type:up", 2.0, 3.0, 10),
            KlBudgetArmResult("block:0", 2.0, 2.0, 10),
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


def test_profile_key_fails_closed_when_path_content_is_stale(tmp_path: Path) -> None:
    profile = KlBudgetProfile(1, _provenance(), 1.0, (), True)
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
