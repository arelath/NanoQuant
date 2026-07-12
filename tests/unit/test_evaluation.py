import math

import pytest
import torch

from nanoquant.application.evaluation import (
    CausalEvaluationRequest,
    EvaluationPartition,
    EvaluationPartitions,
    EvaluatorRegistry,
    EvaluatorSpec,
    GatePolicy,
    GateRule,
    evaluate_causal_nll,
    evaluate_gate,
)


def _uniform_logits(vocabulary_size: int):
    def logits(tokens: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        assert attention_mask is not None
        return torch.zeros(*tokens.shape, vocabulary_size, device=tokens.device)

    return logits


def _next_token_logits(vocabulary_size: int):
    def logits(tokens: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        del attention_mask
        result = torch.full((*tokens.shape, vocabulary_size), -10.0, device=tokens.device)
        if tokens.shape[1] > 1:
            result[:, :-1].scatter_(2, tokens[:, 1:].unsqueeze(-1), 10.0)
        result[:, -1, 0] = 10_000.0
        return result

    return logits


def test_causal_nll_scores_every_shifted_token_once_across_overlap_and_partial_window() -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    result = evaluate_causal_nll(
        CausalEvaluationRequest(tokens, max_length=4, stride=2),
        _uniform_logits(8),
    )

    assert result.token_count == 4
    assert result.window_count == 2
    assert result.mean_negative_log_likelihood == pytest.approx(math.log(8))
    assert result.perplexity == pytest.approx(8)


def test_causal_shift_ignores_last_logit_and_overlap_matches_single_window() -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    full = evaluate_causal_nll(
        CausalEvaluationRequest(tokens, max_length=6, stride=6),
        _next_token_logits(10),
    )
    overlapped = evaluate_causal_nll(
        CausalEvaluationRequest(tokens, max_length=4, stride=2),
        _next_token_logits(10),
    )

    assert full.token_count == overlapped.token_count == 5
    assert full.total_negative_log_likelihood == pytest.approx(overlapped.total_negative_log_likelihood)
    assert full.mean_negative_log_likelihood < 1e-6


def test_bos_eos_padding_and_multiple_sequences_have_exact_denominator() -> None:
    tokens = torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    result = evaluate_causal_nll(
        CausalEvaluationRequest(
            tokens,
            mask,
            max_length=8,
            stride=8,
            prepend_bos_token_id=2,
            append_eos_token_id=1,
        ),
        _uniform_logits(16),
    )

    assert result.token_count == 7
    assert result.window_count == 2
    assert result.perplexity == pytest.approx(16)


def test_invalid_padding_and_empty_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="contiguous right padding"):
        evaluate_causal_nll(
            CausalEvaluationRequest(
                torch.tensor([[1, 2, 3]]),
                torch.tensor([[1, 0, 1]]),
            ),
            _uniform_logits(4),
        )
    with pytest.raises(ValueError, match="no target tokens"):
        evaluate_causal_nll(CausalEvaluationRequest(torch.tensor([[1]])), _uniform_logits(4))


def test_evaluator_registry_is_versioned_and_specs_have_semantic_keys() -> None:
    registry = EvaluatorRegistry()
    first = EvaluatorSpec("causal-ppl", "1", "quick", (("max_length", 128),))
    second = EvaluatorSpec("causal-ppl", "2", "standard", (("max_length", 2048),))
    registry.register(first, lambda request: request)
    registry.register(second, lambda request: request)

    assert registry.specification("causal-ppl", "1") == first
    assert registry.evaluate("causal-ppl", "2", "value") == "value"
    assert first.semantic_key != second.semantic_key
    with pytest.raises(ValueError, match="already registered"):
        registry.register(first, lambda request: request)


def test_registry_builds_exact_and_cumulative_evaluation_tiers() -> None:
    registry = EvaluatorRegistry()
    specs = tuple(EvaluatorSpec(f"eval-{tier}", "1", tier) for tier in EvaluatorRegistry.TIERS)
    for spec in specs:
        registry.register(spec, lambda request: f"result:{request}")
    requests = {(spec.name, spec.version): spec.tier for spec in specs}

    assert registry.specifications_for_tier("standard") == specs[:3]
    assert registry.specifications_for_tier("standard", cumulative=False) == (specs[2],)
    results = registry.evaluate_tier("full", requests)
    assert [result for _spec, result in results] == [f"result:{tier}" for tier in EvaluatorRegistry.TIERS]
    with pytest.raises(ValueError, match="missing"):
        registry.evaluate_tier("quick", {})


def test_evaluation_partitions_are_content_hashed_and_overlap_checked() -> None:
    calibration = EvaluationPartition.build("calibration", "1", ((1, 2, 3), (4, 5, 6)))
    quick = EvaluationPartition.build("quick", "1", ((7, 8, 9),))
    final = EvaluationPartition.build("final", "1", ((10, 11, 12),))
    partitions = EvaluationPartitions(calibration, quick, final)

    assert partitions.calibration.content_hash.startswith("sha256:")
    assert calibration == EvaluationPartition.build("calibration", "1", ((1, 2, 3), (4, 5, 6)))
    assert calibration.content_hash != quick.content_hash
    with pytest.raises(ValueError, match="overlap"):
        EvaluationPartitions(
            calibration,
            EvaluationPartition.build("quick", "1", ((4, 5, 6),)),
            final,
        )
    with pytest.raises(ValueError, match="duplicate"):
        EvaluationPartition.build("bad", "1", ((1, 2), (1, 2)))


def test_immutable_gate_policy_has_all_three_predefined_outcomes() -> None:
    policy = GatePolicy(
        "quality-and-size",
        "1",
        (GateRule("ppl", maximum=400), GateRule("bpw", minimum=0.95, maximum=1.05)),
    )
    assert evaluate_gate(policy, (("ppl", 384.9), ("bpw", 1.0))).outcome == "promotion"
    rejection = evaluate_gate(policy, (("ppl", 401.0), ("bpw", 1.0)))
    assert rejection.outcome == "rejection"
    assert rejection.reasons == ("ppl=401.0 exceeds 400",)
    assert evaluate_gate(policy, (("bpw", 1.0),)).outcome == "inconclusive"
    assert policy.semantic_key == GatePolicy(
        "quality-and-size",
        "1",
        (GateRule("ppl", maximum=400), GateRule("bpw", minimum=0.95, maximum=1.05)),
    ).semantic_key
