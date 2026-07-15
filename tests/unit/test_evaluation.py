import math

import pytest
import torch

from nanoquant.application.evaluation import (
    CausalEvaluationRequest,
    EvaluationDimensions,
    EvaluationPartition,
    EvaluationPartitions,
    EvaluatorRegistry,
    EvaluatorSpec,
    GatePolicy,
    GateRule,
    GenerationRegressionRequest,
    MemoryMetrics,
    PairedComparisonRequest,
    QuantizationCostMetrics,
    RepresentationMetrics,
    RuntimeMetrics,
    compare_paired,
    evaluate_causal_nll,
    evaluate_gate,
    evaluate_generation_regression,
    reduce_causal_evaluation_results,
    register_generation_smoke_evaluator,
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
    assert result.sample_count == 1
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


def test_batched_windows_match_serial_causal_nll() -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    serial = evaluate_causal_nll(
        CausalEvaluationRequest(tokens, max_length=4, stride=2, batch_size=1),
        _uniform_logits(16),
    )
    batched = evaluate_causal_nll(
        CausalEvaluationRequest(tokens, max_length=4, stride=2, batch_size=3),
        _uniform_logits(16),
    )

    assert batched.token_count == serial.token_count
    assert batched.window_count == serial.window_count
    assert batched.total_negative_log_likelihood == pytest.approx(serial.total_negative_log_likelihood)
    assert batched.perplexity == pytest.approx(serial.perplexity)


def test_sample_limit_is_deterministic_and_applied_before_batching() -> None:
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    observed: list[tuple[int, ...]] = []

    def logits(batch: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        del attention_mask
        observed.extend(tuple(int(value) for value in row) for row in batch.cpu())
        return torch.zeros(*batch.shape, 16)

    result = evaluate_causal_nll(
        CausalEvaluationRequest(tokens, batch_size=2, maximum_samples=2),
        logits,
    )

    assert result.sample_count == 2
    assert observed == [(1, 2, 3), (4, 5, 6)]


def test_disjoint_shard_reduction_matches_single_process_evaluation() -> None:
    tokens = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ]
    )
    request = dict(max_length=4, stride=2, batch_size=3)

    def row_sensitive_logits(batch: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        del attention_mask
        result = torch.zeros(*batch.shape, 32)
        for row in range(batch.shape[0]):
            if int(batch[row, 0]) < 6:
                result[row, :-1].scatter_(1, batch[row, 1:].unsqueeze(-1), 5.0)
        return result

    full = evaluate_causal_nll(CausalEvaluationRequest(tokens, **request), row_sensitive_logits)
    shards = tuple(
        evaluate_causal_nll(CausalEvaluationRequest(shard, **request), row_sensitive_logits)
        for shard in (tokens[:1], tokens[1:])
    )
    reduced = reduce_causal_evaluation_results(shards)

    assert reduced.total_negative_log_likelihood == pytest.approx(full.total_negative_log_likelihood)
    assert reduced.mean_negative_log_likelihood == pytest.approx(full.mean_negative_log_likelihood)
    assert reduced.perplexity == pytest.approx(full.perplexity)
    assert reduced.token_count == full.token_count
    assert reduced.window_count == full.window_count
    assert reduced.sample_count == full.sample_count


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
    with pytest.raises(ValueError, match="maximum samples"):
        evaluate_causal_nll(
            CausalEvaluationRequest(torch.tensor([[1, 2]]), maximum_samples=0),
            _uniform_logits(4),
        )
    with pytest.raises(ValueError, match="at least one shard"):
        reduce_causal_evaluation_results(())


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


def test_paired_bootstrap_is_directional_deterministic_and_records_variability() -> None:
    request = PairedComparisonRequest(
        candidate_values=(2.0, 4.0, 6.0),
        baseline_values=(1.0, 1.0, 1.0),
        direction="maximize",
        minimum_meaningful_delta=0.5,
        bootstrap_samples=4_000,
        seed=17,
    )

    first = compare_paired(request)
    second = compare_paired(request)

    assert first == second
    assert first.sample_count == 3
    assert first.candidate_mean == 4.0
    assert first.baseline_mean == 1.0
    assert first.raw_delta == first.improvement_delta == 3.0
    assert first.candidate_standard_deviation == 2.0
    assert first.baseline_standard_deviation == 0.0
    assert first.paired_standard_deviation == 2.0
    assert first.confidence_interval[0] > 0.5
    assert first.outcome == "meaningful-improvement"


@pytest.mark.parametrize(
    ("candidate", "baseline", "direction", "threshold", "outcome"),
    [
        ((2.0, 2.0), (1.0, 1.0), "minimize", 0.5, "meaningful-regression"),
        ((1.1, 1.1), (1.0, 1.0), "maximize", 0.2, "no-meaningful-difference"),
        ((0.0, 2.0), (1.0, 1.0), "maximize", 0.25, "inconclusive"),
    ],
)
def test_paired_comparison_has_predefined_meaningful_outcomes(
    candidate: tuple[float, ...],
    baseline: tuple[float, ...],
    direction: str,
    threshold: float,
    outcome: str,
) -> None:
    result = compare_paired(
        PairedComparisonRequest(
            candidate,
            baseline,
            direction,
            threshold,
            bootstrap_samples=2_000,
            seed=5,
        )
    )

    assert result.outcome == outcome
    if direction == "minimize":
        assert result.improvement_delta == -result.raw_delta


@pytest.mark.parametrize(
    "comparison_request",
    [
        PairedComparisonRequest((), (), "maximize", 0.0),
        PairedComparisonRequest((1.0,), (1.0, 2.0), "maximize", 0.0),
        PairedComparisonRequest((1.0,), (1.0,), "sideways", 0.0),
        PairedComparisonRequest((math.nan,), (1.0,), "maximize", 0.0),
        PairedComparisonRequest((1.0,), (1.0,), "maximize", -1.0),
        PairedComparisonRequest((1.0,), (1.0,), "maximize", 0.0, confidence_level=1.0),
        PairedComparisonRequest((1.0,), (1.0,), "maximize", 0.0, bootstrap_samples=0),
    ],
)
def test_paired_comparison_rejects_invalid_requests(
    comparison_request: PairedComparisonRequest,
) -> None:
    with pytest.raises(ValueError):
        compare_paired(comparison_request)


def test_generation_regression_evaluator_is_versioned_exact_and_repeatable() -> None:
    registry = EvaluatorRegistry()
    register_generation_smoke_evaluator(registry)
    request = GenerationRegressionRequest(
        "quantization-paragraph",
        "1",
        (1, 2),
        (3, 4, 5),
        "max_new_tokens",
        ((3, 4, 5), (3, 4, 5)),
        ("max_new_tokens", "max_new_tokens"),
    )

    result = registry.evaluate("deterministic-generation-regression", "1", request)

    assert registry.specification("deterministic-generation-regression", "1").tier == "smoke"
    assert result == evaluate_generation_regression(request)
    assert result.passed
    assert result.deterministic and result.exact_match and result.stop_reason_match
    assert result.first_mismatch_index is None
    assert result.longest_repeated_token_run == 1
    assert result.expected_token_sha256 == result.observed_token_sha256[0]
    changed_observation = GenerationRegressionRequest(
        request.case_name,
        request.case_version,
        request.prompt_token_ids,
        request.expected_token_ids,
        request.expected_stop_reason,
        ((3, 4, 6),),
        ("max_new_tokens",),
    )
    assert evaluate_generation_regression(changed_observation).case_key == result.case_key


def test_generation_regression_reports_mismatch_nondeterminism_and_sanity_failures() -> None:
    result = evaluate_generation_regression(
        GenerationRegressionRequest(
            "bad-output",
            "1",
            (1,),
            (3, 4),
            "eos",
            ((3, 9), (3, 9, 9, 9)),
            ("eos", "max_new_tokens"),
            maximum_repeated_token_run=2,
        )
    )

    assert not result.passed
    assert not result.deterministic
    assert not result.exact_match
    assert not result.stop_reason_match
    assert result.first_mismatch_index == 1
    assert result.longest_repeated_token_run == 3
    assert len(result.warnings) == 4

    empty = evaluate_generation_regression(
        GenerationRegressionRequest("empty", "1", (1,), (2,), "eos", ((),), ("eos",))
    )
    assert not empty.passed
    assert empty.generated_token_count == 0
    assert "shorter" in " ".join(empty.warnings)


@pytest.mark.parametrize(
    "generation_request",
    [
        GenerationRegressionRequest("", "1", (1,), (2,), "eos", ((2,),), ("eos",)),
        GenerationRegressionRequest("case", "1", (), (2,), "eos", ((2,),), ("eos",)),
        GenerationRegressionRequest("case", "1", (1,), (), "eos", ((2,),), ("eos",)),
        GenerationRegressionRequest("case", "1", (1,), (2,), "eos", (), ()),
        GenerationRegressionRequest("case", "1", (1,), (2,), "eos", ((2,),), ()),
        GenerationRegressionRequest("case", "1", (-1,), (2,), "eos", ((2,),), ("eos",)),
        GenerationRegressionRequest(
            "case", "1", (1,), (2,), "eos", ((2,),), ("eos",), minimum_generated_tokens=0
        ),
    ],
)
def test_generation_regression_rejects_invalid_cases(
    generation_request: GenerationRegressionRequest,
) -> None:
    with pytest.raises(ValueError):
        evaluate_generation_regression(generation_request)

    with pytest.raises(TypeError):
        evaluate_generation_regression(object())


def test_evaluation_dimensions_keep_size_memory_cost_and_runtime_metrics_separate() -> None:
    representation = RepresentationMetrics.build(
        source_parameter_count=100,
        core_bits=95,
        logical_artifact_bytes=200,
        deployable_artifact_bytes=13,
    )
    memory = MemoryMetrics(10, 20, 30, 4, 5)
    cost = QuantizationCostMetrics(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    runtime = RuntimeMetrics(0.04, 380.0, 0.0045, 160.0, 0)
    dimensions = EvaluationDimensions(representation, memory, cost, runtime)

    assert dimensions.representation.effective_core_bpw == 0.95
    assert dimensions.representation.artifact_bpw == 1.04
    assert dimensions.representation.core_bits == 95
    assert dimensions.representation.logical_artifact_bytes == 200
    assert dimensions.representation.deployable_artifact_bytes == 13
    assert dimensions.memory.quantization_peak_device_bytes == 10
    assert dimensions.memory.runtime_peak_device_bytes == 4
    assert dimensions.quantization_cost.accounted_seconds == 21.0
    assert dimensions.runtime.decode_tokens_per_second == 160.0
    assert dimensions.runtime.fallback_count == 0


def test_evaluation_dimensions_reject_invalid_or_conflated_values() -> None:
    with pytest.raises(ValueError, match="source parameter"):
        RepresentationMetrics.build(0, 0, 0, 0)
    with pytest.raises(ValueError, match="non-negative integers"):
        RepresentationMetrics.build(1, -1, 0, 0)
    with pytest.raises(ValueError, match="memory metrics"):
        MemoryMetrics(0, 0, -1, 0, 0)
    with pytest.raises(ValueError, match="cost metrics"):
        QuantizationCostMetrics(0.0, 0.0, 0.0, math.nan, 0.0, 0.0)
    with pytest.raises(ValueError, match="runtime metrics"):
        RuntimeMetrics(0.0, 0.0, -1.0, 0.0, 0)
    with pytest.raises(ValueError, match="fallback"):
        RuntimeMetrics(0.0, 0.0, 0.0, 0.0, -1)
