from __future__ import annotations

from dataclasses import replace

import pytest

from nanoquant.application.comparison_report import (
    AlignmentRequest,
    ArtifactReference,
    CandidateComparisonRequest,
    ComparabilityField,
    LocationValue,
    SampledMetric,
    ScalarMetric,
    compare_candidate,
    render_candidate_comparison,
)
from nanoquant.application.evaluation import (
    EvaluationDimensions,
    GateDecision,
    MemoryMetrics,
    QuantizationCostMetrics,
    RepresentationMetrics,
    RuntimeMetrics,
)


def _dimensions(
    *,
    core_bits: int,
    deployable_bytes: int,
    quantization_device_bytes: int,
    quantization_seconds: float,
    prefill: float,
    decode: float,
    fallbacks: int,
) -> EvaluationDimensions:
    return EvaluationDimensions(
        RepresentationMetrics.build(
            source_parameter_count=1_000,
            core_bits=core_bits,
            logical_artifact_bytes=deployable_bytes - 10,
            deployable_artifact_bytes=deployable_bytes,
        ),
        MemoryMetrics(
            quantization_peak_device_bytes=quantization_device_bytes,
            quantization_peak_host_bytes=2_000,
            quantization_temporary_disk_bytes=3_000,
            runtime_peak_device_bytes=4_000,
            runtime_peak_host_bytes=5_000,
        ),
        QuantizationCostMetrics(
            calibration_seconds=quantization_seconds,
            factorization_seconds=2.0,
            local_tuning_seconds=3.0,
            global_tuning_seconds=4.0,
            packing_seconds=5.0,
            evaluation_seconds=6.0,
        ),
        RuntimeMetrics(
            time_to_first_token_seconds=0.5,
            prefill_tokens_per_second=prefill,
            inter_token_latency_seconds=1.0 / decode,
            decode_tokens_per_second=decode,
            fallback_count=fallbacks,
        ),
    )


def _request(*, comparable: bool = True) -> CandidateComparisonRequest:
    return CandidateComparisonRequest(
        candidate_name="rewrite",
        baseline_name="legacy",
        candidate_config={
            "intent": {"name": "candidate-name"},
            "factorization": {"admm": {"iterations": 80}, "new_option": True},
            "runtime": {"executor": "resident"},
            "output": {"directory": "candidate"},
        },
        baseline_config={
            "intent": {"name": "baseline-name"},
            "factorization": {"admm": {"iterations": 40}, "removed_option": 7},
            "runtime": {"executor": "streaming"},
            "output": {"directory": "baseline"},
        },
        comparability_fields=(
            ComparabilityField("model", "gemma@pinned", "gemma@pinned"),
            ComparabilityField(
                "dataset",
                "wikitext@candidate" if not comparable else "wikitext@pinned",
                "wikitext@pinned",
            ),
            ComparabilityField("environment", "driver-2", "driver-1", required=False),
        ),
        candidate_artifacts=(
            ArtifactReference("sha256-shared", "calibration"),
            ArtifactReference("sha256-new", "factorization"),
            ArtifactReference("sha256-moved", "packing"),
        ),
        baseline_artifacts=(
            ArtifactReference("sha256-shared", "calibration"),
            ArtifactReference("sha256-old", "factorization"),
            ArtifactReference("sha256-moved", "conversion"),
        ),
        layer_metrics=(
            AlignmentRequest(
                "reconstruction loss",
                (
                    LocationValue("block.0.q_proj", 1.0),
                    LocationValue("block.0.k_proj", 2.0),
                    LocationValue("candidate-only", 3.0),
                ),
                (
                    LocationValue("block.0.q_proj", 0.0),
                    LocationValue("block.0.k_proj", 4.0),
                    LocationValue("baseline-only", 5.0),
                ),
            ),
        ),
        block_metrics=(
            AlignmentRequest(
                "post-refit loss",
                (LocationValue("block.0", 8.0), LocationValue("block.1", 9.0)),
                (LocationValue("block.0", 10.0), LocationValue("block.1", 9.0)),
            ),
        ),
        sampled_metrics=(
            SampledMetric(
                "perplexity",
                (9.0, 9.2, 9.4, 9.6),
                (10.0, 10.2, 10.4, 10.6),
                "minimize",
                minimum_meaningful_delta=0.5,
                bootstrap_samples=200,
                seed=19,
            ),
        ),
        quality_metrics=(
            ScalarMetric("perplexity", "quality", 9.3, 10.3, "minimize"),
            ScalarMetric("accuracy", "quality", 0.7, 0.8, "maximize"),
        ),
        candidate_dimensions=_dimensions(
            core_bits=900,
            deployable_bytes=10**16 + 37,
            quantization_device_bytes=7_000,
            quantization_seconds=8.0,
            prefill=120.0,
            decode=160.0,
            fallbacks=0,
        ),
        baseline_dimensions=_dimensions(
            core_bits=1_000,
            deployable_bytes=10**16 + 99,
            quantization_device_bytes=8_000,
            quantization_seconds=7.0,
            prefill=100.0,
            decode=180.0,
            fallbacks=1,
        ),
        candidate_warning_codes=("NQ-SHARED", "NQ-NEW"),
        baseline_warning_codes=("NQ-SHARED", "NQ-RESOLVED"),
        promotion_decision=GateDecision("promotion", "sha256:gate-policy", ()) if comparable else None,
    )


def test_comparison_reports_semantic_diff_reuse_alignment_uncertainty_and_pareto() -> None:
    result = compare_candidate(_request())

    assert result.directly_comparable
    assert result.conclusion == "promotion"
    assert result.promotion_policy_key == "sha256:gate-policy"
    assert [item.path for item in result.config_differences] == [
        "factorization.admm.iterations",
        "factorization.new_option",
        "factorization.removed_option",
        "runtime.executor",
    ]
    assert not any(item.path.startswith(("intent", "output")) for item in result.config_differences)

    assert [item.artifact_id for item in result.artifact_reuse.shared] == [
        "sha256-moved",
        "sha256-shared",
    ]
    assert [item.artifact_id for item in result.artifact_reuse.candidate_only] == ["sha256-new"]
    assert [item.artifact_id for item in result.artifact_reuse.baseline_only] == ["sha256-old"]
    calibration = next(item for item in result.artifact_reuse.stages if item.stage == "calibration")
    assert calibration.shared_count == 1

    layer = result.layer_alignments[0]
    assert [item.location for item in layer.aligned] == ["block.0.k_proj", "block.0.q_proj"]
    assert layer.aligned[0].absolute_delta == -2.0
    assert layer.aligned[0].relative_delta == -0.5
    assert layer.aligned[1].relative_delta is None
    assert [item.location for item in layer.candidate_only] == ["candidate-only"]
    assert [item.location for item in layer.baseline_only] == ["baseline-only"]

    uncertainty = result.uncertainty[0].result
    assert uncertainty.improvement_delta == pytest.approx(1.0)
    assert uncertainty.outcome == "meaningful-improvement"

    dimensions = {(item.category, item.name): item for item in result.pareto.dimensions}
    assert dimensions[("quality", "perplexity")].status == "improved"
    assert dimensions[("quality", "accuracy")].status == "regressed"
    assert dimensions[("storage", "deployable artifact bytes")].absolute_delta == -62
    assert dimensions[("prefill", "prefill throughput")].status == "improved"
    assert dimensions[("decode", "decode throughput")].status == "regressed"
    assert result.pareto.outcome == "trade-off"
    assert result.warning_codes.new == ("NQ-NEW",)
    assert result.warning_codes.resolved == ("NQ-RESOLVED",)
    assert result.warning_codes.shared == ("NQ-SHARED",)
    assert any("different stages" in warning for warning in result.report_warnings)


def test_required_comparability_mismatch_prevents_a_misleading_conclusion() -> None:
    result = compare_candidate(_request(comparable=False))
    report = render_candidate_comparison(result)

    assert not result.directly_comparable
    assert result.conclusion == "not-comparable"
    assert "required comparability field differs: dataset" in result.report_warnings
    assert "informational comparability field differs: environment" in result.report_warnings
    assert "## Metric comparison suppressed" in report
    assert "## Pareto dimensions" not in report
    assert "## Per-layer alignment" not in report


def test_markdown_renders_every_comparison_dimension_and_unavailable_relative_delta() -> None:
    report = render_candidate_comparison(compare_candidate(_request()))

    for section in (
        "## Comparability",
        "## Semantic configuration diff",
        "## Artifact reuse",
        "## Per-layer alignment",
        "## Per-block alignment",
        "## Evaluation uncertainty",
        "## Pareto dimensions",
        "## Warning codes",
        "## Report warnings",
    ):
        assert section in report
    assert "factorization.admm.iterations" in report
    assert "candidate-only" in report
    assert "meaningful-improvement" in report
    assert "Promotion policy: `sha256:gate-policy`" in report
    assert "| `quality` | perplexity |" in report
    assert "| `block.0.q_proj` | 0.0 | 1.0 | 1.0 | n/a |" in report
    assert "- New: `NQ-NEW`" in report
    assert "candidate-name" not in report
    assert "baseline-name" not in report


def test_comparison_rejects_ambiguous_or_incomplete_inputs() -> None:
    base = _request()

    with pytest.raises(ValueError, match="names must differ"):
        compare_candidate(replace(base, baseline_name="rewrite"))
    with pytest.raises(ValueError, match="comparability field"):
        compare_candidate(replace(base, comparability_fields=()))
    with pytest.raises(ValueError, match="quality metric"):
        compare_candidate(replace(base, quality_metrics=()))
    duplicate_artifact = (*base.candidate_artifacts, base.candidate_artifacts[0])
    with pytest.raises(ValueError, match="duplicate artifact"):
        compare_candidate(replace(base, candidate_artifacts=duplicate_artifact))
    duplicate_location = AlignmentRequest(
        "loss",
        (LocationValue("layer", 1.0), LocationValue("layer", 2.0)),
        (LocationValue("layer", 1.0),),
    )
    with pytest.raises(ValueError, match="duplicate location"):
        compare_candidate(replace(base, layer_metrics=(duplicate_location,)))
