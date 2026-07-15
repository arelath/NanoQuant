"""Typed candidate-versus-baseline comparison reports.

The comparison layer consumes already-resolved run results.  It deliberately does not
read console logs or infer comparability from similar-looking metric names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from nanoquant.application.evaluation import (
    EvaluationDimensions,
    GateDecision,
    PairedComparisonRequest,
    PairedComparisonResult,
    compare_paired,
)
from nanoquant.config.codec import canonical_json, to_dict

Numeric: TypeAlias = int | float


@dataclass(frozen=True, slots=True)
class ComparabilityField:
    name: str
    candidate_value: object
    baseline_value: object
    required: bool = True


@dataclass(frozen=True, slots=True)
class ComparabilityAssessment:
    name: str
    candidate_value: object
    baseline_value: object
    required: bool
    matches: bool


@dataclass(frozen=True, slots=True)
class ConfigDifference:
    path: str
    baseline_present: bool
    baseline_value: object | None
    candidate_present: bool
    candidate_value: object | None


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_id: str
    stage: str

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.stage:
            raise ValueError("artifact identity and stage are required")


@dataclass(frozen=True, slots=True)
class SharedArtifact:
    artifact_id: str
    candidate_stage: str
    baseline_stage: str


@dataclass(frozen=True, slots=True)
class StageArtifactReuse:
    stage: str
    shared_count: int
    candidate_only_count: int
    baseline_only_count: int


@dataclass(frozen=True, slots=True)
class ArtifactReuse:
    shared: tuple[SharedArtifact, ...]
    candidate_only: tuple[ArtifactReference, ...]
    baseline_only: tuple[ArtifactReference, ...]
    stages: tuple[StageArtifactReuse, ...]


@dataclass(frozen=True, slots=True)
class LocationValue:
    location: str
    value: Numeric

    def __post_init__(self) -> None:
        _validate_numeric(self.value, f"metric at {self.location or '<empty>'}")
        if not self.location:
            raise ValueError("metric location is required")


@dataclass(frozen=True, slots=True)
class AlignmentRequest:
    metric_name: str
    candidate_values: tuple[LocationValue, ...]
    baseline_values: tuple[LocationValue, ...]
    unit: str = ""


@dataclass(frozen=True, slots=True)
class AlignedDelta:
    location: str
    baseline_value: Numeric
    candidate_value: Numeric
    absolute_delta: Numeric
    relative_delta: float | None


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    metric_name: str
    unit: str
    aligned: tuple[AlignedDelta, ...]
    candidate_only: tuple[LocationValue, ...]
    baseline_only: tuple[LocationValue, ...]


@dataclass(frozen=True, slots=True)
class SampledMetric:
    name: str
    candidate_values: tuple[float, ...]
    baseline_values: tuple[float, ...]
    direction: str
    minimum_meaningful_delta: float
    unit: str = ""
    confidence_level: float = 0.95
    bootstrap_samples: int = 2_000
    seed: int = 0


@dataclass(frozen=True, slots=True)
class UncertaintyComparison:
    name: str
    direction: str
    unit: str
    result: PairedComparisonResult


@dataclass(frozen=True, slots=True)
class ScalarMetric:
    name: str
    category: str
    candidate_value: Numeric
    baseline_value: Numeric
    direction: str
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.category:
            raise ValueError("scalar metric name and category are required")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("scalar metric direction must be minimize or maximize")
        _validate_numeric(self.candidate_value, f"candidate {self.name}")
        _validate_numeric(self.baseline_value, f"baseline {self.name}")


@dataclass(frozen=True, slots=True)
class ParetoDimension:
    name: str
    category: str
    direction: str
    unit: str
    baseline_value: Numeric
    candidate_value: Numeric
    absolute_delta: Numeric
    relative_delta: float | None
    improvement_delta: Numeric
    status: str


@dataclass(frozen=True, slots=True)
class ParetoView:
    dimensions: tuple[ParetoDimension, ...]
    candidate_improvements: tuple[str, ...]
    candidate_regressions: tuple[str, ...]
    equivalent: tuple[str, ...]
    outcome: str


@dataclass(frozen=True, slots=True)
class WarningComparison:
    shared: tuple[str, ...]
    new: tuple[str, ...]
    resolved: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateComparisonRequest:
    candidate_name: str
    baseline_name: str
    candidate_config: Mapping[str, object]
    baseline_config: Mapping[str, object]
    comparability_fields: tuple[ComparabilityField, ...]
    candidate_artifacts: tuple[ArtifactReference, ...]
    baseline_artifacts: tuple[ArtifactReference, ...]
    layer_metrics: tuple[AlignmentRequest, ...]
    block_metrics: tuple[AlignmentRequest, ...]
    sampled_metrics: tuple[SampledMetric, ...]
    quality_metrics: tuple[ScalarMetric, ...]
    candidate_dimensions: EvaluationDimensions
    baseline_dimensions: EvaluationDimensions
    candidate_warning_codes: tuple[str, ...] = ()
    baseline_warning_codes: tuple[str, ...] = ()
    promotion_decision: GateDecision | None = None
    ignored_config_roots: tuple[str, ...] = ("intent", "observability", "output")
    relative_denominator_floor: float = 1e-12


@dataclass(frozen=True, slots=True)
class CandidateComparisonResult:
    candidate_name: str
    baseline_name: str
    directly_comparable: bool
    comparability: tuple[ComparabilityAssessment, ...]
    config_differences: tuple[ConfigDifference, ...]
    artifact_reuse: ArtifactReuse
    layer_alignments: tuple[AlignmentResult, ...]
    block_alignments: tuple[AlignmentResult, ...]
    uncertainty: tuple[UncertaintyComparison, ...]
    warning_codes: WarningComparison
    pareto: ParetoView
    promotion_decision: str
    promotion_policy_key: str | None
    promotion_reasons: tuple[str, ...]
    conclusion: str
    report_warnings: tuple[str, ...]


def _validate_numeric(value: Numeric, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")


def _canonical_value(value: object, label: str) -> tuple[object, str]:
    converted = to_dict(value)
    try:
        encoded = canonical_json(converted)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be canonically serializable") from exc
    return converted, encoded


def _flatten(value: object, path: str, result: dict[str, object]) -> None:
    converted = to_dict(value)
    if isinstance(converted, dict):
        if not converted:
            result[path] = converted
        for key in sorted(converted):
            child = f"{path}.{key}" if path else key
            _flatten(converted[key], child, result)
        return
    if isinstance(converted, list):
        if not converted:
            result[path] = converted
        for index, item in enumerate(converted):
            _flatten(item, f"{path}[{index}]", result)
        return
    _canonical_value(converted, path)
    result[path] = converted


def _semantic_config_diff(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    ignored_roots: tuple[str, ...],
) -> tuple[ConfigDifference, ...]:
    if any(not isinstance(key, str) for key in (*candidate.keys(), *baseline.keys())):
        raise ValueError("configuration roots must be strings")
    ignored = set(ignored_roots)
    if any(not root for root in ignored):
        raise ValueError("ignored config roots must be non-empty")
    candidate_flat: dict[str, object] = {}
    baseline_flat: dict[str, object] = {}
    for key in sorted(candidate):
        if key not in ignored:
            _flatten(candidate[key], key, candidate_flat)
    for key in sorted(baseline):
        if key not in ignored:
            _flatten(baseline[key], key, baseline_flat)
    differences: list[ConfigDifference] = []
    for path in sorted(set(candidate_flat) | set(baseline_flat)):
        candidate_present = path in candidate_flat
        baseline_present = path in baseline_flat
        candidate_value = candidate_flat.get(path)
        baseline_value = baseline_flat.get(path)
        same = False
        if candidate_present and baseline_present:
            _, candidate_json = _canonical_value(candidate_value, f"candidate config {path}")
            _, baseline_json = _canonical_value(baseline_value, f"baseline config {path}")
            same = candidate_json == baseline_json
        if not same:
            differences.append(
                ConfigDifference(
                    path,
                    baseline_present,
                    baseline_value,
                    candidate_present,
                    candidate_value,
                )
            )
    return tuple(differences)


def _artifact_map(artifacts: tuple[ArtifactReference, ...], label: str) -> dict[str, ArtifactReference]:
    result: dict[str, ArtifactReference] = {}
    for artifact in artifacts:
        if artifact.artifact_id in result:
            raise ValueError(f"{label} contains duplicate artifact identity: {artifact.artifact_id}")
        result[artifact.artifact_id] = artifact
    return result


def _artifact_reuse(
    candidate_artifacts: tuple[ArtifactReference, ...],
    baseline_artifacts: tuple[ArtifactReference, ...],
) -> ArtifactReuse:
    candidate = _artifact_map(candidate_artifacts, "candidate artifacts")
    baseline = _artifact_map(baseline_artifacts, "baseline artifacts")
    shared_ids = sorted(set(candidate) & set(baseline))
    shared = tuple(
        SharedArtifact(artifact_id, candidate[artifact_id].stage, baseline[artifact_id].stage)
        for artifact_id in shared_ids
    )
    candidate_only = tuple(candidate[key] for key in sorted(set(candidate) - set(baseline)))
    baseline_only = tuple(baseline[key] for key in sorted(set(baseline) - set(candidate)))
    stages = sorted({artifact.stage for artifact in (*candidate_artifacts, *baseline_artifacts)})
    stage_reuse = tuple(
        StageArtifactReuse(
            stage,
            sum(item.candidate_stage == stage and item.baseline_stage == stage for item in shared),
            sum(item.stage == stage for item in candidate_only),
            sum(item.stage == stage for item in baseline_only),
        )
        for stage in stages
    )
    return ArtifactReuse(shared, candidate_only, baseline_only, stage_reuse)


def _location_map(values: tuple[LocationValue, ...], label: str) -> dict[str, LocationValue]:
    result: dict[str, LocationValue] = {}
    for value in values:
        if value.location in result:
            raise ValueError(f"{label} contains duplicate location: {value.location}")
        result[value.location] = value
    return result


def _align_metric(request: AlignmentRequest, floor: float) -> AlignmentResult:
    if not request.metric_name:
        raise ValueError("alignment metric name is required")
    candidate = _location_map(request.candidate_values, f"candidate {request.metric_name}")
    baseline = _location_map(request.baseline_values, f"baseline {request.metric_name}")
    aligned = []
    for location in sorted(set(candidate) & set(baseline)):
        candidate_value = candidate[location].value
        baseline_value = baseline[location].value
        delta = candidate_value - baseline_value
        relative = None if abs(baseline_value) <= floor else delta / abs(baseline_value)
        aligned.append(AlignedDelta(location, baseline_value, candidate_value, delta, relative))
    candidate_only = tuple(candidate[key] for key in sorted(set(candidate) - set(baseline)))
    baseline_only = tuple(baseline[key] for key in sorted(set(baseline) - set(candidate)))
    return AlignmentResult(request.metric_name, request.unit, tuple(aligned), candidate_only, baseline_only)


def _dimension_metrics(candidate: EvaluationDimensions, baseline: EvaluationDimensions) -> tuple[ScalarMetric, ...]:
    c_rep, b_rep = candidate.representation, baseline.representation
    c_mem, b_mem = candidate.memory, baseline.memory
    c_cost, b_cost = candidate.quantization_cost, baseline.quantization_cost
    c_run, b_run = candidate.runtime, baseline.runtime
    return (
        ScalarMetric(
            "effective core BPW",
            "storage",
            c_rep.effective_core_bpw,
            b_rep.effective_core_bpw,
            "minimize",
            "bits/weight",
        ),
        ScalarMetric("artifact BPW", "storage", c_rep.artifact_bpw, b_rep.artifact_bpw, "minimize", "bits/weight"),
        ScalarMetric(
            "logical artifact bytes",
            "storage",
            c_rep.logical_artifact_bytes,
            b_rep.logical_artifact_bytes,
            "minimize",
            "bytes",
        ),
        ScalarMetric(
            "deployable artifact bytes",
            "storage",
            c_rep.deployable_artifact_bytes,
            b_rep.deployable_artifact_bytes,
            "minimize",
            "bytes",
        ),
        ScalarMetric(
            "quantization peak device bytes",
            "memory",
            c_mem.quantization_peak_device_bytes,
            b_mem.quantization_peak_device_bytes,
            "minimize",
            "bytes",
        ),
        ScalarMetric(
            "quantization peak host bytes",
            "memory",
            c_mem.quantization_peak_host_bytes,
            b_mem.quantization_peak_host_bytes,
            "minimize",
            "bytes",
        ),
        ScalarMetric(
            "quantization temporary disk bytes",
            "memory",
            c_mem.quantization_temporary_disk_bytes,
            b_mem.quantization_temporary_disk_bytes,
            "minimize",
            "bytes",
        ),
        ScalarMetric(
            "runtime peak device bytes",
            "memory",
            c_mem.runtime_peak_device_bytes,
            b_mem.runtime_peak_device_bytes,
            "minimize",
            "bytes",
        ),
        ScalarMetric(
            "runtime peak host bytes",
            "memory",
            c_mem.runtime_peak_host_bytes,
            b_mem.runtime_peak_host_bytes,
            "minimize",
            "bytes",
        ),
        ScalarMetric(
            "calibration time",
            "quantization-cost",
            c_cost.calibration_seconds,
            b_cost.calibration_seconds,
            "minimize",
            "seconds",
        ),
        ScalarMetric(
            "factorization time",
            "quantization-cost",
            c_cost.factorization_seconds,
            b_cost.factorization_seconds,
            "minimize",
            "seconds",
        ),
        ScalarMetric(
            "local tuning time",
            "quantization-cost",
            c_cost.local_tuning_seconds,
            b_cost.local_tuning_seconds,
            "minimize",
            "seconds",
        ),
        ScalarMetric(
            "global tuning time",
            "quantization-cost",
            c_cost.global_tuning_seconds,
            b_cost.global_tuning_seconds,
            "minimize",
            "seconds",
        ),
        ScalarMetric(
            "packing time", "quantization-cost", c_cost.packing_seconds, b_cost.packing_seconds, "minimize", "seconds"
        ),
        ScalarMetric(
            "evaluation time",
            "quantization-cost",
            c_cost.evaluation_seconds,
            b_cost.evaluation_seconds,
            "minimize",
            "seconds",
        ),
        ScalarMetric(
            "accounted quantization time",
            "quantization-cost",
            c_cost.accounted_seconds,
            b_cost.accounted_seconds,
            "minimize",
            "seconds",
        ),
        ScalarMetric(
            "time to first token",
            "prefill",
            c_run.time_to_first_token_seconds,
            b_run.time_to_first_token_seconds,
            "minimize",
            "seconds",
        ),
        ScalarMetric(
            "prefill throughput",
            "prefill",
            c_run.prefill_tokens_per_second,
            b_run.prefill_tokens_per_second,
            "maximize",
            "tokens/second",
        ),
        ScalarMetric(
            "inter-token latency",
            "decode",
            c_run.inter_token_latency_seconds,
            b_run.inter_token_latency_seconds,
            "minimize",
            "seconds",
        ),
        ScalarMetric(
            "decode throughput",
            "decode",
            c_run.decode_tokens_per_second,
            b_run.decode_tokens_per_second,
            "maximize",
            "tokens/second",
        ),
        ScalarMetric(
            "runtime fallback count", "decode", c_run.fallback_count, b_run.fallback_count, "minimize", "count"
        ),
    )


def _pareto_view(metrics: tuple[ScalarMetric, ...], floor: float) -> ParetoView:
    keys: set[tuple[str, str]] = set()
    dimensions: list[ParetoDimension] = []
    for metric in metrics:
        key = (metric.category, metric.name)
        if key in keys:
            raise ValueError(f"duplicate Pareto metric: {metric.category}/{metric.name}")
        keys.add(key)
        delta = metric.candidate_value - metric.baseline_value
        relative = None if abs(metric.baseline_value) <= floor else delta / abs(metric.baseline_value)
        improvement = delta if metric.direction == "maximize" else -delta
        status = "improved" if improvement > 0 else "regressed" if improvement < 0 else "equivalent"
        dimensions.append(
            ParetoDimension(
                metric.name,
                metric.category,
                metric.direction,
                metric.unit,
                metric.baseline_value,
                metric.candidate_value,
                delta,
                relative,
                improvement,
                status,
            )
        )
    improved = tuple(item.name for item in dimensions if item.status == "improved")
    regressed = tuple(item.name for item in dimensions if item.status == "regressed")
    equivalent = tuple(item.name for item in dimensions if item.status == "equivalent")
    if improved and not regressed:
        outcome = "candidate-dominates"
    elif regressed and not improved:
        outcome = "baseline-dominates"
    elif not improved and not regressed:
        outcome = "equivalent"
    else:
        outcome = "trade-off"
    return ParetoView(tuple(dimensions), improved, regressed, equivalent, outcome)


def compare_candidate(request: CandidateComparisonRequest) -> CandidateComparisonResult:
    """Build a complete typed comparison from two resolved candidate results."""

    if not request.candidate_name or not request.baseline_name:
        raise ValueError("candidate and baseline names are required")
    if request.candidate_name == request.baseline_name:
        raise ValueError("candidate and baseline names must differ")
    if not request.comparability_fields:
        raise ValueError("at least one explicit comparability field is required")
    if not request.quality_metrics or not any(metric.category == "quality" for metric in request.quality_metrics):
        raise ValueError("at least one quality metric with category quality is required")
    floor = request.relative_denominator_floor
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("relative denominator floor must be finite and non-negative")

    field_names: set[str] = set()
    comparability = []
    for field in request.comparability_fields:
        if not field.name or field.name in field_names:
            raise ValueError(f"comparability field names must be non-empty and unique: {field.name!r}")
        field_names.add(field.name)
        candidate_value, candidate_json = _canonical_value(
            field.candidate_value, f"candidate comparability field {field.name}"
        )
        baseline_value, baseline_json = _canonical_value(
            field.baseline_value, f"baseline comparability field {field.name}"
        )
        comparability.append(
            ComparabilityAssessment(
                field.name,
                candidate_value,
                baseline_value,
                field.required,
                candidate_json == baseline_json,
            )
        )
    directly_comparable = all(item.matches or not item.required for item in comparability)

    config_differences = _semantic_config_diff(
        request.candidate_config,
        request.baseline_config,
        request.ignored_config_roots,
    )
    artifact_reuse = _artifact_reuse(request.candidate_artifacts, request.baseline_artifacts)
    for scope, metrics in (("layer", request.layer_metrics), ("block", request.block_metrics)):
        names = [metric.metric_name for metric in metrics]
        if len(names) != len(set(names)):
            raise ValueError(f"{scope} alignment metric names must be unique")
    layer_alignments = tuple(_align_metric(item, floor) for item in request.layer_metrics)
    block_alignments = tuple(_align_metric(item, floor) for item in request.block_metrics)

    sampled_names = [metric.name for metric in request.sampled_metrics]
    if any(not name for name in sampled_names) or len(sampled_names) != len(set(sampled_names)):
        raise ValueError("sampled metric names must be non-empty and unique")
    uncertainty = tuple(
        UncertaintyComparison(
            metric.name,
            metric.direction,
            metric.unit,
            compare_paired(
                PairedComparisonRequest(
                    metric.candidate_values,
                    metric.baseline_values,
                    metric.direction,
                    metric.minimum_meaningful_delta,
                    metric.confidence_level,
                    metric.bootstrap_samples,
                    metric.seed,
                )
            ),
        )
        for metric in request.sampled_metrics
    )

    candidate_warnings = set(request.candidate_warning_codes)
    baseline_warnings = set(request.baseline_warning_codes)
    if "" in candidate_warnings or "" in baseline_warnings:
        raise ValueError("warning codes must be non-empty")
    warning_codes = WarningComparison(
        tuple(sorted(candidate_warnings & baseline_warnings)),
        tuple(sorted(candidate_warnings - baseline_warnings)),
        tuple(sorted(baseline_warnings - candidate_warnings)),
    )

    pareto = _pareto_view(
        (*request.quality_metrics, *_dimension_metrics(request.candidate_dimensions, request.baseline_dimensions)),
        floor,
    )
    report_warnings = []
    for assessment in comparability:
        if not assessment.matches:
            importance = "required" if assessment.required else "informational"
            report_warnings.append(f"{importance} comparability field differs: {assessment.name}")
    for scope, alignments in (("layer", layer_alignments), ("block", block_alignments)):
        for alignment in alignments:
            if alignment.candidate_only or alignment.baseline_only:
                report_warnings.append(
                    f"{scope} alignment {alignment.metric_name} is incomplete "
                    f"({len(alignment.candidate_only)} candidate-only, "
                    f"{len(alignment.baseline_only)} baseline-only)"
                )
            if any(item.relative_delta is None for item in alignment.aligned):
                report_warnings.append(
                    f"{scope} alignment {alignment.metric_name} contains a near-zero baseline; "
                    "relative delta is unavailable"
                )
    for shared in artifact_reuse.shared:
        if shared.candidate_stage != shared.baseline_stage:
            report_warnings.append(
                f"artifact {shared.artifact_id} is reused across different stages "
                f"({shared.baseline_stage} -> {shared.candidate_stage})"
            )
    if any(item.relative_delta is None for item in pareto.dimensions):
        report_warnings.append("one or more Pareto dimensions have a near-zero baseline; relative delta is unavailable")

    decision = request.promotion_decision
    if decision is not None and not isinstance(decision, GateDecision):
        raise ValueError("promotion decision must be a GateDecision")
    promotion = decision.outcome if decision is not None else "not-evaluated"
    if not directly_comparable:
        conclusion = "not-comparable"
    elif promotion != "not-evaluated":
        conclusion = promotion
    else:
        conclusion = pareto.outcome
    return CandidateComparisonResult(
        request.candidate_name,
        request.baseline_name,
        directly_comparable,
        tuple(comparability),
        config_differences,
        artifact_reuse,
        layer_alignments,
        block_alignments,
        uncertainty,
        warning_codes,
        pareto,
        promotion,
        decision.policy_key if decision is not None else None,
        decision.reasons if decision is not None else (),
        conclusion,
        tuple(report_warnings),
    )


def _code(value: object) -> str:
    return "`" + canonical_json(value).replace("|", "\\|") + "`"


def _number(value: Numeric | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _relative(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.6%}"


def _render_alignment(lines: list[str], title: str, results: tuple[AlignmentResult, ...]) -> None:
    lines.extend((f"## {title}", ""))
    if not results:
        lines.append("No aligned metrics supplied.")
        lines.append("")
        return
    for result in results:
        lines.extend(
            (
                f"### {result.metric_name}",
                "",
                "| Location | Baseline | Candidate | Absolute delta | Relative delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            )
        )
        lines.extend(
            f"| `{item.location}` | {_number(item.baseline_value)} | {_number(item.candidate_value)} | "
            f"{_number(item.absolute_delta)} | {_relative(item.relative_delta)} |"
            for item in result.aligned
        )
        if not result.aligned:
            lines.append("| _none_ | n/a | n/a | n/a | n/a |")
        if result.candidate_only:
            lines.append(
                f"\nCandidate-only locations: {', '.join(f'`{item.location}`' for item in result.candidate_only)}"
            )
        if result.baseline_only:
            lines.append(
                f"\nBaseline-only locations: {', '.join(f'`{item.location}`' for item in result.baseline_only)}"
            )
        lines.append("")


def render_candidate_comparison(result: CandidateComparisonResult) -> str:
    """Render a deterministic Markdown comparison without consulting external state."""

    lines = [
        f"# Candidate comparison: {result.candidate_name} vs {result.baseline_name}",
        "",
        f"- Directly comparable: **{'yes' if result.directly_comparable else 'no'}**",
        f"- Promotion decision: `{result.promotion_decision}`",
        f"- Conclusion: `{result.conclusion}`",
        f"- Pareto outcome: `{'not-reported' if not result.directly_comparable else result.pareto.outcome}`",
        "",
        "## Comparability",
        "",
        "| Field | Required | Match | Baseline | Candidate |",
        "| --- | --- | --- | --- | --- |",
    ]
    if result.promotion_policy_key is not None:
        lines[6:6] = [
            f"- Promotion policy: `{result.promotion_policy_key}`",
            f"- Promotion reasons: {', '.join(result.promotion_reasons) or 'none'}",
        ]
    lines.extend(
        f"| {item.name} | {'yes' if item.required else 'no'} | {'yes' if item.matches else 'no'} | "
        f"{_code(item.baseline_value)} | {_code(item.candidate_value)} |"
        for item in result.comparability
    )
    lines.extend(("", "## Semantic configuration diff", ""))
    if result.config_differences:
        lines.extend(("| Path | Baseline | Candidate |", "| --- | --- | --- |"))
        for difference in result.config_differences:
            baseline = _code(difference.baseline_value) if difference.baseline_present else "_missing_"
            candidate = _code(difference.candidate_value) if difference.candidate_present else "_missing_"
            lines.append(f"| `{difference.path}` | {baseline} | {candidate} |")
    else:
        lines.append("No semantic configuration differences.")

    reuse = result.artifact_reuse
    lines.extend(
        (
            "",
            "## Artifact reuse",
            "",
            f"- Shared artifacts: {len(reuse.shared)}",
            f"- Candidate-only artifacts: {len(reuse.candidate_only)}",
            f"- Baseline-only artifacts: {len(reuse.baseline_only)}",
            "",
            "| Stage | Shared in same stage | Candidate only | Baseline only |",
            "| --- | ---: | ---: | ---: |",
        )
    )
    lines.extend(
        f"| `{stage.stage}` | {stage.shared_count} | {stage.candidate_only_count} | {stage.baseline_only_count} |"
        for stage in reuse.stages
    )
    if not result.directly_comparable:
        lines.extend(
            (
                "",
                "## Metric comparison suppressed",
                "",
                "Required source, dataset, evaluator, or protocol identities differ. Per-layer, per-block, "
                "uncertainty, and Pareto deltas are intentionally not rendered.",
                "",
                "## Warning codes",
                "",
            )
        )
        for label, codes in (
            ("New", result.warning_codes.new),
            ("Resolved", result.warning_codes.resolved),
            ("Shared", result.warning_codes.shared),
        ):
            lines.append(f"- {label}: {', '.join(f'`{code}`' for code in codes) if codes else 'none'}")
        lines.extend(("", "## Report warnings", ""))
        lines.extend(f"- {warning}" for warning in result.report_warnings)
        return "\n".join(lines) + "\n"
    _render_alignment(lines, "Per-layer alignment", result.layer_alignments)
    _render_alignment(lines, "Per-block alignment", result.block_alignments)

    lines.extend(("## Evaluation uncertainty", ""))
    if result.uncertainty:
        lines.extend(
            (
                "| Metric | Baseline mean | Candidate mean | Improvement | Confidence interval | Outcome |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            )
        )
        lines.extend(
            f"| {item.name} | {item.result.baseline_mean} | {item.result.candidate_mean} | "
            f"{item.result.improvement_delta} | [{item.result.confidence_interval[0]}, "
            f"{item.result.confidence_interval[1]}] | `{item.result.outcome}` |"
            for item in result.uncertainty
        )
    else:
        lines.append("No sampled metrics supplied.")

    lines.extend(
        (
            "",
            "## Pareto dimensions",
            "",
            "| Category | Metric | Direction | Baseline | Candidate | Absolute delta | Relative delta | Status |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        )
    )
    lines.extend(
        f"| `{item.category}` | {item.name} | `{item.direction}` | {_number(item.baseline_value)} | "
        f"{_number(item.candidate_value)} | {_number(item.absolute_delta)} | "
        f"{_relative(item.relative_delta)} | `{item.status}` |"
        for item in result.pareto.dimensions
    )
    lines.extend(("", "## Warning codes", ""))
    for label, codes in (
        ("New", result.warning_codes.new),
        ("Resolved", result.warning_codes.resolved),
        ("Shared", result.warning_codes.shared),
    ):
        lines.append(f"- {label}: {', '.join(f'`{code}`' for code in codes) if codes else 'none'}")
    lines.extend(("", "## Report warnings", ""))
    lines.extend(f"- {warning}" for warning in result.report_warnings)
    if not result.report_warnings:
        lines.append("No report warnings.")
    return "\n".join(lines) + "\n"
