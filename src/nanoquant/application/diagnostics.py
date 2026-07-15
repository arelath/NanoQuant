"""Versioned diagnostic rules derived from structured quantization metrics."""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass

from nanoquant.config.codec import canonical_json


@dataclass(frozen=True, slots=True)
class DiagnosticPolicy:
    name: str = "nanoquant-diagnostics"
    version: str = "1"
    calibration_max_relative_spread: float = 0.25
    calibration_scale_floor: float = 1e-12
    hessian_max_condition_number: float = 1e8
    hessian_max_jitter_escalations: int = 2
    admm_plateau_window: int = 5
    admm_min_relative_improvement: float = 0.01
    export_max_error_ratio: float = 1.25
    retry_min_relative_improvement_per_bit: float = 1e-6
    outlier_min_relative_block_improvement: float = 0.005
    tuning_min_recovery_fraction: float = 0.5
    runtime_max_fallback_count: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("diagnostic policy name and version are required")
        finite_nonnegative = (
            self.calibration_max_relative_spread,
            self.calibration_scale_floor,
            self.hessian_max_condition_number,
            self.admm_min_relative_improvement,
            self.export_max_error_ratio,
            self.retry_min_relative_improvement_per_bit,
            self.outlier_min_relative_block_improvement,
            self.tuning_min_recovery_fraction,
        )
        if any(not math.isfinite(value) or value < 0 for value in finite_nonnegative):
            raise ValueError("diagnostic policy thresholds must be finite and non-negative")
        if self.calibration_scale_floor == 0 or self.export_max_error_ratio < 1:
            raise ValueError("diagnostic scale floor must be positive and export ratio at least one")
        if self.admm_plateau_window < 2 or self.hessian_max_jitter_escalations < 0:
            raise ValueError("diagnostic window/escalation thresholds are invalid")
        if self.runtime_max_fallback_count < 0:
            raise ValueError("diagnostic fallback threshold must be non-negative")

    @property
    def semantic_key(self) -> str:
        return "sha256:" + hashlib.sha256(canonical_json(self).encode()).hexdigest()


DEFAULT_DIAGNOSTIC_POLICY = DiagnosticPolicy()


@dataclass(frozen=True, slots=True)
class DiagnosticObservation:
    location: str
    calibration_partition_statistics: tuple[float, ...] = ()
    hessian_condition_number: float | None = None
    hessian_jitter_escalations: int | None = None
    admm_residuals: tuple[float, ...] = ()
    latent_error: float | None = None
    exported_error: float | None = None
    retry_error_before: float | None = None
    retry_error_after: float | None = None
    retry_added_bits: int | None = None
    outlier_bits: int | None = None
    block_loss_without_outliers: float | None = None
    block_loss_with_outliers: float | None = None
    block_entry_loss: float | None = None
    post_quantization_loss: float | None = None
    post_tuning_loss: float | None = None
    runtime_fallback_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticFinding:
    code: str
    severity: str
    location: str
    message: str
    evidence: tuple[tuple[str, object], ...]
    artifact_valid: bool
    recommended_next_diagnostic: str


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    policy_key: str
    findings: tuple[DiagnosticFinding, ...]


def _finding(
    code: str,
    severity: str,
    observation: DiagnosticObservation,
    message: str,
    evidence: tuple[tuple[str, object], ...],
    artifact_valid: bool,
    recommendation: str,
) -> DiagnosticFinding:
    return DiagnosticFinding(
        code,
        severity,
        observation.location,
        message,
        evidence,
        artifact_valid,
        recommendation,
    )


def _relative_improvement(before: float, after: float, floor: float) -> float:
    return (before - after) / max(abs(before), floor)


def diagnose(
    observation: DiagnosticObservation,
    policy: DiagnosticPolicy = DEFAULT_DIAGNOSTIC_POLICY,
) -> DiagnosticResult:
    if not observation.location:
        raise ValueError("diagnostic observation location is required")
    findings: list[DiagnosticFinding] = []
    calibration = observation.calibration_partition_statistics
    if calibration:
        if not all(math.isfinite(value) for value in calibration):
            findings.append(
                _finding(
                    "NQ-CAL-001",
                    "error",
                    observation,
                    "calibration statistic contains non-finite values",
                    (("partition_statistics", calibration),),
                    False,
                    "replay calibration with per-partition statistic and input-range capture",
                )
            )
        elif len(calibration) > 1:
            mean = statistics.fmean(calibration)
            relative_spread = (max(calibration) - min(calibration)) / max(
                abs(mean), policy.calibration_scale_floor
            )
            if relative_spread > policy.calibration_max_relative_spread:
                findings.append(
                    _finding(
                        "NQ-CAL-003",
                        "warning",
                        observation,
                        "calibration statistics are unstable across partitions",
                        (("relative_spread", relative_spread), ("partition_count", len(calibration))),
                        True,
                        "compare partition token diversity, clipping, and sample-order sensitivity",
                    )
                )

    condition = observation.hessian_condition_number
    escalations = observation.hessian_jitter_escalations
    poor_condition = condition is not None and (
        not math.isfinite(condition) or condition > policy.hessian_max_condition_number
    )
    repeated_jitter = escalations is not None and escalations > policy.hessian_max_jitter_escalations
    if poor_condition or repeated_jitter:
        findings.append(
            _finding(
                "NQ-HES-001",
                "warning",
                observation,
                "Hessian conditioning required excessive regularization",
                (("condition_number", condition), ("jitter_escalations", escalations)),
                True,
                "inspect eigenvalue range and compare diagonal or low-rank objective fallback",
            )
        )

    residuals = observation.admm_residuals
    if len(residuals) >= policy.admm_plateau_window and all(math.isfinite(value) for value in residuals):
        tail = residuals[-policy.admm_plateau_window :]
        improvement = _relative_improvement(tail[0], tail[-1], policy.calibration_scale_floor)
        if improvement < policy.admm_min_relative_improvement:
            findings.append(
                _finding(
                    "NQ-FAC-001",
                    "warning",
                    observation,
                    "ADMM residual plateaued or diverged",
                    (("tail", tail), ("relative_improvement", improvement)),
                    True,
                    "inspect primal/dual residual traces and rho/iteration sensitivity",
                )
            )

    latent = observation.latent_error
    exported = observation.exported_error
    if latent is not None and exported is not None and all(math.isfinite(value) for value in (latent, exported)):
        ratio = exported / max(abs(latent), policy.calibration_scale_floor)
        if ratio > policy.export_max_error_ratio:
            findings.append(
                _finding(
                    "NQ-FAC-002",
                    "warning",
                    observation,
                    "export error materially exceeds latent error",
                    (("latent_error", latent), ("exported_error", exported), ("ratio", ratio)),
                    True,
                    "compare latent, sign-export, scale-fit, and packed-reference errors",
                )
            )

    retry_values = (
        observation.retry_error_before,
        observation.retry_error_after,
        observation.retry_added_bits,
    )
    if all(value is not None for value in retry_values):
        before = observation.retry_error_before
        after = observation.retry_error_after
        bits = observation.retry_added_bits
        assert before is not None and after is not None and bits is not None
        if bits > 0 and before > 0 and all(math.isfinite(value) for value in (before, after)):
            per_bit = _relative_improvement(before, after, policy.calibration_scale_floor) / bits
            if per_bit < policy.retry_min_relative_improvement_per_bit:
                findings.append(
                    _finding(
                        "NQ-RNK-002",
                        "warning",
                        observation,
                        "retry added bits without sufficient reconstruction improvement",
                        (("added_bits", bits), ("relative_improvement_per_bit", per_bit)),
                        True,
                        "compare retry rank utility with neighboring layers and the global bit budget",
                    )
                )

    outlier_values = (
        observation.outlier_bits,
        observation.block_loss_without_outliers,
        observation.block_loss_with_outliers,
    )
    if all(value is not None for value in outlier_values):
        bits = observation.outlier_bits
        without = observation.block_loss_without_outliers
        with_outliers = observation.block_loss_with_outliers
        assert bits is not None and without is not None and with_outliers is not None
        if bits > 0 and without > 0 and all(math.isfinite(value) for value in (without, with_outliers)):
            improvement = _relative_improvement(
                without, with_outliers, policy.calibration_scale_floor
            )
            if improvement < policy.outlier_min_relative_block_improvement:
                findings.append(
                    _finding(
                        "NQ-RNK-003",
                        "warning",
                        observation,
                        "outlier allocation consumes bits without sufficient block-loss benefit",
                        (("outlier_bits", bits), ("relative_block_improvement", improvement)),
                        True,
                        "compare block loss with outliers disabled and reallocate the measured bit cost",
                    )
                )

    tuning_values = (
        observation.block_entry_loss,
        observation.post_quantization_loss,
        observation.post_tuning_loss,
    )
    if all(value is not None for value in tuning_values):
        entry = observation.block_entry_loss
        quantized = observation.post_quantization_loss
        tuned = observation.post_tuning_loss
        assert entry is not None and quantized is not None and tuned is not None
        if all(math.isfinite(value) for value in (entry, quantized, tuned)):
            jump = quantized - entry
            if jump > policy.calibration_scale_floor:
                recovery = (quantized - tuned) / jump
                if recovery < policy.tuning_min_recovery_fraction:
                    findings.append(
                        _finding(
                            "NQ-TUN-002",
                            "warning",
                            observation,
                            "tuning recovered too little of the quantization loss jump",
                            (("quantization_jump", jump), ("recovery_fraction", recovery)),
                            True,
                            "inspect tuning loss trajectory, best-state restore, and block-boundary targets",
                        )
                    )

    if len(observation.runtime_fallback_reasons) > policy.runtime_max_fallback_count:
        findings.append(
            _finding(
                "NQ-INF-001",
                "warning",
                observation,
                "optimized runtime backend used unexpected fallbacks",
                (("fallback_reasons", observation.runtime_fallback_reasons),),
                True,
                "inspect per-layer capability rejection codes and packed layout compatibility",
            )
        )
    return DiagnosticResult(policy.semantic_key, tuple(findings))
