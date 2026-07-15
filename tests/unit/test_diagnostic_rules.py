from __future__ import annotations

import math

import pytest

from nanoquant.application.diagnostics import (
    DiagnosticObservation,
    DiagnosticPolicy,
    diagnose,
)
from nanoquant.infrastructure.diagnostics import get


def test_diagnostic_rules_cover_every_required_failure_family() -> None:
    result = diagnose(
        DiagnosticObservation(
            location="blocks.4.mlp.down_proj",
            calibration_partition_statistics=(1.0, 2.0),
            hessian_condition_number=1e9,
            hessian_jitter_escalations=3,
            admm_residuals=(1.0, 1.0, 1.0, 1.0, 1.0),
            latent_error=1.0,
            exported_error=2.0,
            retry_error_before=1.0,
            retry_error_after=0.9999,
            retry_added_bits=1_000,
            outlier_bits=100,
            block_loss_without_outliers=1.0,
            block_loss_with_outliers=0.999,
            block_entry_loss=1.0,
            post_quantization_loss=2.0,
            post_tuning_loss=1.8,
            runtime_fallback_reasons=("NQ-INF-RANK-ALIGNMENT",),
        )
    )

    assert [finding.code for finding in result.findings] == [
        "NQ-CAL-003",
        "NQ-HES-001",
        "NQ-FAC-001",
        "NQ-FAC-002",
        "NQ-RNK-002",
        "NQ-RNK-003",
        "NQ-TUN-002",
        "NQ-INF-001",
    ]
    assert all(finding.location == "blocks.4.mlp.down_proj" for finding in result.findings)
    assert all(finding.artifact_valid for finding in result.findings)
    assert all(finding.evidence and finding.recommended_next_diagnostic for finding in result.findings)
    assert all(
        get(finding.code).documentation == "Docs/07-observability-and-reporting.md"
        for finding in result.findings
    )


def test_healthy_diagnostic_observation_has_no_findings() -> None:
    result = diagnose(
        DiagnosticObservation(
            location="blocks.0.self_attn.q_proj",
            calibration_partition_statistics=(1.0, 1.01),
            hessian_condition_number=1e3,
            hessian_jitter_escalations=0,
            admm_residuals=(1.0, 0.8, 0.6, 0.4, 0.2),
            latent_error=1.0,
            exported_error=1.1,
            retry_error_before=1.0,
            retry_error_after=0.5,
            retry_added_bits=1_000,
            outlier_bits=100,
            block_loss_without_outliers=1.0,
            block_loss_with_outliers=0.9,
            block_entry_loss=1.0,
            post_quantization_loss=2.0,
            post_tuning_loss=1.2,
        )
    )

    assert result.findings == ()
    assert result.policy_key == DiagnosticPolicy().semantic_key


def test_nonfinite_calibration_is_an_invalid_artifact_finding() -> None:
    finding = diagnose(
        DiagnosticObservation("calibration", calibration_partition_statistics=(1.0, math.nan))
    ).findings[0]

    assert finding.code == "NQ-CAL-001"
    assert finding.severity == "error"
    assert not finding.artifact_valid
    assert get(finding.code).title == "Non-finite calibration statistic"


def test_diagnostic_policy_is_versioned_and_thresholds_are_configurable() -> None:
    default = DiagnosticPolicy()
    relaxed = DiagnosticPolicy(calibration_max_relative_spread=1.0)

    assert default.semantic_key != relaxed.semantic_key
    assert diagnose(
        DiagnosticObservation("calibration", calibration_partition_statistics=(1.0, 2.0)),
        relaxed,
    ).findings == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": ""},
        {"calibration_scale_floor": 0.0},
        {"export_max_error_ratio": 0.5},
        {"admm_plateau_window": 1},
        {"hessian_max_jitter_escalations": -1},
        {"runtime_max_fallback_count": -1},
        {"tuning_min_recovery_fraction": math.nan},
    ],
)
def test_diagnostic_policy_rejects_invalid_thresholds(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DiagnosticPolicy(**overrides)  # type: ignore[arg-type]


def test_diagnostic_observation_requires_location() -> None:
    with pytest.raises(ValueError, match="location"):
        diagnose(DiagnosticObservation(""))
