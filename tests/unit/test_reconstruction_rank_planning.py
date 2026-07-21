import math
from dataclasses import replace

import pytest

from nanoquant.config.schema import (
    ADMMConfig,
    AllocationStrategy,
    KlAllocationObjective,
    KlSensitivityGranularity,
    LayerRankBudgetConfig,
    ModelConfig,
    RankResponseCurveConfig,
    RankResponseSegmentConfig,
    RankResponseSource,
    ReconstructionImportanceConfig,
    ReconstructionRankPlanningConfig,
    RunConfig,
)
from nanoquant.config.validation import validate
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.domain.planning import (
    RankResponseSegment,
    ReconstructionAllocationDecision,
    ReconstructionAllocationResult,
    ReconstructionAllocationUnit,
    allocate_reconstruction_rank_budget,
    apply_reconstruction_rank_trust_region,
    integrated_rank_response,
    predicted_squared_error,
)
from nanoquant.resident_quantization import _measured_response_curve, _RankProbePoint, _reconstruction_importance_policy


def test_piecewise_response_is_continuous_and_uses_squared_error_slope() -> None:
    segments = (
        RankResponseSegment(1.0, 0.01),
        RankResponseSegment(2.0, 0.005),
    )
    unit = ReconstructionAllocationUnit(
        "0:self_attn.attn_qkv",
        128,
        128,
        100,
        4.0,
        1.0,
        False,
        0.5,
        2.0,
        segments,
    )

    assert integrated_rank_response(100, 50, 0.5, segments) == pytest.approx(-0.5)
    assert integrated_rank_response(100, 100, 0.5, segments) == 0
    assert integrated_rank_response(100, 150, 0.5, segments) == pytest.approx(0.25)
    assert predicted_squared_error(unit, 100) == 4.0
    assert predicted_squared_error(unit, 150) == pytest.approx(4.0 * math.exp(-0.5))


def test_allocator_protects_sensitive_units_and_never_exceeds_exact_budget() -> None:
    segments = (RankResponseSegment(1.5, 0.01),)
    units = (
        ReconstructionAllocationUnit("0:sensitive", 64, 64, 32, 100.0, 4.0, True, 0.5, 1.5, segments),
        ReconstructionAllocationUnit("0:ordinary", 64, 64, 32, 100.0, 1.0, False, 0.5, 1.5, segments),
    )
    result = allocate_reconstruction_rank_budget(
        units,
        target_bits=14_000,
        multiple=8,
        floor_fraction=0.5,
        ceiling_fraction=1.5,
        sensitivity_strength=0.5,
        protected_rank_floor_fraction=1.0,
        target_protected_error_reduction_fraction=0,
    )
    decisions = {decision.unit_id: decision for decision in result.decisions}

    assert decisions["0:sensitive"].planned_rank >= 32
    assert result.spent_bits <= 14_000
    assert result.remaining_bits == 14_000 - result.spent_bits
    assert result.protected_planned_objective <= result.protected_baseline_objective


def test_rank_trust_region_projects_and_trims_to_exact_budget() -> None:
    segments = (RankResponseSegment(1.5, 0.01),)
    units = (
        ReconstructionAllocationUnit("0:a", 64, 64, 32, 100.0, 4.0, False, 0.5, 1.5, segments),
        ReconstructionAllocationUnit("0:b", 64, 64, 32, 100.0, 1.0, False, 0.5, 1.5, segments),
    )
    unconstrained = ReconstructionAllocationResult(
        (
            ReconstructionAllocationDecision("0:a", 32, 48, 100.0, predicted_squared_error(units[0], 48)),
            ReconstructionAllocationDecision("0:b", 32, 16, 100.0, predicted_squared_error(units[1], 16)),
        ),
        0,
        0,
        0,
        0,
    )

    projected = apply_reconstruction_rank_trust_region(
        units,
        unconstrained,
        (("0:a", 32), ("0:b", 32)),
        13_000,
        multiple=8,
        floor_fraction=0.5,
        ceiling_fraction=1.5,
        sensitivity_strength=0.5,
        protected_rank_floor_fraction=1.0,
        target_protected_error_reduction_fraction=0,
        step_fraction=0.5,
    )
    ranks = {decision.unit_id: decision.planned_rank for decision in projected.decisions}

    assert projected.spent_bits <= 13_000
    assert projected.remaining_bits == 13_000 - projected.spent_bits
    assert ranks["0:a"] <= 40
    assert ranks["0:b"] <= 32
    assert ranks != {"0:a": 48, "0:b": 16}

    fixed = apply_reconstruction_rank_trust_region(
        units,
        unconstrained,
        (("0:a", 32), ("0:b", 32)),
        13_312,
        multiple=8,
        floor_fraction=0.5,
        ceiling_fraction=1.5,
        sensitivity_strength=0.5,
        protected_rank_floor_fraction=1.0,
        target_protected_error_reduction_fraction=0,
        step_fraction=0,
    )
    assert {decision.unit_id: decision.planned_rank for decision in fixed.decisions} == {
        "0:a": 32,
        "0:b": 32,
    }


def test_architecture_importance_protects_selected_layers_and_first_last_blocks() -> None:
    layers = tuple(
        LayerId(BlockId(block), path)
        for block in range(4)
        for path in ("self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj")
    )
    importance = ReconstructionImportanceConfig(
        layer_multipliers=(
            LayerRankBudgetConfig("self_attn.q_proj", 1.25),
            LayerRankBudgetConfig("mlp.down_proj", 1.25),
        ),
        protected_layer_patterns=("self_attn.q_proj", "mlp.down_proj"),
        edge_block_multiplier=1.5,
        protected_edge_block_count=1,
    )

    multipliers, protected, edge_blocks = _reconstruction_importance_policy(layers, importance)

    assert edge_blocks == {0, 3}
    assert multipliers[LayerId(BlockId(0), "self_attn.q_proj")] == pytest.approx(1.875)
    assert multipliers[LayerId(BlockId(1), "self_attn.q_proj")] == pytest.approx(1.25)
    assert multipliers[LayerId(BlockId(0), "mlp.gate_proj")] == pytest.approx(1.5)
    assert LayerId(BlockId(1), "self_attn.q_proj") in protected
    assert LayerId(BlockId(1), "mlp.down_proj") in protected
    assert LayerId(BlockId(0), "mlp.gate_proj") in protected
    assert LayerId(BlockId(3), "mlp.gate_proj") in protected
    assert LayerId(BlockId(2), "mlp.gate_proj") not in protected


def test_measured_response_curve_fits_current_unit_weighted_error_ratios() -> None:
    points = (
        _RankProbePoint(16, 30.0, 0.30, 0.36),
        _RankProbePoint(32, 20.0, 0.20, 0.20),
        _RankProbePoint(48, 15.0, 0.15, 0.12),
    )

    curve = _measured_response_curve("physical", 32, points)
    segments = tuple(RankResponseSegment(item.maximum_rank_fraction, item.beta_per_rank) for item in curve.segments)
    unit = ReconstructionAllocationUnit("0:physical", 64, 64, 32, 1.0, 2.0, False, 0.5, 1.5, segments)

    assert predicted_squared_error(unit, 16) == pytest.approx(0.36 / 0.20)
    assert predicted_squared_error(unit, 32) == pytest.approx(1.0)
    assert predicted_squared_error(unit, 48) == pytest.approx(0.12 / 0.20)


def test_measured_response_curve_does_not_invent_gain_from_noisy_regression() -> None:
    points = (
        _RankProbePoint(16, 30.0, 0.30, 0.30),
        _RankProbePoint(32, 20.0, 0.20, 0.20),
        _RankProbePoint(48, 21.0, 0.21, 0.21),
    )

    curve = _measured_response_curve("physical", 32, points)

    assert curve.segments[-1].beta_per_rank == 0


def test_measured_unit_kl_config_requires_exact_untempered_same_run_response() -> None:
    base = RunConfig(ModelConfig("fixture"))
    reconstruction = ReconstructionRankPlanningConfig(
        enabled=True,
        objective_mode="calibration_weighted",
        probe_admm=ADMMConfig(outer_iterations=1, inner_iterations=1),
        response_source=RankResponseSource.MEASURED,
        kl_objective=KlAllocationObjective.MEASURED_UNIT_KL,
        sensitivity_strength=1,
        target_protected_error_reduction_fraction=0,
    )
    valid = replace(
        base,
        allocation=replace(
            base.allocation,
            strategy=AllocationStrategy.KL_CALIBRATED,
            kl_profile_artifact="fresh-profile",
            kl_profile_key="sha256:fresh",
            kl_sensitivity_granularity=KlSensitivityGranularity.EXACT,
            reconstruction=reconstruction,
        ),
    )

    assert validate(valid) == ()
    tempered = replace(
        valid,
        allocation=replace(
            valid.allocation,
            reconstruction=replace(reconstruction, sensitivity_strength=0.75),
        ),
    )
    assert "CFG095" in {issue.code for issue in validate(tempered)}


def test_reconstruction_config_round_trips_and_requires_explicit_evidence() -> None:
    base = RunConfig(ModelConfig("fixture"))
    curve = RankResponseCurveConfig(
        "mlp.down_proj",
        0.6,
        1.4,
        (RankResponseSegmentConfig(1.4, 6.22e-4),),
    )
    reconstruction = ReconstructionRankPlanningConfig(
        enabled=True,
        probe_admm=ADMMConfig(outer_iterations=400, transpose_wide=True),
        response_curves=(curve,),
        response_profile_provenance="Docs/ImprovementSuggestions/ReconstructionHeadroom.md#8",
    )
    valid = replace(
        base,
        allocation=replace(
            base.allocation,
            strategy=AllocationStrategy.RECONSTRUCTION_AWARE,
            bounds=replace(
                base.allocation.bounds,
                floor_fraction_of_uniform=0.6,
                ceiling_fraction_of_uniform=1.4,
            ),
            reconstruction=reconstruction,
        ),
    )

    assert validate(valid) == ()
    invalid = replace(
        valid, allocation=replace(valid.allocation, reconstruction=replace(reconstruction, enabled=False))
    )
    assert "CFG049" in {issue.code for issue in validate(invalid)}


def test_rank_trust_region_requires_kl_strategy_and_reference() -> None:
    base = RunConfig(ModelConfig("fixture"))
    reconstruction = replace(
        base.allocation.reconstruction,
        rank_trust_fraction=0.25,
    )
    missing_reference = replace(
        base,
        allocation=replace(base.allocation, reconstruction=reconstruction),
    )
    non_kl_reference = replace(
        missing_reference,
        allocation=replace(
            missing_reference.allocation,
            reconstruction=replace(reconstruction, rank_trust_reference_run="evidence/016"),
        ),
    )

    assert "CFG090" in {issue.code for issue in validate(missing_reference)}
    assert "CFG091" in {issue.code for issue in validate(non_kl_reference)}
