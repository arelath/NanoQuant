import math
from dataclasses import replace

import pytest

from nanoquant.config.schema import (
    ADMMConfig,
    AllocationStrategy,
    LayerRankBudgetConfig,
    ModelConfig,
    RankResponseCurveConfig,
    RankResponseSegmentConfig,
    ReconstructionImportanceConfig,
    ReconstructionRankPlanningConfig,
    RunConfig,
)
from nanoquant.config.validation import validate
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.domain.planning import (
    RankResponseSegment,
    ReconstructionAllocationUnit,
    allocate_reconstruction_rank_budget,
    integrated_rank_response,
    predicted_squared_error,
)
from nanoquant.resident_quantization import _reconstruction_importance_policy


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
