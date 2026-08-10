from __future__ import annotations

from dataclasses import replace

from nanoquant.application.product_codebook_allocation import (
    ProductCodebookAllocationGroup,
    ProductCodebookAllocationOption,
    allocate_product_codebook_options,
    validate_product_codebook_plan,
)
from nanoquant.config.schema import ProductCodebookConfig
from nanoquant.domain.models import (
    ArtifactRef,
    ArtifactTypes,
    BitCost,
    BlockId,
    BlockPlan,
    ComponentRef,
    LayerId,
    LayerPlan,
    ModelIdentity,
    ObjectiveSpec,
    OutlierPlan,
    QuantizationPlan,
    RetryPolicy,
    SourceTensor,
    TensorId,
    TensorSpec,
)
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.resident_quantization import (
    _product_codebook_option_cost,
    _product_codebook_probe_candidates,
)


def _artifact(name: str, artifact_type: ArtifactTypes = ArtifactTypes.QUANTIZATION_PLAN) -> ArtifactRef:
    return ArtifactRef(artifact_type, f"sha256-{name:0<64}"[:71], 1)


def _layer(block: BlockId, path: str, bits: int) -> LayerPlan:
    layer = LayerId(block, path)
    source = SourceTensor(
        TensorId(layer, "weight"),
        f"{path}.weight",
        "fixture.safetensors",
        TensorSpec((10, 10), "float16"),
        f"sha256:{path}",
    )
    calibration = _artifact("calibration")
    objective = ObjectiveSpec(
        1,
        layer,
        "diagonal",
        # The allocator never reads objective tensors; a typed fixture still
        # keeps the plan contract complete.
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,
        0.0,
        "none",
        None,
        calibration,
    )
    return LayerPlan(
        1,
        layer,
        source,
        4,
        1,
        8,
        objective,
        OutlierPlan("none", 0, "float16", True),
        RetryPolicy(1, 0.0, None, None, 8, 0),
        BitCost(binary_factor_bits=bits),
    )


def _plan() -> QuantizationPlan:
    block = BlockId(0)
    first = _layer(block, "mlp.up_proj", 80)
    second = _layer(block, "mlp.down_proj", 80)
    fixed = _layer(block, "self_attn.o_proj", 20)
    return QuantizationPlan(
        2,
        ComponentRef("planner", "2"),
        ModelIdentity("fixture", "rev", "config", "fixture", "rev", ComponentRef("adapter", "1")),
        _artifact("calibration"),
        (BlockPlan(block, tuple(layer.layer for layer in (first, second, fixed)), (first, second, fixed), 0),),
        0.5,
        BitCost(binary_factor_bits=180),
    )


def _option(layer: LayerId, rank: int, free_rows: int | None, bits: int, error: float, name: str):
    return ProductCodebookAllocationOption(
        layer,
        rank,
        free_rows,
        BitCost(binary_factor_bits=bits),
        error,
        1.0,
        _artifact(name, ArtifactTypes.PRODUCT_CODEBOOK_ALLOCATION_PROFILE),
    )


def test_global_allocation_trades_bits_between_layers_and_rewrites_exact_costs() -> None:
    plan = _plan()
    first, second, _fixed = plan.blocks[0].layers
    groups = (
        ProductCodebookAllocationGroup(
            first.layer,
            4,
            (
                _option(first.layer, 4, 2, 40, 5.0, "first-cheap"),
                _option(first.layer, 8, 5, 80, 1.0, "first-rich"),
            ),
        ),
        ProductCodebookAllocationGroup(
            second.layer,
            4,
            (
                _option(second.layer, 4, 2, 40, 2.0, "second-cheap"),
                _option(second.layer, 8, 5, 80, 1.5, "second-rich"),
            ),
        ),
    )
    profile = _artifact("profile", ArtifactTypes.PRODUCT_CODEBOOK_ALLOCATION_PROFILE)

    result = allocate_product_codebook_options(plan, groups, profile)

    # 300 total weights at 0.5 BPW gives 150 bits; 20 are fixed.  Spending the
    # one available 40-bit upgrade on the first layer buys the larger gain.
    assert result.budget_bits == 150
    assert result.selected_bits == 140
    selected = {decision.layer.path: decision for decision in result.decisions}
    assert selected["mlp.up_proj"].selected_rank == 8
    assert selected["mlp.down_proj"].selected_rank == 4
    assert result.plan.planned_cost.total == 140
    assert result.plan.schema_version == 3
    validate_product_codebook_plan(
        result.plan,
        eligible_layers=(first.layer, second.layer),
    )


def test_validator_rejects_an_unallocated_product_plan() -> None:
    plan = _plan()
    eligible = tuple(layer.layer for layer in plan.blocks[0].layers[:2])

    try:
        validate_product_codebook_plan(plan, eligible_layers=eligible)
    except ValueError as error:
        assert "measured allocation profile" in str(error)
    else:
        raise AssertionError("unallocated product plan unexpectedly validated")


def test_probe_grid_jointly_varies_rank_and_free_rows_with_a_free_control() -> None:
    plan = _plan()
    config = ProductCodebookConfig(
        enabled=True,
        free_row_multiple=1,
        minimum_coded_rows=1,
        rank_span_fractions=(0.0, 1.0),
        free_row_fractions=(0.5,),
    )

    candidates = _product_codebook_probe_candidates(plan, config)
    first = [candidate for candidate in candidates if candidate.layer.path == "mlp.up_proj"]

    assert [(candidate.rank, candidate.free_rows) for candidate in first] == [
        (4, None),
        (4, 2),
        (8, 4),
    ]
    layer = replace(
        plan.blocks[0].layers[0],
        estimated_cost=factor_bit_cost(10, 10, 4),
    )
    assert _product_codebook_option_cost(layer, first[0], config) == layer.estimated_cost
    assert _product_codebook_option_cost(layer, first[-1], config).total > 0


def test_global_budget_preserves_declared_uncharged_outlier_sidecars() -> None:
    plan = _plan()
    first, second, fixed = plan.blocks[0].layers
    fixed = replace(
        fixed,
        outliers=replace(fixed.outliers, charge_to_budget=False),
        estimated_cost=fixed.estimated_cost + BitCost(outlier_value_bits=30),
    )
    plan = replace(
        plan,
        blocks=(replace(plan.blocks[0], layers=(first, second, fixed)),),
        planned_cost=plan.planned_cost + BitCost(outlier_value_bits=30),
    )
    groups = tuple(
        ProductCodebookAllocationGroup(
            layer.layer,
            layer.rank,
            (_option(layer.layer, 4, 2, 40, 1.0, f"{layer.layer.path}-product"),),
        )
        for layer in (first, second)
    )

    result = allocate_product_codebook_options(
        plan,
        groups,
        _artifact("sidecar-profile", ArtifactTypes.PRODUCT_CODEBOOK_ALLOCATION_PROFILE),
    )

    assert result.budget_bits == 180
    assert result.selected_bits == 130
