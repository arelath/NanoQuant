from dataclasses import replace
from pathlib import Path

from nanoquant.application.planning import PlanningRequest, build_quantization_plan, persist_plan
from nanoquant.config.schema import (
    AllocationStrategy,
    BiasCorrectionConfig,
    LayerRankBudgetConfig,
    LowRankPatchConfig,
    OutlierConfig,
    RankAllocationConfig,
    RankBoundsConfig,
    RankRetryConfig,
)
from nanoquant.domain.models import (
    ArtifactRef,
    BlockId,
    BlockInventory,
    CalibrationStats,
    ComponentRef,
    DatasetIdentity,
    LayerCalibrationStats,
    LayerId,
    LayerInventory,
    ModelIdentity,
    ModelInventory,
    ObjectiveSpec,
    SourceTensor,
    StatisticSummary,
    TensorId,
    TensorRef,
    TensorSpec,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore


def _request(strategy: AllocationStrategy = AllocationStrategy.UNIFORM) -> PlanningRequest:
    component = ComponentRef("tiny", "1")
    model = ModelIdentity("fixture", "rev", "hash", "fixture", "tok", component)
    artifact = ArtifactRef("fixture", "sha256-" + "0" * 64, 1)
    blocks = []
    stats = []
    objectives = []
    for index, importance in enumerate((1.0, 10.0)):
        path = "self_attn.v_proj" if index == 0 else "self_attn.q_proj"
        layer_id = LayerId(BlockId(index), path)
        source = SourceTensor(
            TensorId(layer_id, "weight"),
            f"blocks.{index}.weight",
            "model.safetensors",
            TensorSpec((64, 64), "float32"),
            f"hash-{index}",
        )
        layer = LayerInventory(layer_id, source, None, 64, 64)
        blocks.append(BlockInventory(BlockId(index), (source,), (layer,)))
        tensor = TensorRef(artifact, f"importance-{index}", TensorSpec((64,), "float32"), f"tensor-{index}")
        summary = StatisticSummary(importance, importance, importance, 0, 0)
        stats.append(LayerCalibrationStats(layer_id, tensor, tensor, None, summary, summary))
        objectives.append(
            ObjectiveSpec(
                1, layer_id, "diagonal", tensor, tensor, None, 0.01, "target_weighted_norm_squared", None, artifact
            )
        )
    inventory = ModelInventory(1, model, tuple(blocks), (), 32768)
    dataset = DatasetIdentity("data", ("fixture",), ("rev",), "tok", "v1")
    calibration = CalibrationStats(1, component, model, dataset, "online_fisher", "float32", tuple(stats), 2, 128)
    allocation = RankAllocationConfig(
        target_bpw=1.0,
        strategy=strategy,
        bounds=RankBoundsConfig(
            multiple=1, floor_fraction_of_uniform=0.5, ceiling_fraction_of_uniform=2.0, edge_block_boost=0
        ),
    )
    return PlanningRequest(inventory, calibration, artifact, tuple(objectives), allocation, OutlierConfig())


def test_uniform_plan_is_complete_budgeted_and_persisted_before_mutation(tmp_path: Path) -> None:
    request = _request()
    plan = build_quantization_plan(request)
    assert len(plan.blocks) == 2 and all(len(block.layers) == 1 for block in plan.blocks)
    ranks = [layer.rank for block in plan.blocks for layer in block.layers]
    assert ranks[0] == ranks[1]
    assert plan.planned_cost.total <= 64 * 64 * 2
    stored = persist_plan(plan, LocalArtifactStore(tmp_path / "artifacts"))
    assert stored.plan == plan


def test_sensitivity_plan_allocates_no_less_rank_to_more_sensitive_layer() -> None:
    plan = build_quantization_plan(_request(AllocationStrategy.SENSITIVITY))
    ranks = [layer.rank for block in plan.blocks for layer in block.layers]
    assert ranks[1] >= ranks[0]


def test_bias_and_patch_bits_are_funded_by_reducing_factor_rank() -> None:
    request = _request()
    allocation = replace(request.allocation, target_bpw=2.0)
    baseline = build_quantization_plan(replace(request, allocation=allocation))
    sided = build_quantization_plan(
        replace(
            request,
            allocation=allocation,
            bias_correction=BiasCorrectionConfig(enabled=True),
            low_rank_patch=LowRankPatchConfig(enabled=True, layer_patterns=("*",), rank=1),
        )
    )
    baseline_layers = [layer for block in baseline.blocks for layer in block.layers]
    sided_layers = [layer for block in sided.blocks for layer in block.layers]

    assert all(layer.estimated_cost.bias_bits == 64 * 16 for layer in sided_layers)
    assert all(layer.estimated_cost.patch_bits == 16 * (64 + 64) for layer in sided_layers)
    assert all(
        sided_layer.rank < baseline_layer.rank
        for sided_layer, baseline_layer in zip(sided_layers, baseline_layers, strict=True)
    )
    assert sided.planned_cost.total <= int(2.0 * 2 * 64 * 64)


def test_edge_boost_changes_utility_without_relaxing_rank_ceiling() -> None:
    request = _request(AllocationStrategy.SENSITIVITY)
    unboosted = build_quantization_plan(request)
    boosted_allocation = replace(
        request.allocation,
        bounds=replace(request.allocation.bounds, edge_block_boost=0.75),
    )
    boosted = build_quantization_plan(replace(request, allocation=boosted_allocation))

    assert [layer.allocator_cap for block in boosted.blocks for layer in block.layers] == [
        layer.allocator_cap for block in unboosted.blocks for layer in block.layers
    ]


def test_uniform_bpw_uses_shape_specific_ranks_for_heterogeneous_layers() -> None:
    request = _request()
    dimensions = ((1152, 6912), (1152, 256))
    blocks = []
    for block, (inputs, outputs) in zip(request.inventory.blocks, dimensions, strict=True):
        layer = block.quantizable_layers[0]
        source = replace(layer.weight, spec=TensorSpec((outputs, inputs), "float32"))
        updated = replace(layer, weight=source, in_features=inputs, out_features=outputs)
        blocks.append(replace(block, source_tensors=(source,), quantizable_layers=(updated,)))
    inventory = replace(request.inventory, blocks=tuple(blocks))
    allocation = replace(request.allocation, bounds=replace(request.allocation.bounds, multiple=32))

    plan = build_quantization_plan(replace(request, inventory=inventory, allocation=allocation))
    ranks = [layer.rank for block in plan.blocks for layer in block.layers]
    elements = sum(inputs * outputs for inputs, outputs in dimensions)

    assert ranks[0] > ranks[1]
    assert ranks == [960, 160]
    assert 0.98 <= plan.planned_cost.total / elements <= 1.0


def test_utility_profile_must_cover_every_layer() -> None:
    request = _request(AllocationStrategy.UTILITY_PROFILE)
    try:
        build_quantization_plan(request)
    except ValueError as error:
        assert "missing layer" in str(error)
    else:
        raise AssertionError("incomplete utility profile was accepted")


def test_retry_plan_honors_enabled_and_above_allocator_cap_policy() -> None:
    request = _request()
    bounded = build_quantization_plan(request).blocks[0].layers[0]
    assert bounded.retry.maximum_attempts == 2
    assert bounded.retry.hard_rank_cap == bounded.allocator_cap

    disabled = replace(request.allocation, retry=RankRetryConfig(enabled=False, maximum_attempts=3))
    disabled_layer = build_quantization_plan(replace(request, allocation=disabled)).blocks[0].layers[0]
    assert disabled_layer.retry.maximum_attempts == 1

    above_cap = replace(
        request.allocation,
        retry=RankRetryConfig(enabled=True, maximum_attempts=3, allow_above_allocator_cap=True),
    )
    above_cap_layer = build_quantization_plan(replace(request, allocation=above_cap)).blocks[0].layers[0]
    assert above_cap_layer.retry.maximum_attempts == 3
    assert above_cap_layer.retry.hard_rank_cap == 64
    assert above_cap_layer.retry.hard_rank_cap >= above_cap_layer.allocator_cap


def test_additive_maximum_rank_pattern_overrides_rank_after_budget_allocation() -> None:
    request = _request(AllocationStrategy.SENSITIVITY)
    baseline = build_quantization_plan(request)
    overridden = build_quantization_plan(
        replace(
            request,
            allocation=replace(
                request.allocation,
                maximum_rank_layer_patterns=("self_attn.v_proj",),
            ),
        )
    )

    baseline_ranks = [layer.rank for block in baseline.blocks for layer in block.layers]
    overridden_layers = [layer for block in overridden.blocks for layer in block.layers]
    assert baseline_ranks[0] < 64
    assert [layer.rank for layer in overridden_layers] == [64, baseline_ranks[1]]
    assert overridden_layers[0].allocator_cap == 64
    nominal_target_bits = sum(
        layer.in_features * layer.out_features
        for block in request.inventory.blocks
        for layer in block.quantizable_layers
    )
    assert overridden.planned_cost.total > nominal_target_bits


def test_additive_maximum_rank_pattern_must_match_a_quantizable_layer() -> None:
    request = _request()

    try:
        build_quantization_plan(
            replace(
                request,
                allocation=replace(
                    request.allocation,
                    maximum_rank_layer_patterns=("self_attn.k_proj",),
                ),
            )
        )
    except ValueError as error:
        assert "matched no quantizable layer" in str(error)
    else:
        raise AssertionError("unmatched maximum-rank pattern was accepted")


def test_additive_layer_budget_multiplier_promotes_only_matching_rank() -> None:
    request = _request(AllocationStrategy.SENSITIVITY)
    baseline = build_quantization_plan(request)
    promoted = build_quantization_plan(
        replace(
            request,
            allocation=replace(
                request.allocation,
                layer_budget_multipliers=(LayerRankBudgetConfig("self_attn.q_proj", 1.25),),
            ),
        )
    )

    baseline_layers = [layer for block in baseline.blocks for layer in block.layers]
    promoted_layers = [layer for block in promoted.blocks for layer in block.layers]
    assert promoted_layers[0].rank == baseline_layers[0].rank
    assert promoted_layers[1].rank > baseline_layers[1].rank
    baseline_cost = baseline_layers[1].estimated_cost
    promoted_cost = promoted_layers[1].estimated_cost
    baseline_factor_bits = baseline_cost.binary_factor_bits + baseline_cost.scale_bits + baseline_cost.padding_bits
    promoted_factor_bits = promoted_cost.binary_factor_bits + promoted_cost.scale_bits + promoted_cost.padding_bits
    assert promoted_factor_bits <= baseline_factor_bits * 1.25
    assert promoted_layers[1].allocator_cap >= promoted_layers[1].rank


def test_layer_budget_multiplier_must_match_exactly_one_pattern() -> None:
    request = _request()
    unmatched = replace(
        request.allocation,
        layer_budget_multipliers=(LayerRankBudgetConfig("self_attn.k_proj", 1.25),),
    )
    overlapping = replace(
        request.allocation,
        layer_budget_multipliers=(
            LayerRankBudgetConfig("self_attn.*", 1.25),
            LayerRankBudgetConfig("self_attn.q_proj", 1.5),
        ),
    )

    try:
        build_quantization_plan(replace(request, allocation=unmatched))
    except ValueError as error:
        assert "matched no quantizable layer" in str(error)
    else:
        raise AssertionError("unmatched layer-budget pattern was accepted")
    try:
        build_quantization_plan(replace(request, allocation=overlapping))
    except ValueError as error:
        assert "multiple layer-budget patterns" in str(error)
    else:
        raise AssertionError("overlapping layer-budget patterns were accepted")
