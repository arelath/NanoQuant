"""Complete immutable quantization planning before model mutation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import OutlierConfig, RankAllocationConfig
from nanoquant.domain.models import (
    ArtifactRef,
    BitCost,
    BlockPlan,
    CalibrationStats,
    ComponentRef,
    LayerPlan,
    ModelInventory,
    ObjectiveSpec,
    OutlierPlan,
    QuantizationPlan,
    RetryPolicy,
)
from nanoquant.domain.planning import factor_bit_cost, outlier_bit_cost
from nanoquant.ports.artifact_store import ArtifactStore


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    inventory: ModelInventory
    calibration: CalibrationStats
    calibration_ref: ArtifactRef
    objectives: tuple[ObjectiveSpec, ...]
    allocation: RankAllocationConfig
    outliers: OutlierConfig
    utility_profile: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class PersistedPlan:
    reference: ArtifactRef
    plan: QuantizationPlan


def _add_costs(costs: list[BitCost]) -> BitCost:
    result = BitCost()
    for cost in costs:
        result = result + cost
    return result


def build_quantization_plan(request: PlanningRequest) -> QuantizationPlan:
    layers = [layer for block in request.inventory.blocks for layer in block.quantizable_layers]
    if not layers:
        raise ValueError("model inventory has no quantizable layers")
    objective_map = {objective.layer: objective for objective in request.objectives}
    calibration_map = {stats.layer: stats for stats in request.calibration.layers}
    if set(objective_map) != {layer.layer for layer in layers}:
        raise ValueError("objective set does not exactly match model inventory")
    total_elements = sum(layer.in_features * layer.out_features for layer in layers)
    target_bits = math.floor(total_elements * request.allocation.target_bpw)
    multiple = request.allocation.bounds.multiple
    utility_profile = dict(request.utility_profile)
    outlier_plans: dict[object, OutlierPlan] = {}
    outlier_costs: dict[object, BitCost] = {}
    for layer in layers:
        count = 0
        if request.outliers.selector.value != "none" and request.outliers.fraction > 0:
            raw = math.ceil(layer.in_features * request.outliers.fraction)
            count = math.ceil(raw / request.outliers.count_multiple) * request.outliers.count_multiple
            count = min(count, layer.in_features)
        plan = OutlierPlan(
            request.outliers.selector.value,
            count,
            request.outliers.storage_dtype.value,
            request.outliers.charge_to_bit_budget,
            request.outliers.removed_column_importance,
        )
        bits = {"bfloat16": 16, "float16": 16, "int8": 8}[request.outliers.storage_dtype.value]
        cost = outlier_bit_cost(layer.out_features, count, value_bits=bits) if count else BitCost()
        outlier_plans[layer.layer] = plan
        outlier_costs[layer.layer] = cost
    base_ranks: dict[object, int] = {}
    for layer in layers:
        layer_target_bits = math.floor(layer.in_features * layer.out_features * request.allocation.target_bpw)
        charged = outlier_costs[layer.layer].total if outlier_plans[layer.layer].charge_to_budget else 0
        base_rank = 0
        for candidate in range(multiple, min(layer.in_features, layer.out_features) + 1, multiple):
            if factor_bit_cost(layer.out_features, layer.in_features, candidate).total + charged > layer_target_bits:
                break
            base_rank = candidate
        if base_rank == 0:
            raise ValueError(f"target bit budget cannot fund minimum aligned factors for {layer.layer}")
        base_ranks[layer.layer] = base_rank
    ranks: dict[object, int] = {}
    caps: dict[object, int] = {}
    utilities: dict[object, float] = {}
    for layer in layers:
        maximum = min(layer.in_features, layer.out_features)
        base_rank = base_ranks[layer.layer]
        if request.allocation.strategy.value == "uniform":
            floor = cap = base_rank
        else:
            floor = max(
                multiple,
                math.floor(base_rank * request.allocation.bounds.floor_fraction_of_uniform / multiple) * multiple,
            )
            cap = min(
                maximum,
                max(
                    multiple,
                    math.floor(base_rank * request.allocation.bounds.ceiling_fraction_of_uniform / multiple) * multiple,
                ),
            )
        ranks[layer.layer] = min(floor, cap)
        caps[layer.layer] = cap
        stats = calibration_map.get(layer.layer)
        sensitivity = 1.0 if stats is None else max(1e-12, stats.input_summary.mean * stats.output_summary.mean)
        profile_key = f"{layer.layer.block.index}:{layer.layer.path}"
        if request.allocation.strategy.value == "utility_profile" and profile_key not in utility_profile:
            raise ValueError(f"utility profile is missing layer {profile_key}")
        utility = utility_profile.get(profile_key, sensitivity)
        if request.allocation.strategy.value != "uniform" and request.allocation.bounds.edge_block_boost > 0:
            last_block = len(request.inventory.blocks) - 1
            edge_distance = min(layer.layer.block.index, last_block - layer.layer.block.index)
            edge_score = 1.0 - edge_distance / max(1.0, last_block / 2.0)
            utility *= 1.0 + request.allocation.bounds.edge_block_boost * max(0.0, edge_score)
        utilities[layer.layer] = utility

    def current_cost(layer: object, rank: int) -> BitCost:
        inventory = next(item for item in layers if item.layer == layer)
        return factor_bit_cost(inventory.out_features, inventory.in_features, rank) + outlier_costs[layer]

    def budget_cost(layer: object, rank: int) -> int:
        inventory = next(item for item in layers if item.layer == layer)
        factor = factor_bit_cost(inventory.out_features, inventory.in_features, rank).total
        outliers = outlier_costs[layer].total if outlier_plans[layer].charge_to_budget else 0
        return factor + outliers

    spent = sum(budget_cost(layer.layer, ranks[layer.layer]) for layer in layers)
    if spent > target_bits:
        raise ValueError(f"minimum rank/outlier plan costs {spent} bits, exceeding target {target_bits}")
    if request.allocation.strategy.value == "sensitivity" and utility_profile:
        costs_per_rank = {
            layer.layer: layer.in_features + layer.out_features + 16
            for layer in layers
        }
        budget_units = sum(base_ranks[layer.layer] * costs_per_rank[layer.layer] for layer in layers)
        score_units = sum(
            base_ranks[layer.layer] * costs_per_rank[layer.layer] * utilities[layer.layer] for layer in layers
        )
        mean_score = score_units / budget_units
        targets = {
            layer.layer: base_ranks[layer.layer] * utilities[layer.layer] / mean_score for layer in layers
        }
        for layer in layers:
            proposed = math.floor(targets[layer.layer] / multiple) * multiple
            ranks[layer.layer] = min(caps[layer.layer], max(ranks[layer.layer], proposed))

        def used_units() -> int:
            return sum(ranks[layer.layer] * costs_per_rank[layer.layer] for layer in layers)

        while used_units() > budget_units:
            candidates = [layer for layer in layers if ranks[layer.layer] - multiple >= max(multiple, math.floor(
                base_ranks[layer.layer] * request.allocation.bounds.floor_fraction_of_uniform / multiple
            ) * multiple)]
            if not candidates:
                break
            victim = max(candidates, key=lambda layer: ranks[layer.layer] - targets[layer.layer])
            ranks[victim.layer] -= multiple
        while True:
            remaining = budget_units - used_units()
            candidates = [
                layer
                for layer in layers
                if ranks[layer.layer] + multiple <= caps[layer.layer]
                and ranks[layer.layer] < targets[layer.layer]
                and costs_per_rank[layer.layer] * multiple <= remaining
            ]
            if not candidates:
                break
            winner = max(candidates, key=lambda layer: targets[layer.layer] - ranks[layer.layer])
            ranks[winner.layer] += multiple
        spent = sum(budget_cost(layer.layer, ranks[layer.layer]) for layer in layers)
    else:
        while True:
            rank_candidates: list[tuple[float, int, str, object, int]] = []
            for layer in layers:
                rank = ranks[layer.layer]
                if rank + multiple > caps[layer.layer]:
                    continue
                marginal = budget_cost(layer.layer, rank + multiple) - budget_cost(layer.layer, rank)
                if spent + marginal <= target_bits:
                    rank_candidates.append(
                        (
                            utilities[layer.layer] / marginal,
                            -layer.layer.block.index,
                            layer.layer.path,
                            layer.layer,
                            marginal,
                        )
                    )
            if not rank_candidates:
                break
            *_, selected, marginal = max(rank_candidates, key=lambda candidate: candidate[:3])
            ranks[selected] += multiple
            spent += marginal
    extra_budget = math.floor(target_bits * request.allocation.retry.extra_bit_budget_fraction)
    block_plans = []
    all_costs = []
    for block in request.inventory.blocks:
        planned_layers = []
        for layer in block.quantizable_layers:
            cost = current_cost(layer.layer, ranks[layer.layer])
            all_costs.append(cost)
            planned_layers.append(
                LayerPlan(
                    1,
                    layer.layer,
                    layer.weight,
                    ranks[layer.layer],
                    multiple,
                    caps[layer.layer],
                    objective_map[layer.layer],
                    outlier_plans[layer.layer],
                    RetryPolicy(
                        request.allocation.retry.maximum_attempts if request.allocation.retry.enabled else 1,
                        request.allocation.retry.rank_increase_fraction,
                        request.allocation.retry.thresholds.weighted_normalized_error,
                        request.allocation.retry.thresholds.raw_normalized_error,
                        (
                            min(layer.in_features, layer.out_features)
                            if request.allocation.retry.allow_above_allocator_cap
                            else caps[layer.layer]
                        ),
                        extra_budget,
                    ),
                    cost,
                )
            )
        block_plans.append(
            BlockPlan(
                block.block,
                tuple(layer.layer for layer in block.quantizable_layers),
                tuple(planned_layers),
                max((layer.in_features * layer.out_features * 4 for layer in block.quantizable_layers), default=0),
            )
        )
    planned_cost = _add_costs(all_costs)
    if spent > target_bits:
        raise AssertionError("planner exceeded exact target bit budget")
    return QuantizationPlan(
        1,
        ComponentRef("quantization-planner", "1"),
        request.inventory.model,
        request.calibration_ref,
        tuple(block_plans),
        request.allocation.target_bpw,
        planned_cost,
    )


def persist_plan(plan: QuantizationPlan, artifacts: ArtifactStore) -> PersistedPlan:
    layer_ids = [layer.layer for block in plan.blocks for layer in block.layers]
    if len(layer_ids) != len(set(layer_ids)) or any(layer.rank <= 0 for block in plan.blocks for layer in block.layers):
        raise ValueError("quantization plan is structurally invalid")
    with artifacts.begin_write("quantization-plan") as writer:
        (writer.path / "plan.json").write_text(json.dumps(to_dict(plan), sort_keys=True, indent=2), encoding="utf-8")
        descriptor = writer.commit()
    return PersistedPlan(ArtifactRef("quantization-plan", descriptor.artifact_id, 1), plan)
