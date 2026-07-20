"""Complete immutable quantization planning before model mutation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fnmatch import fnmatchcase

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import BiasCorrectionConfig, LowRankPatchConfig, OutlierConfig, RankAllocationConfig
from nanoquant.domain.models import (
    ArtifactRef,
    ArtifactTypes,
    BitCost,
    BlockPlan,
    CalibrationStats,
    ComponentRef,
    LayerInventory,
    LayerPlan,
    ModelInventory,
    ObjectiveSpec,
    OutlierPlan,
    QuantizationPlan,
    ReconstructionRankDecision,
    RetryPolicy,
    SharedInputGroupCandidate,
    SharedInputGroupPlan,
)
from nanoquant.domain.planning import bias_bit_cost, factor_bit_cost, outlier_bit_cost, patch_bit_cost
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
    shared_input_groups: tuple[SharedInputGroupCandidate, ...] = ()
    reconstruction_profile: ArtifactRef | None = None
    reconstruction_decisions: tuple[ReconstructionRankDecision, ...] = ()
    bias_correction: BiasCorrectionConfig = BiasCorrectionConfig()
    low_rank_patch: LowRankPatchConfig = LowRankPatchConfig()


@dataclass(frozen=True, slots=True)
class PersistedPlan:
    reference: ArtifactRef
    plan: QuantizationPlan


def _add_costs(costs: list[BitCost]) -> BitCost:
    result = BitCost()
    for cost in costs:
        result = result + cost
    return result


def _storage_bits(dtype: str) -> int:
    return {"bfloat16": 16, "float16": 16, "int8": 8}[dtype]


def _side_cost(request: PlanningRequest, name: str, out_features: int, in_features: int) -> BitCost:
    cost = BitCost()
    if request.bias_correction.enabled:
        cost += bias_bit_cost(
            out_features,
            value_bits=_storage_bits(request.bias_correction.storage_dtype.value),
        )
    if request.low_rank_patch.enabled and any(
        fnmatchcase(name, pattern) for pattern in request.low_rank_patch.layer_patterns
    ):
        cost += patch_bit_cost(
            out_features,
            in_features,
            request.low_rank_patch.rank,
            value_bits=_storage_bits(request.low_rank_patch.storage_dtype.value),
        )
    return cost


def _charged_side_bits(request: PlanningRequest, cost: BitCost) -> int:
    return (
        (cost.bias_bits if request.bias_correction.charge_to_bit_budget else 0)
        + (cost.patch_bits if request.low_rank_patch.charge_to_bit_budget else 0)
    )


def build_quantization_plan(request: PlanningRequest) -> QuantizationPlan:
    if request.shared_input_groups or request.allocation.strategy.value in {"reconstruction_aware", "kl_calibrated"}:
        return _build_grouped_quantization_plan(request)
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
    side_costs: dict[object, BitCost] = {}
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
        side_costs[layer.layer] = _side_cost(
            request,
            layer.layer.path,
            layer.out_features,
            layer.in_features,
        )
    base_ranks: dict[object, int] = {}
    for layer in layers:
        layer_target_bits = math.floor(layer.in_features * layer.out_features * request.allocation.target_bpw)
        charged = (
            outlier_costs[layer.layer].total if outlier_plans[layer.layer].charge_to_budget else 0
        ) + _charged_side_bits(request, side_costs[layer.layer])
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
        if (
            request.allocation.strategy.value in {"utility_profile", "kl_calibrated"}
            and profile_key not in utility_profile
        ):
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
        return (
            factor_bit_cost(inventory.out_features, inventory.in_features, rank)
            + outlier_costs[layer]
            + side_costs[layer]
        )

    def budget_cost(layer: object, rank: int) -> int:
        inventory = next(item for item in layers if item.layer == layer)
        factor = factor_bit_cost(inventory.out_features, inventory.in_features, rank).total
        outliers = outlier_costs[layer].total if outlier_plans[layer].charge_to_budget else 0
        return factor + outliers + _charged_side_bits(request, side_costs[layer])

    spent = sum(budget_cost(layer.layer, ranks[layer.layer]) for layer in layers)
    if spent > target_bits:
        raise ValueError(f"minimum rank/outlier plan costs {spent} bits, exceeding target {target_bits}")
    if request.allocation.strategy.value == "sensitivity" and utility_profile:
        costs_per_rank = {layer.layer: layer.in_features + layer.out_features + 16 for layer in layers}
        budget_units = sum(base_ranks[layer.layer] * costs_per_rank[layer.layer] for layer in layers)
        score_units = sum(
            base_ranks[layer.layer] * costs_per_rank[layer.layer] * utilities[layer.layer] for layer in layers
        )
        mean_score = score_units / budget_units
        targets = {layer.layer: base_ranks[layer.layer] * utilities[layer.layer] / mean_score for layer in layers}
        for layer in layers:
            proposed = math.floor(targets[layer.layer] / multiple) * multiple
            ranks[layer.layer] = min(caps[layer.layer], max(ranks[layer.layer], proposed))

        def used_units() -> int:
            return sum(ranks[layer.layer] * costs_per_rank[layer.layer] for layer in layers)

        while used_units() > budget_units:
            candidates = [
                layer
                for layer in layers
                if ranks[layer.layer] - multiple
                >= max(
                    multiple,
                    math.floor(base_ranks[layer.layer] * request.allocation.bounds.floor_fraction_of_uniform / multiple)
                    * multiple,
                )
            ]
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
    if spent > target_bits:
        raise AssertionError("planner exceeded exact target bit budget before additive promotions")

    budget_multipliers = request.allocation.layer_budget_multipliers
    matched_budget_patterns: set[str] = set()
    for layer in layers:
        budget_matches = tuple(item for item in budget_multipliers if fnmatchcase(layer.layer.path, item.pattern))
        if len(budget_matches) > 1:
            raise ValueError(f"multiple layer-budget patterns matched {layer.layer.path}")
        if not budget_matches:
            continue
        rule = budget_matches[0]
        matched_budget_patterns.add(rule.pattern)
        maximum = min(layer.in_features, layer.out_features)
        original_factor_bits = factor_bit_cost(layer.out_features, layer.in_features, ranks[layer.layer]).total
        promoted_budget = math.floor(original_factor_bits * rule.multiplier)
        promoted_rank = ranks[layer.layer]
        while promoted_rank + multiple <= maximum:
            candidate = promoted_rank + multiple
            if factor_bit_cost(layer.out_features, layer.in_features, candidate).total > promoted_budget:
                break
            promoted_rank = candidate
        ranks[layer.layer] = promoted_rank
        caps[layer.layer] = max(caps[layer.layer], promoted_rank)
    unmatched_budget_patterns = {item.pattern for item in budget_multipliers} - matched_budget_patterns
    if unmatched_budget_patterns:
        raise ValueError(f"layer-budget patterns matched no quantizable layer: {sorted(unmatched_budget_patterns)}")

    maximum_rank_patterns = request.allocation.maximum_rank_layer_patterns
    matched_patterns: set[str] = set()
    if maximum_rank_patterns:
        for layer in layers:
            maximum_matches = tuple(
                pattern for pattern in maximum_rank_patterns if fnmatchcase(layer.layer.path, pattern)
            )
            if not maximum_matches:
                continue
            matched_patterns.update(maximum_matches)
            maximum = min(layer.in_features, layer.out_features)
            if maximum % multiple:
                raise ValueError(
                    f"maximum rank {maximum} for {layer.layer.path} is not aligned to "
                    f"allocation rank multiple {multiple}"
                )
            ranks[layer.layer] = maximum
            caps[layer.layer] = maximum
        unmatched = set(maximum_rank_patterns) - matched_patterns
        if unmatched:
            raise ValueError(f"maximum-rank layer patterns matched no quantizable layer: {sorted(unmatched)}")
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
    return QuantizationPlan(
        1,
        ComponentRef("quantization-planner", "1"),
        request.inventory.model,
        request.calibration_ref,
        tuple(block_plans),
        request.allocation.target_bpw,
        planned_cost,
    )


@dataclass(frozen=True, slots=True)
class _PlanningUnit:
    block_index: int
    name: str
    members: tuple[LayerInventory, ...]
    objective_multipliers: tuple[float, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.block_index}:{self.name}"

    @property
    def in_features(self) -> int:
        return self.members[0].in_features

    @property
    def out_features(self) -> int:
        return sum(member.out_features for member in self.members)

    @property
    def maximum_rank(self) -> int:
        return min(self.in_features, self.out_features)

    @property
    def grouped(self) -> bool:
        return len(self.members) > 1


def _build_grouped_quantization_plan(request: PlanningRequest) -> QuantizationPlan:
    """Plan selected shared-input groups as physical factor owners.

    The ungrouped planner above is intentionally left unchanged. This path is
    selected only by an explicit, adapter-validated topology.
    """

    layers = [layer for block in request.inventory.blocks for layer in block.quantizable_layers]
    if not layers:
        raise ValueError("model inventory has no quantizable layers")
    inventory_by_id = {layer.layer: layer for layer in layers}
    objective_map = {objective.layer: objective for objective in request.objectives}
    calibration_map = {stats.layer: stats for stats in request.calibration.layers}
    if set(objective_map) != set(inventory_by_id):
        raise ValueError("objective set does not exactly match model inventory")

    grouped_members: set[object] = set()
    group_by_block: dict[int, list[_PlanningUnit]] = {}
    for candidate in request.shared_input_groups:
        if candidate.block.index >= len(request.inventory.blocks):
            raise ValueError(f"shared-input group block is absent: {candidate.block.index}")
        try:
            members = tuple(inventory_by_id[layer] for layer in candidate.members)
        except KeyError as error:
            raise ValueError(f"shared-input group member is not quantizable: {error.args[0]}") from error
        overlap = grouped_members.intersection(candidate.members)
        if overlap:
            raise ValueError(f"shared-input groups overlap: {sorted(str(item) for item in overlap)}")
        if len({member.in_features for member in members}) != 1:
            raise ValueError(f"shared-input group members have different input widths: {candidate.name}")
        if any(member.bias is not None for member in members):
            raise ValueError(f"shared-input groups with bias are not yet supported: {candidate.name}")
        grouped_members.update(candidate.members)
        group_by_block.setdefault(candidate.block.index, []).append(
            _PlanningUnit(candidate.block.index, candidate.name, members, candidate.objective_multipliers)
        )

    units: list[_PlanningUnit] = []
    units_by_block: dict[int, list[_PlanningUnit]] = {}
    for block in request.inventory.blocks:
        selected = {
            member.layer: group for group in group_by_block.get(block.block.index, ()) for member in group.members
        }
        emitted: set[str] = set()
        block_units: list[_PlanningUnit] = []
        for layer in block.quantizable_layers:
            group = selected.get(layer.layer)
            if group is not None:
                if group.name not in emitted:
                    block_units.append(group)
                    emitted.add(group.name)
            else:
                block_units.append(_PlanningUnit(block.block.index, layer.layer.path, (layer,)))
        units.extend(block_units)
        units_by_block[block.block.index] = block_units

    multiple = request.allocation.bounds.multiple
    target_bits = math.floor(
        sum(layer.in_features * layer.out_features for layer in layers) * request.allocation.target_bpw
    )
    utility_profile = dict(request.utility_profile)
    outlier_plans: dict[str, OutlierPlan] = {}
    outlier_costs: dict[str, BitCost] = {}
    side_costs: dict[str, BitCost] = {}
    base_ranks: dict[str, int] = {}
    for unit in units:
        count = 0
        if request.outliers.selector.value != "none" and request.outliers.fraction > 0:
            raw = math.ceil(unit.in_features * request.outliers.fraction)
            count = math.ceil(raw / request.outliers.count_multiple) * request.outliers.count_multiple
            count = min(count, unit.in_features)
        outlier_plan = OutlierPlan(
            request.outliers.selector.value,
            count,
            request.outliers.storage_dtype.value,
            request.outliers.charge_to_bit_budget,
            request.outliers.removed_column_importance,
        )
        value_bits = {"bfloat16": 16, "float16": 16, "int8": 8}[request.outliers.storage_dtype.value]
        outlier_cost = outlier_bit_cost(unit.out_features, count, value_bits=value_bits) if count else BitCost()
        outlier_plans[unit.key] = outlier_plan
        outlier_costs[unit.key] = outlier_cost
        side_costs[unit.key] = _side_cost(request, unit.name, unit.out_features, unit.in_features)
        funded_bits = sum(
            math.floor(member.in_features * member.out_features * request.allocation.target_bpw)
            for member in unit.members
        )
        charged = (outlier_cost.total if outlier_plan.charge_to_budget else 0) + _charged_side_bits(
            request, side_costs[unit.key]
        )
        base_rank = 0
        for candidate_rank in range(multiple, unit.maximum_rank + 1, multiple):
            if factor_bit_cost(unit.out_features, unit.in_features, candidate_rank).total + charged > funded_bits:
                break
            base_rank = candidate_rank
        if base_rank == 0:
            raise ValueError(f"target bit budget cannot fund minimum aligned factors for {unit.key}")
        base_ranks[unit.key] = base_rank

    ranks: dict[str, int] = {}
    caps: dict[str, int] = {}
    utilities: dict[str, float] = {}
    for unit in units:
        base_rank = base_ranks[unit.key]
        if request.allocation.strategy.value == "uniform":
            floor = cap = base_rank
        else:
            floor = max(
                multiple,
                math.floor(base_rank * request.allocation.bounds.floor_fraction_of_uniform / multiple) * multiple,
            )
            cap = min(
                unit.maximum_rank,
                max(
                    multiple,
                    math.floor(base_rank * request.allocation.bounds.ceiling_fraction_of_uniform / multiple) * multiple,
                ),
            )
        ranks[unit.key] = min(floor, cap)
        caps[unit.key] = cap
        weighted_sensitivity = 0.0
        weight = 0
        for member in unit.members:
            stats = calibration_map.get(member.layer)
            sensitivity = 1.0 if stats is None else max(1e-12, stats.input_summary.mean * stats.output_summary.mean)
            elements = member.in_features * member.out_features
            weighted_sensitivity += sensitivity * elements
            weight += elements
        profile_key = unit.key
        if (
            request.allocation.strategy.value in {"utility_profile", "kl_calibrated"}
            and profile_key not in utility_profile
        ):
            raise ValueError(f"utility profile is missing unit {profile_key}")
        member_profile = [
            (utility_profile.get(f"{unit.block_index}:{member.layer.path}"), member.in_features * member.out_features)
            for member in unit.members
        ]
        profiled_members = [(value, elements) for value, elements in member_profile if value is not None]
        utility = utility_profile.get(
            profile_key,
            (
                sum(float(value) * elements for value, elements in profiled_members)
                / sum(elements for _value, elements in profiled_members)
                if len(profiled_members) == len(unit.members)
                else weighted_sensitivity / weight
            ),
        )
        if request.allocation.strategy.value != "uniform" and request.allocation.bounds.edge_block_boost > 0:
            last_block = len(request.inventory.blocks) - 1
            edge_distance = min(unit.block_index, last_block - unit.block_index)
            edge_score = 1.0 - edge_distance / max(1.0, last_block / 2.0)
            utility *= 1.0 + request.allocation.bounds.edge_block_boost * max(0.0, edge_score)
        utilities[unit.key] = utility

    def budget_cost(unit: _PlanningUnit, rank: int) -> int:
        factor = factor_bit_cost(unit.out_features, unit.in_features, rank).total
        outliers = outlier_costs[unit.key].total if outlier_plans[unit.key].charge_to_budget else 0
        return factor + outliers + _charged_side_bits(request, side_costs[unit.key])

    spent = sum(budget_cost(unit, ranks[unit.key]) for unit in units)
    if spent > target_bits:
        raise ValueError(f"minimum rank/outlier plan costs {spent} bits, exceeding target {target_bits}")
    if request.allocation.strategy.value in {"reconstruction_aware", "kl_calibrated"}:
        if request.reconstruction_profile is None:
            raise ValueError("reconstruction-aware planning requires a persisted profile")
        decision_map = {decision.unit_id: decision for decision in request.reconstruction_decisions}
        if set(decision_map) != {unit.key for unit in units}:
            raise ValueError("reconstruction decisions do not exactly cover quantization units")
        for unit in units:
            decision = decision_map[unit.key]
            if decision.baseline_rank != base_ranks[unit.key]:
                raise ValueError(f"reconstruction baseline rank differs for {unit.key}")
            if decision.planned_rank % multiple or not multiple <= decision.planned_rank <= caps[unit.key]:
                raise ValueError(f"reconstruction planned rank is outside aligned bounds for {unit.key}")
            if decision.members != tuple(member.layer for member in unit.members):
                raise ValueError(f"reconstruction decision members differ for {unit.key}")
            ranks[unit.key] = decision.planned_rank
        spent = sum(budget_cost(unit, ranks[unit.key]) for unit in units)
        if spent > target_bits:
            raise ValueError("reconstruction-aware plan exceeds exact target bit budget")
    else:
        while True:
            candidates: list[tuple[float, int, str, _PlanningUnit, int]] = []
            for unit in units:
                rank = ranks[unit.key]
                if rank + multiple > caps[unit.key]:
                    continue
                marginal = budget_cost(unit, rank + multiple) - budget_cost(unit, rank)
                if spent + marginal <= target_bits:
                    candidates.append((utilities[unit.key] / marginal, -unit.block_index, unit.name, unit, marginal))
            if not candidates:
                break
            *_, selected_unit, marginal = max(candidates, key=lambda item: item[:3])
            ranks[selected_unit.key] += multiple
            spent += marginal

    matched_budget_patterns: set[str] = set()
    for unit in units:
        budget_matches = tuple(
            rule for rule in request.allocation.layer_budget_multipliers if fnmatchcase(unit.name, rule.pattern)
        )
        if len(budget_matches) > 1:
            raise ValueError(f"multiple layer-budget patterns matched {unit.name}")
        if budget_matches:
            rule = budget_matches[0]
            matched_budget_patterns.add(rule.pattern)
            promoted_budget = math.floor(
                factor_bit_cost(unit.out_features, unit.in_features, ranks[unit.key]).total * rule.multiplier
            )
            while ranks[unit.key] + multiple <= unit.maximum_rank:
                candidate_rank = ranks[unit.key] + multiple
                if factor_bit_cost(unit.out_features, unit.in_features, candidate_rank).total > promoted_budget:
                    break
                ranks[unit.key] = candidate_rank
            caps[unit.key] = max(caps[unit.key], ranks[unit.key])
    unmatched_budget = {rule.pattern for rule in request.allocation.layer_budget_multipliers} - matched_budget_patterns
    if unmatched_budget:
        raise ValueError(f"layer-budget patterns matched no quantization unit: {sorted(unmatched_budget)}")

    matched_maximum: set[str] = set()
    for unit in units:
        maximum_matches = tuple(
            pattern for pattern in request.allocation.maximum_rank_layer_patterns if fnmatchcase(unit.name, pattern)
        )
        if maximum_matches:
            matched_maximum.update(maximum_matches)
            if unit.maximum_rank % multiple:
                raise ValueError(
                    f"maximum rank {unit.maximum_rank} for {unit.name} is not aligned to "
                    f"allocation rank multiple {multiple}"
                )
            ranks[unit.key] = caps[unit.key] = unit.maximum_rank
    unmatched_maximum = set(request.allocation.maximum_rank_layer_patterns) - matched_maximum
    if unmatched_maximum:
        raise ValueError(f"maximum-rank patterns matched no quantization unit: {sorted(unmatched_maximum)}")

    extra_budget = math.floor(target_bits * request.allocation.retry.extra_bit_budget_fraction)
    blocks: list[BlockPlan] = []
    all_costs: list[BitCost] = []
    for inventory_block in request.inventory.blocks:
        ordinary_plans: list[LayerPlan] = []
        group_plans: list[SharedInputGroupPlan] = []
        for unit in units_by_block[inventory_block.block.index]:
            cost = (
                factor_bit_cost(unit.out_features, unit.in_features, ranks[unit.key])
                + outlier_costs[unit.key]
                + side_costs[unit.key]
            )
            all_costs.append(cost)
            retry = RetryPolicy(
                request.allocation.retry.maximum_attempts if request.allocation.retry.enabled else 1,
                request.allocation.retry.rank_increase_fraction,
                request.allocation.retry.thresholds.weighted_normalized_error,
                request.allocation.retry.thresholds.raw_normalized_error,
                unit.maximum_rank if request.allocation.retry.allow_above_allocator_cap else caps[unit.key],
                extra_budget,
            )
            if unit.grouped:
                group_plans.append(
                    SharedInputGroupPlan(
                        1,
                        inventory_block.block,
                        unit.name,
                        unit.members,
                        ranks[unit.key],
                        multiple,
                        caps[unit.key],
                        tuple(objective_map[member.layer] for member in unit.members),
                        outlier_plans[unit.key],
                        retry,
                        cost,
                        unit.objective_multipliers,
                    )
                )
            else:
                member = unit.members[0]
                ordinary_plans.append(
                    LayerPlan(
                        1,
                        member.layer,
                        member.weight,
                        ranks[unit.key],
                        multiple,
                        caps[unit.key],
                        objective_map[member.layer],
                        outlier_plans[unit.key],
                        retry,
                        cost,
                    )
                )
        block_units = units_by_block[inventory_block.block.index]
        blocks.append(
            BlockPlan(
                inventory_block.block,
                tuple(layer.layer for layer in inventory_block.quantizable_layers),
                tuple(ordinary_plans),
                max((unit.in_features * unit.out_features * 4 for unit in block_units), default=0),
                tuple(group_plans),
                tuple(unit.name for unit in block_units),
            )
        )
    return QuantizationPlan(
        2,
        ComponentRef("quantization-planner", "2"),
        request.inventory.model,
        request.calibration_ref,
        tuple(blocks),
        request.allocation.target_bpw,
        _add_costs(all_costs),
        request.reconstruction_profile,
        request.reconstruction_decisions,
    )


def persist_plan(plan: QuantizationPlan, artifacts: ArtifactStore) -> PersistedPlan:
    layer_ids = [layer.layer for block in plan.blocks for layer in block.layers]
    group_member_ids = [
        member.layer for block in plan.blocks for group in block.shared_input_groups for member in group.members
    ]
    all_ids = [*layer_ids, *group_member_ids]
    if (
        len(all_ids) != len(set(all_ids))
        or any(layer.rank <= 0 for block in plan.blocks for layer in block.layers)
        or any(group.rank <= 0 for block in plan.blocks for group in block.shared_input_groups)
    ):
        raise ValueError("quantization plan is structurally invalid")
    with artifacts.begin_write(ArtifactTypes.QUANTIZATION_PLAN) as writer:
        (writer.path / "plan.json").write_text(json.dumps(to_dict(plan), sort_keys=True, indent=2), encoding="utf-8")
        descriptor = writer.commit()
    return PersistedPlan(ArtifactRef(ArtifactTypes.QUANTIZATION_PLAN, descriptor.artifact_id, 1), plan)
