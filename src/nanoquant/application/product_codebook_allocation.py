"""Exact-bit global allocation for measured product-codebook layer options."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from nanoquant.domain.models import (
    ArtifactRef,
    BitCost,
    LayerId,
    LayerPlan,
    ProductCodebookAllocationDecision,
    QuantizationPlan,
)


@dataclass(frozen=True, slots=True)
class ProductCodebookAllocationOption:
    """One independently measured rank/encoding option for a layer."""

    layer: LayerId
    rank: int
    free_rows: int | None
    estimated_cost: BitCost
    measured_weighted_error: float
    objective_multiplier: float
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("product-codebook option rank must be positive")
        if self.free_rows is not None and not 0 <= self.free_rows < self.rank:
            raise ValueError("product-codebook option must leave coded rows")
        if not math.isfinite(self.measured_weighted_error) or self.measured_weighted_error < 0:
            raise ValueError("product-codebook option error must be finite and nonnegative")
        if not math.isfinite(self.objective_multiplier) or self.objective_multiplier <= 0:
            raise ValueError("product-codebook option objective multiplier must be positive")

    @property
    def objective_value(self) -> float:
        return self.measured_weighted_error * self.objective_multiplier


@dataclass(frozen=True, slots=True)
class ProductCodebookAllocationGroup:
    layer: LayerId
    baseline_rank: int
    options: tuple[ProductCodebookAllocationOption, ...]

    def __post_init__(self) -> None:
        if self.baseline_rank <= 0 or not self.options:
            raise ValueError("product-codebook allocation group must contain options")
        if any(option.layer != self.layer for option in self.options):
            raise ValueError("product-codebook allocation options must match their group")
        if len({(option.rank, option.free_rows) for option in self.options}) != len(self.options):
            raise ValueError("product-codebook allocation options must be unique")


@dataclass(frozen=True, slots=True)
class ProductCodebookAllocationResult:
    plan: QuantizationPlan
    selected_bits: int
    budget_bits: int
    decisions: tuple[ProductCodebookAllocationDecision, ...]


@dataclass(frozen=True, slots=True)
class _FrontierState:
    objective: float
    choices: tuple[int, ...]
    has_product_encoding: bool


def _total_weight_elements(plan: QuantizationPlan) -> int:
    total = 0
    for block in plan.blocks:
        total += sum(math.prod(layer.source_weight.spec.shape) for layer in block.layers)
        total += sum(
            math.prod(member.weight.spec.shape)
            for group in block.shared_input_groups
            for member in group.members
        )
    return total


def _uncharged_outlier_bits(plan: QuantizationPlan) -> int:
    total = 0
    for block in plan.blocks:
        for layer in block.layers:
            if not layer.outliers.charge_to_budget:
                total += layer.estimated_cost.outlier_value_bits
                total += layer.estimated_cost.outlier_index_bits
        for group in block.shared_input_groups:
            if not group.outliers.charge_to_budget:
                total += group.estimated_cost.outlier_value_bits
                total += group.estimated_cost.outlier_index_bits
    return total


def _pareto_allocate(
    groups: tuple[ProductCodebookAllocationGroup, ...],
    variable_budget_bits: int,
    *,
    require_product_encoding: bool,
) -> tuple[int, _FrontierState]:
    frontier: dict[tuple[int, bool], _FrontierState] = {
        (0, False): _FrontierState(0.0, (), False)
    }
    for group in groups:
        expanded: dict[tuple[int, bool], _FrontierState] = {}
        for (current_bits, _current_product), state in frontier.items():
            for option_index, option in enumerate(group.options):
                bits = current_bits + option.estimated_cost.total
                if bits > variable_budget_bits:
                    continue
                has_product = state.has_product_encoding or option.free_rows is not None
                objective = state.objective + option.objective_value
                key = (bits, has_product)
                existing = expanded.get(key)
                if existing is None or objective < existing.objective:
                    expanded[key] = _FrontierState(
                        objective,
                        state.choices + (option_index,),
                        has_product,
                    )
        if not expanded:
            raise ValueError(
                f"product-codebook bit budget cannot represent layer {group.layer}"
            )
        frontier = {}
        for product_state in (False, True):
            best_objective = math.inf
            for (bits, has_product), state in sorted(expanded.items()):
                if has_product == product_state and state.objective < best_objective:
                    frontier[(bits, has_product)] = state
                    best_objective = state.objective
    allowed = (
        ((bits, state) for (bits, has_product), state in frontier.items() if has_product)
        if require_product_encoding
        else ((bits, state) for (bits, _has_product), state in frontier.items())
    )
    try:
        return min(allowed, key=lambda item: (item[1].objective, -item[0]))
    except ValueError as error:
        raise ValueError("bit budget cannot fund a product-coded allocation") from error


def allocate_product_codebook_options(
    plan: QuantizationPlan,
    groups: tuple[ProductCodebookAllocationGroup, ...],
    profile: ArtifactRef,
    *,
    require_product_encoding: bool = False,
) -> ProductCodebookAllocationResult:
    """Select measured options and return a plan whose exact costs match them.

    Group options replace the complete existing ``LayerPlan.estimated_cost``.
    Shared-input owners and unlisted ordinary layers are fixed costs.  This
    makes outlier and sidecar accounting identical to the source plan while
    allowing rank and right-factor encoding to vary jointly.
    """

    layers = {layer.layer: layer for block in plan.blocks for layer in block.layers}
    if len(layers) != sum(len(block.layers) for block in plan.blocks):
        raise ValueError("quantization plan contains duplicate ordinary layers")
    if not groups or len({group.layer for group in groups}) != len(groups):
        raise ValueError("product-codebook allocation groups must be non-empty and unique")
    if any(group.layer not in layers for group in groups):
        raise ValueError("product-codebook allocation refers to an absent or grouped layer")
    for group in groups:
        layer = layers[group.layer]
        if group.baseline_rank != layer.rank:
            raise ValueError("product-codebook allocation baseline rank differs from plan")
        if any(option.rank > layer.allocator_cap for option in group.options):
            raise ValueError("product-codebook option exceeds the physical allocator cap")

    target_bits = (
        math.floor(_total_weight_elements(plan) * plan.target_bpw)
        + _uncharged_outlier_bits(plan)
    )
    replaced_baseline_bits = sum(layers[group.layer].estimated_cost.total for group in groups)
    fixed_bits = plan.planned_cost.total - replaced_baseline_bits
    if fixed_bits < 0 or fixed_bits > target_bits:
        raise ValueError("fixed quantization-plan cost exceeds the product-codebook budget")
    option_bits, selected = _pareto_allocate(
        groups,
        target_bits - fixed_bits,
        require_product_encoding=require_product_encoding,
    )
    chosen = {
        group.layer: group.options[choice]
        for group, choice in zip(groups, selected.choices, strict=True)
    }
    decisions = tuple(
        ProductCodebookAllocationDecision(
            group.layer,
            group.baseline_rank,
            chosen[group.layer].rank,
            chosen[group.layer].free_rows,
            chosen[group.layer].estimated_cost,
            chosen[group.layer].measured_weighted_error,
            chosen[group.layer].objective_value,
            chosen[group.layer].artifact,
        )
        for group in groups
    )
    updated_blocks = []
    for block in plan.blocks:
        updated_layers: list[LayerPlan] = []
        for layer in block.layers:
            option = chosen.get(layer.layer)
            if option is None:
                updated_layers.append(layer)
            else:
                updated_layers.append(
                    replace(
                        layer,
                        rank=option.rank,
                        estimated_cost=option.estimated_cost,
                        product_codebook_free_rows=option.free_rows,
                    )
                )
        updated_blocks.append(replace(block, layers=tuple(updated_layers)))

    selected_bits = fixed_bits + option_bits
    updated_cost = BitCost()
    for block in updated_blocks:
        for layer in block.layers:
            updated_cost += layer.estimated_cost
        for shared_group in block.shared_input_groups:
            updated_cost += shared_group.estimated_cost
    if updated_cost.total != selected_bits or selected_bits > target_bits:
        raise AssertionError("product-codebook allocation bit accounting is inconsistent")
    updated_plan = replace(
        plan,
        schema_version=max(plan.schema_version, 3),
        blocks=tuple(updated_blocks),
        planned_cost=updated_cost,
        product_codebook_allocation_profile=profile,
        product_codebook_decisions=decisions,
    )
    return ProductCodebookAllocationResult(
        updated_plan,
        selected_bits,
        target_bits,
        decisions,
    )


def validate_product_codebook_plan(
    plan: QuantizationPlan,
    *,
    eligible_layers: tuple[LayerId, ...],
) -> None:
    """Reject enabled product execution unless allocation is authoritative."""

    if plan.product_codebook_allocation_profile is None:
        raise ValueError("product-codebook execution requires a measured allocation profile")
    decisions = {decision.layer: decision for decision in plan.product_codebook_decisions}
    if set(decisions) != set(eligible_layers):
        raise ValueError("product-codebook allocation decisions do not cover eligible layers")
    layers = {layer.layer: layer for block in plan.blocks for layer in block.layers}
    for layer_id in eligible_layers:
        layer = layers[layer_id]
        decision = decisions[layer_id]
        if (
            layer.rank != decision.selected_rank
            or layer.product_codebook_free_rows != decision.selected_free_rows
            or layer.estimated_cost != decision.selected_cost
        ):
            raise ValueError("product-codebook layer plan differs from its allocation decision")


__all__ = [
    "ProductCodebookAllocationGroup",
    "ProductCodebookAllocationOption",
    "ProductCodebookAllocationResult",
    "allocate_product_codebook_options",
    "validate_product_codebook_plan",
]
