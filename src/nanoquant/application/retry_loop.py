"""Pure-policy-driven factorization retry loop with commit-coupled budgeting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from nanoquant.application.quantization_stages import FactorizationAttemptStage
from nanoquant.application.stages import StageContext, execute_stage
from nanoquant.domain.models import (
    AttemptSummary,
    FactorizationRequest,
    FactorizationResult,
    LayerPlan,
    TensorRef,
)
from nanoquant.domain.planning import decide_retry, factor_bit_cost, retry_score
from nanoquant.domain.runs import BudgetState
from nanoquant.domain.seeds import logical_seed


@dataclass(frozen=True, slots=True)
class AcceptedFactorization:
    result: FactorizationResult
    attempts: tuple[AttemptSummary, ...]
    budget: BudgetState


AcceptCommit = Callable[[FactorizationResult, tuple[AttemptSummary, ...]], None]


def run_factorization_attempts(
    layer_plan: LayerPlan,
    source_weight: TensorRef,
    residual_weight: TensorRef,
    run_seed: int,
    factorizer_config_hash: str,
    budget: BudgetState,
    context: StageContext,
    accept_commit: AcceptCommit,
) -> AcceptedFactorization:
    rank = layer_plan.rank
    base_factor_cost = factor_bit_cost(
        layer_plan.source_weight.spec.shape[0], layer_plan.source_weight.spec.shape[1], layer_plan.rank
    )
    results: list[FactorizationResult] = []
    summaries: list[AttemptSummary] = []
    while True:
        attempt = len(results)
        request = FactorizationRequest(
            1,
            layer_plan.layer,
            source_weight,
            residual_weight,
            layer_plan.objective,
            rank,
            logical_seed(run_seed, "factorize-attempt", layer_plan.layer.block.index, layer_plan.layer.path, attempt),
            factorizer_config_hash,
        )
        result = execute_stage(FactorizationAttemptStage(), request, context)
        results.append(result)
        weighted = result.metrics.export_weighted_normalized_error
        raw = result.metrics.raw_normalized_error
        score = retry_score(
            weighted, raw, layer_plan.retry.weighted_error_threshold, layer_plan.retry.raw_error_threshold
        )
        cost = factor_bit_cost(layer_plan.source_weight.spec.shape[0], layer_plan.source_weight.spec.shape[1], rank)
        summaries.append(
            AttemptSummary(
                attempt,
                rank,
                result.factors.left_binary.artifact,
                weighted,
                raw,
                cost,
                score,
                False,
                "pending retry decision",
            )
        )
        available = max(0, layer_plan.retry.extra_bit_budget - budget.retry_bits_spent)
        decision = decide_retry(
            tuple(summaries),
            maximum_attempts=layer_plan.retry.maximum_attempts,
            rank_increase_fraction=layer_plan.retry.rank_increase_fraction,
            rank_multiple=layer_plan.rank_multiple,
            hard_rank_cap=layer_plan.retry.hard_rank_cap,
            available_extra_bits=available,
            cost_per_rank=sum(layer_plan.source_weight.spec.shape) + 16,
            weighted_threshold=layer_plan.retry.weighted_error_threshold,
            raw_threshold=layer_plan.retry.raw_error_threshold,
        )
        context.events.emit(
            "factorize-attempt",
            "info",
            "factorization.retry_decision",
            layer=str(layer_plan.layer),
            attempt=attempt,
            action=decision.action,
            next_rank=decision.next_rank,
            reason=decision.reason,
        )
        if decision.action == "retry":
            if decision.next_rank is None:
                raise AssertionError("retry decision omitted next rank")
            rank = decision.next_rank
            continue
        if decision.accepted_attempt is None:
            raise AssertionError("accept decision omitted accepted attempt")
        accepted_index = decision.accepted_attempt
        summaries[accepted_index] = replace(summaries[accepted_index], accepted=True, decision_reason=decision.reason)
        accepted_result = results[accepted_index]
        accepted_summaries = tuple(summaries)
        accept_commit(accepted_result, accepted_summaries)
        extra_bits = max(0, summaries[accepted_index].bit_cost.total - base_factor_cost.total)
        updated_budget = replace(
            budget,
            accepted_bits=budget.accepted_bits + summaries[accepted_index].bit_cost.total,
            retry_bits_spent=budget.retry_bits_spent + extra_bits,
        )
        return AcceptedFactorization(accepted_result, accepted_summaries, updated_budget)
