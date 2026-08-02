"""Deterministic scalar logit-temperature calibration statistics and solver."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from nanoquant.domain.linear_math import chunk_slices

TEMPERATURE_CALIBRATION_VERSION = 1


@dataclass(frozen=True, slots=True)
class TemperatureNllStatistics:
    token_count: int
    negative_log_likelihood_sum: float
    gradient_sum: float
    hessian_sum: float

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("temperature statistics require a positive token count")
        if not all(
            math.isfinite(value)
            for value in (
                self.negative_log_likelihood_sum,
                self.gradient_sum,
                self.hessian_sum,
            )
        ):
            raise ValueError("temperature statistics must be finite")
        if self.hessian_sum < 0:
            raise ValueError("temperature-statistics Hessian must not be negative")
        if self.negative_log_likelihood_sum < 0:
            raise ValueError("temperature-statistics NLL must not be negative")

    @property
    def mean_negative_log_likelihood(self) -> float:
        return self.negative_log_likelihood_sum / self.token_count


@dataclass(frozen=True, slots=True)
class TemperatureFitIteration:
    iteration: int
    logit_scale: float
    equivalent_temperature: float
    token_count: int
    mean_negative_log_likelihood: float
    gradient_sum: float
    hessian_sum: float
    proposed_logit_scale: float
    next_logit_scale: float

    def __post_init__(self) -> None:
        values = (
            self.logit_scale,
            self.equivalent_temperature,
            self.mean_negative_log_likelihood,
            self.gradient_sum,
            self.hessian_sum,
            self.proposed_logit_scale,
            self.next_logit_scale,
        )
        if (
            self.iteration <= 0
            or self.token_count <= 0
            or not all(math.isfinite(value) for value in values)
            or self.logit_scale <= 0
            or self.next_logit_scale <= 0
            or self.equivalent_temperature <= 0
            or self.mean_negative_log_likelihood < 0
            or self.hessian_sum < 0
        ):
            raise ValueError("temperature-fit iteration is invalid")
        if not math.isclose(
            self.equivalent_temperature,
            1 / self.logit_scale,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("temperature-fit iteration has an invalid equivalent temperature")


@dataclass(frozen=True, slots=True)
class TemperatureFitResult:
    version: int
    initial_logit_scale: float
    minimum_logit_scale: float
    maximum_logit_scale: float
    convergence_tolerance: float
    hessian_floor: float
    maximum_update_passes: int
    iterations: tuple[TemperatureFitIteration, ...]
    final_logit_scale: float
    equivalent_temperature: float
    final_mean_negative_log_likelihood: float
    token_count: int
    converged: bool
    boundary_reached: bool

    def __post_init__(self) -> None:
        values = (
            self.initial_logit_scale,
            self.minimum_logit_scale,
            self.maximum_logit_scale,
            self.convergence_tolerance,
            self.hessian_floor,
            self.final_logit_scale,
            self.equivalent_temperature,
            self.final_mean_negative_log_likelihood,
        )
        if (
            self.version != TEMPERATURE_CALIBRATION_VERSION
            or not all(math.isfinite(value) for value in values)
            or not 0
            < self.minimum_logit_scale
            < self.initial_logit_scale
            < self.maximum_logit_scale
            or self.maximum_update_passes <= 0
            or self.convergence_tolerance <= 0
            or self.hessian_floor <= 0
            or not self.iterations
            or len(self.iterations) > self.maximum_update_passes
            or self.token_count <= 0
            or self.final_mean_negative_log_likelihood < 0
            or not self.converged
            or self.boundary_reached
        ):
            raise ValueError("temperature-fit result is invalid")
        scale = self.initial_logit_scale
        for expected_iteration, iteration in enumerate(self.iterations, start=1):
            if iteration.hessian_sum <= 0:
                raise ValueError("temperature-fit result iteration Hessian is not positive")
            proposed = iteration.logit_scale - iteration.gradient_sum / iteration.hessian_sum
            expected_next = min(
                max(proposed, self.minimum_logit_scale),
                self.maximum_logit_scale,
            )
            if (
                iteration.iteration != expected_iteration
                or iteration.token_count != self.token_count
                or not math.isclose(iteration.logit_scale, scale, rel_tol=0, abs_tol=1e-12)
                or not math.isclose(
                    iteration.proposed_logit_scale,
                    proposed,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    iteration.next_logit_scale,
                    expected_next,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                or expected_next in {self.minimum_logit_scale, self.maximum_logit_scale}
            ):
                raise ValueError("temperature-fit result iteration sequence is invalid")
            scale = expected_next
        if (
            abs(self.iterations[-1].next_logit_scale - self.iterations[-1].logit_scale)
            > self.convergence_tolerance
            or not math.isclose(self.final_logit_scale, scale, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(
                self.equivalent_temperature,
                1 / self.final_logit_scale,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("temperature-fit result final state is invalid")


@dataclass(frozen=True, slots=True)
class TemperatureSequenceMetrics:
    raw_negative_log_likelihood: float
    raw_kl_nats_per_token: float
    fitted_negative_log_likelihood: float
    fitted_kl_nats_per_token: float
    teacher_top1_agreement: float
    teacher_topk_mass: float
    raw_student_teacher_topk_mass: float
    fitted_student_teacher_topk_mass: float
    token_count: int

    def __post_init__(self) -> None:
        values = (
            self.raw_negative_log_likelihood,
            self.raw_kl_nats_per_token,
            self.fitted_negative_log_likelihood,
            self.fitted_kl_nats_per_token,
            self.teacher_top1_agreement,
            self.teacher_topk_mass,
            self.raw_student_teacher_topk_mass,
            self.fitted_student_teacher_topk_mass,
        )
        if self.token_count <= 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("temperature sequence metrics must be finite with positive tokens")
        if self.raw_kl_nats_per_token < 0 or self.fitted_kl_nats_per_token < 0:
            raise ValueError("temperature sequence KL must not be negative")
        if not all(0 <= value <= 1 for value in values[4:]):
            raise ValueError("temperature sequence agreement and masses must be in [0, 1]")


def causal_raw_fitted_temperature_metrics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    logit_scale: float,
    top_k: int = 64,
    attention_mask: torch.Tensor | None = None,
    token_chunk_size: int = 128,
) -> tuple[TemperatureSequenceMetrics, ...]:
    """Reduce raw and scaled capability/calibration metrics in bounded chunks."""

    if (
        teacher_logits.shape != student_logits.shape
        or teacher_logits.ndim != 3
        or token_ids.shape != teacher_logits.shape[:2]
        or not math.isfinite(logit_scale)
        or logit_scale <= 0
        or not 0 < top_k < teacher_logits.shape[-1]
        or token_chunk_size <= 0
    ):
        raise ValueError("raw/fitted temperature metric protocol is invalid")
    if attention_mask is not None and attention_mask.shape != token_ids.shape:
        raise ValueError("raw/fitted temperature attention mask must match token IDs")
    results = []
    for sequence in range(teacher_logits.shape[0]):
        teacher = teacher_logits[sequence, :-1]
        student = student_logits[sequence, :-1]
        labels = token_ids[sequence, 1:]
        valid = torch.ones_like(labels, dtype=torch.bool)
        if attention_mask is not None:
            mask = attention_mask[sequence]
            valid = mask[1:].bool() & mask[:-1].bool()
        valid_rows = valid.nonzero(as_tuple=False).reshape(-1)
        if valid_rows.numel() == 0:
            raise ValueError("raw/fitted temperature metrics have no valid tokens")
        sums: dict[str, list[float]] = {
            name: []
            for name in (
                "raw_nll",
                "raw_kl",
                "fitted_nll",
                "fitted_kl",
                "agreement",
                "teacher_mass",
                "raw_mass",
                "fitted_mass",
            )
        }
        for row_slice in chunk_slices(int(valid_rows.numel()), token_chunk_size):
            rows = valid_rows[row_slice]
            teacher_values = teacher.index_select(0, rows).float()
            student_values = student.index_select(0, rows).float()
            teacher_logp = torch.log_softmax(teacher_values, dim=-1)
            raw_logp = torch.log_softmax(student_values, dim=-1)
            fitted_logp = torch.log_softmax(student_values * logit_scale, dim=-1)
            teacher_p = teacher_logp.exp()
            selected = labels.index_select(0, rows).reshape(-1, 1)
            indices = torch.topk(teacher_values, top_k, dim=-1).indices
            sums["raw_nll"].append(float(-raw_logp.gather(1, selected).sum()))
            sums["fitted_nll"].append(float(-fitted_logp.gather(1, selected).sum()))
            sums["raw_kl"].append(float((teacher_p * (teacher_logp - raw_logp)).sum()))
            sums["fitted_kl"].append(
                float((teacher_p * (teacher_logp - fitted_logp)).sum())
            )
            sums["agreement"].append(
                float(teacher_values.argmax(dim=-1).eq(student_values.argmax(dim=-1)).sum())
            )
            sums["teacher_mass"].append(float(teacher_p.gather(1, indices).sum()))
            sums["raw_mass"].append(float(raw_logp.gather(1, indices).exp().sum()))
            sums["fitted_mass"].append(float(fitted_logp.gather(1, indices).exp().sum()))
        count = int(valid_rows.numel())
        results.append(
            TemperatureSequenceMetrics(
                math.fsum(sums["raw_nll"]) / count,
                math.fsum(sums["raw_kl"]) / count,
                math.fsum(sums["fitted_nll"]) / count,
                math.fsum(sums["fitted_kl"]) / count,
                math.fsum(sums["agreement"]) / count,
                math.fsum(sums["teacher_mass"]) / count,
                math.fsum(sums["raw_mass"]) / count,
                math.fsum(sums["fitted_mass"]) / count,
                count,
            )
        )
    return tuple(results)


def temperature_nll_statistics(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    logit_scale: float,
    attention_mask: torch.Tensor | None = None,
    token_chunk_size: int = 128,
) -> TemperatureNllStatistics:
    """Return exact NLL, first derivative, and Hessian for one logit scale."""

    if logits.ndim != 3 or token_ids.ndim != 2 or logits.shape[:2] != token_ids.shape:
        raise ValueError("temperature calibration logits and tokens must have aligned ranks")
    if not math.isfinite(logit_scale) or logit_scale <= 0 or token_chunk_size <= 0:
        raise ValueError("temperature calibration scale and token chunk must be positive")
    valid = torch.ones_like(token_ids[:, 1:], dtype=torch.bool)
    if attention_mask is not None:
        if attention_mask.shape != token_ids.shape:
            raise ValueError("temperature calibration attention mask must match token IDs")
        valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    values = logits[:, :-1].reshape(-1, logits.shape[-1])
    labels = token_ids[:, 1:].reshape(-1)
    valid_rows = valid.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    if valid_rows.numel() == 0:
        raise ValueError("temperature calibration has no valid next-token positions")

    nll_parts: list[float] = []
    gradient_parts: list[float] = []
    hessian_parts: list[float] = []
    for row_slice in chunk_slices(int(valid_rows.numel()), token_chunk_size):
        rows = valid_rows[row_slice]
        raw = values.index_select(0, rows).float()
        scaled = raw * logit_scale
        probabilities = torch.softmax(scaled, dim=-1)
        expected = (probabilities * raw).sum(dim=-1)
        variance = (probabilities * raw.square()).sum(dim=-1) - expected.square()
        selected_labels = labels.index_select(0, rows).reshape(-1, 1)
        selected = raw.gather(1, selected_labels).reshape(-1)
        nll_parts.append(
            float((torch.logsumexp(scaled, dim=-1) - logit_scale * selected).sum())
        )
        gradient_parts.append(float((expected - selected).sum()))
        hessian_parts.append(float(variance.clamp_min(0).sum()))
    return TemperatureNllStatistics(
        int(valid_rows.numel()),
        math.fsum(nll_parts),
        math.fsum(gradient_parts),
        math.fsum(hessian_parts),
    )


def combine_temperature_nll_statistics(
    values: Iterable[TemperatureNllStatistics],
) -> TemperatureNllStatistics:
    ordered = tuple(values)
    if not ordered:
        raise ValueError("temperature calibration cannot combine an empty statistic inventory")
    return TemperatureNllStatistics(
        sum(item.token_count for item in ordered),
        math.fsum(item.negative_log_likelihood_sum for item in ordered),
        math.fsum(item.gradient_sum for item in ordered),
        math.fsum(item.hessian_sum for item in ordered),
    )


def fit_logit_temperature(
    evaluate: Callable[[float], TemperatureNllStatistics],
    *,
    initial_logit_scale: float = 1.0,
    minimum_logit_scale: float = 0.5,
    maximum_logit_scale: float = 1.5,
    maximum_update_passes: int = 4,
    convergence_tolerance: float = 1e-4,
    hessian_floor: float = 1e-12,
    resume: tuple[TemperatureFitIteration, ...] = (),
    checkpoint: Callable[[tuple[TemperatureFitIteration, ...]], None] | None = None,
) -> TemperatureFitResult:
    """Fit a bounded positive logit scale with deterministic Newton passes."""

    if (
        not all(
            math.isfinite(value)
            for value in (
                initial_logit_scale,
                minimum_logit_scale,
                maximum_logit_scale,
                convergence_tolerance,
                hessian_floor,
            )
        )
        or not 0 < minimum_logit_scale < initial_logit_scale < maximum_logit_scale
        or maximum_update_passes <= 0
        or convergence_tolerance <= 0
        or hessian_floor <= 0
    ):
        raise ValueError("temperature fitting protocol is invalid")

    scale = initial_logit_scale
    iterations: list[TemperatureFitIteration] = []
    token_count: int | None = None
    converged = False
    for expected_iteration, restored in enumerate(resume, start=1):
        if expected_iteration > maximum_update_passes:
            raise ValueError("temperature checkpoint exceeds its frozen pass limit")
        if restored.iteration != expected_iteration or not math.isclose(
            restored.logit_scale,
            scale,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("temperature checkpoint iteration sequence is invalid")
        if not math.isclose(
            restored.equivalent_temperature,
            1 / restored.logit_scale,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("temperature checkpoint equivalent temperature is invalid")
        if restored.token_count <= 0 or (
            token_count is not None and restored.token_count != token_count
        ):
            raise ValueError("temperature checkpoint token inventory is invalid")
        if restored.hessian_sum <= hessian_floor:
            raise ValueError("temperature checkpoint Hessian is not positive")
        expected_proposed = (
            restored.logit_scale - restored.gradient_sum / restored.hessian_sum
        )
        expected_next = min(
            max(expected_proposed, minimum_logit_scale),
            maximum_logit_scale,
        )
        if (
            not math.isclose(
                restored.proposed_logit_scale,
                expected_proposed,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                restored.next_logit_scale,
                expected_next,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("temperature checkpoint Newton update is invalid")
        if expected_next in {minimum_logit_scale, maximum_logit_scale}:
            raise ValueError("temperature checkpoint reached its allowed scale boundary")
        token_count = restored.token_count
        iterations.append(restored)
        scale = expected_next
        converged = abs(restored.next_logit_scale - restored.logit_scale) <= convergence_tolerance
        if converged and expected_iteration != len(resume):
            raise ValueError("temperature checkpoint continues after convergence")
    for iteration in range(len(iterations) + 1, maximum_update_passes + 1):
        if converged:
            break
        statistics = evaluate(scale)
        if token_count is not None and statistics.token_count != token_count:
            raise ValueError("temperature fitting token inventory changed between passes")
        token_count = statistics.token_count
        if statistics.hessian_sum <= hessian_floor:
            raise ValueError("temperature fitting Hessian is not positive")
        proposed = scale - statistics.gradient_sum / statistics.hessian_sum
        if not math.isfinite(proposed):
            raise ValueError("temperature fitting proposed a non-finite scale")
        next_scale = min(max(proposed, minimum_logit_scale), maximum_logit_scale)
        boundary = next_scale != proposed or next_scale in {
            minimum_logit_scale,
            maximum_logit_scale,
        }
        iterations.append(
            TemperatureFitIteration(
                iteration,
                scale,
                1 / scale,
                statistics.token_count,
                statistics.mean_negative_log_likelihood,
                statistics.gradient_sum,
                statistics.hessian_sum,
                proposed,
                next_scale,
            )
        )
        if checkpoint is not None:
            checkpoint(tuple(iterations))
        if boundary:
            raise ValueError("temperature fitting reached its allowed scale boundary")
        if abs(next_scale - scale) <= convergence_tolerance:
            scale = next_scale
            converged = True
            break
        scale = next_scale
    if not converged:
        raise ValueError("temperature fitting did not converge within its frozen pass limit")
    final = evaluate(scale)
    if token_count is not None and final.token_count != token_count:
        raise ValueError("temperature fitting token inventory changed at final evaluation")
    return TemperatureFitResult(
        TEMPERATURE_CALIBRATION_VERSION,
        initial_logit_scale,
        minimum_logit_scale,
        maximum_logit_scale,
        convergence_tolerance,
        hessian_floor,
        maximum_update_passes,
        tuple(iterations),
        scale,
        1 / scale,
        final.mean_negative_log_likelihood,
        final.token_count,
        True,
        False,
    )


__all__ = [
    "TEMPERATURE_CALIBRATION_VERSION",
    "TemperatureFitIteration",
    "TemperatureFitResult",
    "TemperatureNllStatistics",
    "TemperatureSequenceMetrics",
    "causal_raw_fitted_temperature_metrics",
    "combine_temperature_nll_statistics",
    "fit_logit_temperature",
    "temperature_nll_statistics",
]
