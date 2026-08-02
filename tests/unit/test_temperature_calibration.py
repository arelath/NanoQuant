from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from nanoquant.application.temperature_calibration import (
    TemperatureNllStatistics,
    causal_raw_fitted_temperature_metrics,
    combine_temperature_nll_statistics,
    fit_logit_temperature,
    temperature_nll_statistics,
)


def test_temperature_statistics_match_autograd() -> None:
    logits = torch.tensor(
        [[[0.2, -0.3, 1.1], [0.5, 0.7, -0.4], [1.2, -0.8, 0.1]]],
        dtype=torch.float64,
    )
    tokens = torch.tensor([[0, 2, 1]])
    scale = torch.tensor(0.83, dtype=torch.float64, requires_grad=True)
    selected = logits[:, :-1].reshape(-1, 3)
    labels = tokens[:, 1:].reshape(-1)
    loss = torch.nn.functional.cross_entropy(selected * scale, labels, reduction="sum")
    gradient = torch.autograd.grad(loss, scale, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient, scale)[0]

    actual = temperature_nll_statistics(
        logits,
        tokens,
        logit_scale=float(scale.detach()),
    )

    assert actual.token_count == 2
    assert actual.negative_log_likelihood_sum == pytest.approx(float(loss.detach()), rel=1e-6)
    assert actual.gradient_sum == pytest.approx(float(gradient.detach()), rel=1e-6)
    assert actual.hessian_sum == pytest.approx(float(hessian.detach()), rel=1e-6)


def test_temperature_solver_recovers_known_binary_scale() -> None:
    expected_scale = 1.25
    raw_difference = math.log(3) / expected_scale
    logits = torch.tensor([[[0.0, raw_difference]] * 5])
    tokens = torch.tensor([[0, 1, 1, 1, 0]])

    result = fit_logit_temperature(
        lambda scale: temperature_nll_statistics(
            logits,
            tokens,
            logit_scale=scale,
        ),
    )

    assert result.converged
    assert result.final_logit_scale == pytest.approx(expected_scale, abs=1e-4)
    assert result.equivalent_temperature == pytest.approx(0.8, abs=1e-4)
    assert result.token_count == 4


def test_temperature_statistics_combine_in_fixed_order() -> None:
    combined = combine_temperature_nll_statistics(
        (
            TemperatureNllStatistics(2, 3.0, -1.0, 4.0),
            TemperatureNllStatistics(3, 5.0, 2.0, 6.0),
        )
    )

    assert combined == TemperatureNllStatistics(5, 8.0, 1.0, 10.0)


def test_raw_fitted_metrics_preserve_top1_and_match_raw_at_unit_scale() -> None:
    teacher = torch.tensor([[[0.2, 1.0, -0.3], [0.7, 0.1, -0.2], [0.0, 0.0, 0.0]]])
    student = torch.tensor([[[0.1, 0.8, -0.4], [0.5, 0.3, -0.1], [0.0, 0.0, 0.0]]])
    tokens = torch.tensor([[0, 1, 0]])

    unit = causal_raw_fitted_temperature_metrics(
        teacher,
        student,
        tokens,
        logit_scale=1.0,
        top_k=2,
        token_chunk_size=1,
    )[0]
    scaled = causal_raw_fitted_temperature_metrics(
        teacher,
        student,
        tokens,
        logit_scale=1.25,
        top_k=2,
        token_chunk_size=1,
    )[0]

    assert unit.raw_negative_log_likelihood == pytest.approx(unit.fitted_negative_log_likelihood)
    assert unit.raw_kl_nats_per_token == pytest.approx(unit.fitted_kl_nats_per_token)
    assert unit.raw_student_teacher_topk_mass == pytest.approx(
        unit.fitted_student_teacher_topk_mass
    )
    assert scaled.teacher_top1_agreement == unit.teacher_top1_agreement
    assert scaled.raw_student_teacher_topk_mass != pytest.approx(
        scaled.fitted_student_teacher_topk_mass
    )


def test_temperature_solver_resume_is_exact() -> None:
    logits = torch.tensor([[[0.0, math.log(3) / 1.25]] * 5])
    tokens = torch.tensor([[0, 1, 1, 1, 0]])

    def evaluate(scale: float) -> TemperatureNllStatistics:
        return temperature_nll_statistics(logits, tokens, logit_scale=scale)

    checkpoint: tuple = ()

    def interrupt(iterations: tuple) -> None:
        nonlocal checkpoint
        checkpoint = iterations
        raise InterruptedError("injected temperature-fit interruption")

    with pytest.raises(InterruptedError, match="temperature-fit interruption"):
        fit_logit_temperature(evaluate, checkpoint=interrupt)

    resumed = fit_logit_temperature(evaluate, resume=checkpoint)
    uninterrupted = fit_logit_temperature(evaluate)

    assert resumed == uninterrupted


def test_temperature_solver_rejects_tampered_resume() -> None:
    saved: tuple = ()

    def interrupt(iterations: tuple) -> None:
        nonlocal saved
        saved = iterations
        raise InterruptedError

    with pytest.raises(InterruptedError):
        fit_logit_temperature(
            lambda scale: TemperatureNllStatistics(2, 1.0, scale - 1.2, 1.0),
            checkpoint=interrupt,
        )

    tampered = (replace(saved[0], next_logit_scale=1.1),)
    with pytest.raises(ValueError, match="Newton update"):
        fit_logit_temperature(
            lambda scale: TemperatureNllStatistics(2, 1.0, scale - 1.2, 1.0),
            resume=tampered,
        )


@pytest.mark.parametrize(
    ("evaluate", "message"),
    [
        (lambda _scale: TemperatureNllStatistics(1, 1.0, 1.0, 0.0), "Hessian"),
        (lambda _scale: TemperatureNllStatistics(1, 1.0, -10.0, 1.0), "boundary"),
        (lambda scale: TemperatureNllStatistics(1, 1.0, scale - 1.2, 1e6), "did not converge"),
    ],
)
def test_temperature_solver_fails_closed(
    evaluate,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_logit_temperature(
            evaluate,
            maximum_update_passes=1,
            convergence_tolerance=1e-12,
        )
