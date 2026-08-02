from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from nanoquant.application.kl_budget import KlBudgetArmResult, KlSequenceResult


def _probe_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("probe_composed_context_mlp_refit")


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(2, 3, bias=False)
        self.up_proj = nn.Linear(2, 3, bias=False)
        self.down_proj = nn.Linear(3, 2, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(value))
            * self.up_proj(value)
        )


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = _Mlp()

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        residual = hidden + 0.25
        return (residual + self.mlp(self.post_attention_layernorm(residual)),)


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((_Block(), _Block()))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Base()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        use_cache: bool,
    ) -> tuple[torch.Tensor]:
        assert not use_cache
        hidden = torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)
        for block in self.model.layers:
            hidden = block(hidden)[0]
        return (hidden,)


def test_composed_context_capture_pairs_all_requested_block_values() -> None:
    probe = _probe_module()
    tokens = torch.tensor(((1, 2, 3), (4, 5, 6)))

    captured = probe._capture_mlp_context(
        _Model(),
        (0, 1),
        tokens,
        device="cpu",
    )

    assert set(captured.mlp_inputs) == {0, 1}
    assert captured.mlp_inputs[0].shape == (6, 2)
    assert captured.post_attention_residuals[1].shape == (6, 2)
    assert captured.block_outputs[1].shape == (6, 2)
    assert captured.mlp_inputs[0].dtype is torch.bfloat16
    torch.testing.assert_close(
        captured.mlp_inputs[0],
        captured.post_attention_residuals[0],
    )


def test_state_target_cancels_the_student_residual() -> None:
    probe = _probe_module()
    teacher = probe.MlpContextCapture(
        mlp_inputs={0: torch.zeros((2, 2))},
        post_attention_residuals={0: torch.zeros((2, 2))},
        mlp_outputs={0: torch.ones((2, 2))},
        block_outputs={0: torch.full((2, 2), 3.0)},
        post_feedforward_norm_kinds={0: "identity"},
        post_feedforward_norm_multipliers={0: torch.ones(1)},
        post_feedforward_norm_epsilons={0: 0.0},
    )
    student = probe.MlpContextCapture(
        mlp_inputs={0: torch.zeros((2, 2))},
        post_attention_residuals={0: torch.full((2, 2), 1.25)},
        mlp_outputs={0: torch.ones((2, 2))},
        block_outputs={0: torch.zeros((2, 2))},
        post_feedforward_norm_kinds={0: "identity"},
        post_feedforward_norm_multipliers={0: torch.ones(1)},
        post_feedforward_norm_epsilons={0: 0.0},
    )

    target = probe._state_targets(teacher, student)

    torch.testing.assert_close(target[0], torch.full((2, 2), 1.75).bfloat16())


def test_context_policy_parser_rejects_duplicate_blocks() -> None:
    probe = _probe_module()

    assert probe._parse_policy("0:output,18:joint") == (
        (0, "output"),
        (18, "joint"),
    )
    with pytest.raises(Exception, match="unique"):
        probe._parse_policy("0:output,0:joint")


def test_context_parser_can_restrict_probe_to_teacher_only() -> None:
    probe = _probe_module()
    args = probe._parser().parse_args(
        [
            "--run-output",
            "run",
            "--snapshot",
            "snapshot",
            "--output",
            "output.json",
            "--context",
            "teacher_function",
        ]
    )

    assert args.context == ["teacher_function"]


def test_paired_metric_payload_uses_requested_sequence_metric() -> None:
    probe = _probe_module()
    baseline_sequences = (
        KlSequenceResult(2.0, 0.5, 4),
        KlSequenceResult(4.0, 0.7, 4),
    )
    candidate_sequences = (
        KlSequenceResult(1.5, 0.8, 4),
        KlSequenceResult(3.5, 1.0, 4),
    )
    baseline = KlBudgetArmResult("full", 3.0, 0.6, 8, None, baseline_sequences)
    candidate = KlBudgetArmResult("full", 2.5, 0.9, 8, None, candidate_sequences)

    result = probe._paired_metric_payload(
        baseline,
        candidate,
        "negative_log_likelihood",
        resamples=100,
    )

    assert result["point_delta"] == -0.5
    assert result["lower_delta"] == -0.5
    assert result["upper_delta"] == -0.5
    assert result["improved_with_confidence"] is True


def test_paired_metric_payload_supports_higher_is_better_metrics() -> None:
    probe = _probe_module()
    baseline_sequences = (
        KlSequenceResult(2.0, 0.5, 4, 0.5),
        KlSequenceResult(4.0, 0.7, 4, 0.5),
    )
    candidate_sequences = (
        KlSequenceResult(1.5, 0.8, 4, 0.75),
        KlSequenceResult(3.5, 1.0, 4, 0.75),
    )
    baseline = KlBudgetArmResult(
        "full",
        3.0,
        0.6,
        8,
        None,
        baseline_sequences,
        teacher_top1_agreement=0.5,
    )
    candidate = KlBudgetArmResult(
        "full",
        2.5,
        0.9,
        8,
        None,
        candidate_sequences,
        teacher_top1_agreement=0.75,
    )

    result = probe._paired_metric_payload(
        baseline,
        candidate,
        "teacher_top1_agreement",
        resamples=100,
        higher_is_better=True,
    )

    assert result["point_delta"] == 0.25
    assert result["lower_delta"] == 0.25
    assert result["upper_delta"] == 0.25
    assert result["improved_with_confidence"] is True
