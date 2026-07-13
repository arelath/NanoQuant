from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.config.schema import ProfilingConfig
from nanoquant.infrastructure.profiling import Profiler
from tools.evaluate_wikitext import _evaluate


class _UniformModel(nn.Module):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 8, device=input_ids.device))


def test_profiled_wikitext_evaluation_preserves_exact_serial_result() -> None:
    model = _UniformModel()
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 1]])
    expected = _evaluate(model, tokens, "cpu", 1)
    profiler = Profiler(ProfilingConfig(), run_id="wikitext")

    actual = _evaluate(model, tokens, "cpu", 1, profiler)

    for metric in (
        "total_negative_log_likelihood",
        "mean_negative_log_likelihood",
        "perplexity",
        "token_count",
        "window_count",
    ):
        assert actual[metric] == expected[metric]
    assert actual["perplexity"] == pytest.approx(8.0)
    assert actual["mean_negative_log_likelihood"] == pytest.approx(math.log(8))
    payload = profiler.snapshot()
    phases = {phase["path"]: phase for phase in payload["phases"]}
    assert {"tokens_to_device", "causal_nll"} <= phases.keys()
    counters = {counter["name"]: counter for counter in payload["counters"]}
    assert counters["evaluation.tokens"]["total"] == 6
    assert counters["evaluation.windows"]["total"] == 2
