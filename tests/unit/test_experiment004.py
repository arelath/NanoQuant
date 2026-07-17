from __future__ import annotations

from pathlib import Path

import pytest

from nanoquant.rank_expansion_experiment import _candidate_comparison
from tests.support.experiments import load_experiment

_DEFINITION = load_experiment(4)


def test_experiment004_is_a_selective_experiment003_derivative() -> None:
    config = _DEFINITION.config
    experiment = _DEFINITION.workflow
    assert config.intent.experiment_number == 4
    assert config.intent.baseline_run == "003-compress-and-benchmark-gemma-3-4b-it"
    assert experiment.parent_run == Path("evidence/003/003-compress-and-benchmark-gemma-3-4b-it")
    assert experiment.source_packed == Path("outputs/003/packed")
    assert experiment.layer_suffix == "self_attn.v_proj"
    assert experiment.bit_multiplier == 1.30
    assert experiment.expected_blocks == 34


def test_candidate_comparison_uses_experiment003_frozen_metrics() -> None:
    baseline = {
        "comparison": {
            "wikitext": {"frozen_perplexity": 80.0},
            "tasks": [{"task_name": "piqa", "metric": "acc_norm", "frozen": 0.60}],
        }
    }
    candidate = {
        "comparison": {
            "wikitext": {"frozen_perplexity": 72.0},
            "tasks": [{"task_name": "piqa", "metric": "acc_norm", "frozen": 0.65}],
        }
    }

    result = _candidate_comparison(baseline, candidate)

    assert result["wikitext"]["relative_change"] == pytest.approx(-0.10)
    assert result["tasks"][0]["delta"] == pytest.approx(0.05)
