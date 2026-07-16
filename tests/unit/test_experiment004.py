from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from recipes import EXPERIMENT_004, EXPERIMENT_004_CONFIG

from nanoquant.rank_expansion_experiment import _candidate_comparison


def test_experiment004_is_a_selective_experiment003_derivative() -> None:
    assert EXPERIMENT_004_CONFIG.intent.experiment_number == 4
    assert EXPERIMENT_004_CONFIG.intent.baseline_run == "003-compress-and-benchmark-gemma-3-4b-it-v5"
    assert EXPERIMENT_004.parent_run == Path(
        "evidence/m10/003-compress-and-benchmark-gemma-3-4b-it-v5"
    )
    assert EXPERIMENT_004.layer_suffix == "self_attn.v_proj"
    assert EXPERIMENT_004.bit_multiplier == 1.30
    assert EXPERIMENT_004.expected_blocks == 34


def test_experiment004_runfile_imports_canonical_recipe() -> None:
    namespace = runpy.run_path("experiments/004-gemma-3-4b-it-vproj-plus30.py")

    assert namespace["CONFIG"] is EXPERIMENT_004_CONFIG
    assert namespace["EXPERIMENT"] is EXPERIMENT_004


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
