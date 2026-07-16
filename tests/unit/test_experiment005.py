from __future__ import annotations

import runpy
from pathlib import Path

from nanoquant.recipes import EXPERIMENT_005, EXPERIMENT_005_CONFIG


def test_experiment005_requests_double_vproj_bits_from_experiment003() -> None:
    assert EXPERIMENT_005_CONFIG.intent.experiment_number == 5
    assert EXPERIMENT_005_CONFIG.intent.baseline_run == "003-compress-and-benchmark-gemma-3-4b-it-v5"
    assert EXPERIMENT_005.parent_run == Path(
        "evidence/m10/003-compress-and-benchmark-gemma-3-4b-it-v5"
    )
    assert EXPERIMENT_005.layer_suffix == "self_attn.v_proj"
    assert EXPERIMENT_005.bit_multiplier == 2.0
    assert EXPERIMENT_005.expected_blocks == 34


def test_experiment005_runfile_imports_canonical_recipe() -> None:
    namespace = runpy.run_path("experiments/005-gemma-3-4b-it-vproj-double-request.py")

    assert namespace["CONFIG"] is EXPERIMENT_005_CONFIG
    assert namespace["EXPERIMENT"] is EXPERIMENT_005
