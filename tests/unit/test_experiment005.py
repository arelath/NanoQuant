from __future__ import annotations

from pathlib import Path

from tests.support.experiments import load_experiment

_DEFINITION = load_experiment(5)


def test_experiment005_requests_double_vproj_bits_from_experiment003() -> None:
    config = _DEFINITION.config
    experiment = _DEFINITION.workflow
    assert config.intent.experiment_number == 5
    assert config.intent.baseline_run == "003-compress-and-benchmark-gemma-3-4b-it"
    assert experiment.parent_run == Path("evidence/003/003-compress-and-benchmark-gemma-3-4b-it")
    assert experiment.source_packed == Path("outputs/003/packed")
    assert experiment.gguf_output == Path("Results/005/gemma-3-4b-it-vproj-maxrank-nanoquant.gguf")
    assert experiment.layer_suffix == "self_attn.v_proj"
    assert experiment.bit_multiplier == 2.0
    assert experiment.expected_blocks == 34
