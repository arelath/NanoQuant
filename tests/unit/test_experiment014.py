from recipes import GEMMA_3_270M_COMPRESSION_TEMPLATE

from nanoquant.config.schema import AllocationStrategy
from tests.support.experiments import load_experiment


def test_experiment014_runs_full_reconstruction_planning_before_stacked_compression() -> None:
    experiment = load_experiment(14)
    config = experiment.config
    reconstruction = config.allocation.reconstruction

    assert config.model == GEMMA_3_270M_COMPRESSION_TEMPLATE.model
    assert config.allocation.strategy is AllocationStrategy.RECONSTRUCTION_AWARE
    assert reconstruction.enabled is True
    assert reconstruction.probe_admm is not None
    assert reconstruction.probe_admm.outer_iterations == 400
    assert reconstruction.probe_admm.transpose_wide is True
    assert reconstruction.sensitivity_strength == 0.25
    assert {curve.unit_pattern for curve in reconstruction.response_curves} == {
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.o_proj",
        "self_attn.attn_qkv",
    }
    assert config.allocation.retry.enabled is False
    assert config.allocation.maximum_rank_layer_patterns == ()
    assert config.allocation.layer_budget_multipliers == ()
    assert config.factorization.shared_input.enabled is True
    assert experiment.workflow.expected_blocks == 18
