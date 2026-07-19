from recipes import GEMMA_3_270M_ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE

from nanoquant.config.schema import AllocationStrategy
from tests.support.experiments import load_experiment


def test_experiment015_protects_important_layers_and_edge_blocks() -> None:
    experiment = load_experiment(15)
    config = experiment.config
    reconstruction = config.allocation.reconstruction
    importance = reconstruction.importance

    assert config.model == GEMMA_3_270M_ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE.model
    assert config.allocation.strategy is AllocationStrategy.RECONSTRUCTION_AWARE
    assert importance.protected_layer_patterns == (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.down_proj",
    )
    assert {item.pattern: item.multiplier for item in importance.layer_multipliers} == {
        pattern: 1.25 for pattern in importance.protected_layer_patterns
    }
    assert importance.edge_block_multiplier == 1.25
    assert importance.protected_edge_block_count == 1
    assert config.factorization.shared_input.enabled is True
    assert experiment.workflow.expected_blocks == 18
