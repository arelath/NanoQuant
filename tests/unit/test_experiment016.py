from recipes import GEMMA_3_270M_COMPRESSION_TEMPLATE

from tests.support.experiments import load_experiment


def test_experiment016_increases_only_down_and_edge_importance() -> None:
    experiment = load_experiment(16)
    importance = experiment.config.allocation.reconstruction.importance
    multipliers = {item.pattern: item.multiplier for item in importance.layer_multipliers}

    assert experiment.config.model == GEMMA_3_270M_COMPRESSION_TEMPLATE.model
    assert multipliers["mlp.down_proj"] == 1.50
    assert {value for pattern, value in multipliers.items() if pattern != "mlp.down_proj"} == {1.25}
    assert importance.edge_block_multiplier == 1.30
    assert importance.protected_edge_block_count == 1
    assert experiment.config.allocation.reconstruction.sensitivity_strength == 0.75
    assert experiment.workflow.expected_blocks == 18
