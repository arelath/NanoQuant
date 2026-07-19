from recipes import GEMMA_3_270M_STACKED_QKV_COMPRESSION_TEMPLATE

from tests.support.experiments import load_experiment


def test_experiment013_uses_one_fixed_budget_qkv_group_per_block() -> None:
    experiment = load_experiment(13)
    config = experiment.config

    assert config.model == GEMMA_3_270M_STACKED_QKV_COMPRESSION_TEMPLATE.model
    assert config.factorization.shared_input.enabled is True
    assert tuple((group.name, group.members) for group in config.factorization.shared_input.groups) == (
        (
            "self_attn.attn_qkv",
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ),
    )
    assert config.allocation.retry.enabled is False
    assert config.allocation.maximum_rank_layer_patterns == ()
    assert config.allocation.layer_budget_multipliers == ()
    assert config.block_tuning.non_factorized.epochs_by_layer_position == (8, 4, 3, 6, 2)
    assert experiment.workflow.expected_blocks == 18
