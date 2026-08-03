from __future__ import annotations

import pytest
import torch

from tools.probe_real_block_tabu import _residual_target
from tools.probe_real_block_tabu_functional import _parse_member, _reconstruction_sets


def test_residual_target_protects_removed_outlier_columns() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    importance = torch.tensor([1.0, 2.0, 4.0, 8.0])

    target, protected = _residual_target(
        source,
        importance,
        torch.tensor([1, 3]),
        removed_column_importance="zero",
    )

    assert torch.equal(target[:, 1], torch.zeros(3))
    assert torch.equal(target[:, 3], torch.zeros(3))
    assert torch.equal(target[:, [0, 2]], source[:, [0, 2]])
    expected_floor = importance.median() * 1e-4
    assert protected[1].item() == pytest.approx(expected_floor.item())
    assert protected[3].item() == pytest.approx(expected_floor.item())
    assert torch.equal(protected[[0, 2]], importance[[0, 2]])


def test_functional_inventory_can_select_one_owner_from_full_weight_file() -> None:
    probe = {
        "status": "completed",
        "results": [
            {
                "block": 0,
                "owner": "self_attn.attn_qkv",
                "members": ["0:self_attn.q_proj", "0:self_attn.k_proj"],
                "control_nrmse": 0.5,
                "tabu_nrmse": 0.4,
            },
            {
                "block": 12,
                "owner": "mlp.gate_proj",
                "members": ["12:mlp.gate_proj"],
                "control_nrmse": 0.3,
                "tabu_nrmse": 0.2,
            },
        ],
    }
    weights = {
        f"{variant}.block_{block}.{path}": torch.ones(2, 2)
        for variant in ("control", "tabu")
        for block, path in (
            (0, "self_attn.q_proj"),
            (0, "self_attn.k_proj"),
            (12, "mlp.gate_proj"),
        )
    }

    sets, blocks, owners = _reconstruction_sets(
        probe,
        weights,
        ("12:mlp.gate_proj",),
    )

    assert blocks == (12,)
    assert owners == ("12:mlp.gate_proj",)
    assert tuple(item.layer.path for item in sets["control"].layers) == ("mlp.gate_proj",)
    assert sets["control"].layers[0].weighted_normalized_squared_error == pytest.approx(0.09)
    assert sets["tabu"].layers[0].weighted_normalized_squared_error == pytest.approx(0.04)
    assert sets["tabu"].unit_members == (
        ("12:mlp.gate_proj", (_parse_member("12:mlp.gate_proj"),)),
    )


def test_functional_inventory_rejects_missing_selected_owner() -> None:
    with pytest.raises(ValueError, match="absent"):
        _reconstruction_sets(
            {
                "status": "completed",
                "results": [
                    {
                        "block": 0,
                        "owner": "mlp.gate_proj",
                        "members": ["0:mlp.gate_proj"],
                        "control_nrmse": 0.5,
                        "tabu_nrmse": 0.4,
                    }
                ],
            },
            {},
            ("25:self_attn.attn_qkv",),
        )
