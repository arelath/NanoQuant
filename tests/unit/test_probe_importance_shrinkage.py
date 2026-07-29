from __future__ import annotations

from tools.probe_importance_shrinkage import (
    _aggregate_summaries,
    _parse_floats,
    _shrinkage_key,
    block_topology,
)


def test_block_topology_covers_all_layers_with_qkv_fused() -> None:
    topology = block_topology(12, 0.6)

    assert topology.variant == "shrink-0.6"
    assert [group.label for group in topology.groups] == ["qkv", "o", "gate", "up", "down"]
    assert sorted(member.projection for group in topology.groups for member in group.members) == [
        "down",
        "gate",
        "k",
        "o",
        "q",
        "up",
        "v",
    ]


def test_aggregate_summaries_uses_energy_weighting_and_physical_bits() -> None:
    result = _aggregate_summaries(
        (
            {
                "error_energy": 1.0,
                "target_energy": 4.0,
                "original_error_energy": 4.0,
                "original_target_energy": 16.0,
                "source_elements": 10,
                "actual_bits": 9,
            },
            {
                "error_energy": 3.0,
                "target_energy": 12.0,
                "original_error_energy": 12.0,
                "original_target_energy": 48.0,
                "source_elements": 30,
                "actual_bits": 27,
            },
        )
    )

    assert result["objective_normalized_rmse"] == 0.5
    assert result["original_normalized_rmse"] == 0.5
    assert result["actual_bpw"] == 0.9


def test_shrinkage_parser_and_key_are_stable() -> None:
    assert _parse_floats("0, 0.3, 1") == (0.0, 0.3, 1.0)
    assert _shrinkage_key(0.6000000000000001) == "0.6"
