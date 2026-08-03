from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from nanoquant.domain.planning import factor_bit_cost
from tools.probe_factor_grouping import (
    GroupSpec,
    MemberSpec,
    _group_cache_key,
    _materialized_member_reconstructions,
    _unique_input_profile_count,
    adjacent_topologies,
    attention_partition_topologies,
    attention_topologies,
    load_objective_profiles,
    maximum_rank_for_budget,
    mlp_partition_topologies,
    requested_topologies,
    summarize_topology,
)


def test_maximum_rank_for_budget_accounts_for_packing_and_scales() -> None:
    target = 1024 * 1152
    rank = maximum_rank_for_budget(1024, 1152, target, scale_bits=16, rank_alignment=32)

    assert factor_bit_cost(1024, 1152, rank, rank_alignment=32).total <= target
    assert factor_bit_cost(1024, 1152, rank + 1, rank_alignment=32).total > target


def test_attention_topologies_cover_the_same_members_with_transposed_o() -> None:
    current, reciprocal = attention_topologies(7)

    current_members = sorted(member.label for group in current.groups for member in group.members)
    reciprocal_members = sorted(member.label for group in reciprocal.groups for member in group.members)
    assert current_members == reciprocal_members == ["7:k", "7:o^T", "7:q", "7:v"]
    assert [group.label for group in current.groups] == ["qkv", "o-transpose"]
    assert [group.label for group in reciprocal.groups] == ["qk", "v-o-transpose"]


def test_attention_rank_shift_moves_capacity_from_qk_to_vo() -> None:
    _current, reciprocal = attention_topologies(7, 64)

    assert reciprocal.variant == "candidate-qk-plus-vo-vo-shift-64"
    assert [group.rank_adjustment for group in reciprocal.groups] == [-64, 64]


def test_attention_partitions_enumerate_bell_four_once() -> None:
    topologies = attention_partition_topologies(7)

    assert len(topologies) == 15
    assert topologies[0].variant == "partition-qkv-o"
    assert {topology.variant for topology in topologies} >= {
        "partition-qkv-o",
        "partition-qk-vo",
        "partition-q-k-v-o",
        "partition-qkvo",
    }
    canonical_partitions = set()
    for topology in topologies:
        labels = []
        covered = []
        for group in topology.groups:
            member_labels = tuple(sorted(member.label for member in group.members))
            labels.append(member_labels)
            covered.extend(member_labels)
        assert sorted(covered) == ["7:k", "7:o^T", "7:q", "7:v"]
        canonical_partitions.add(tuple(sorted(labels)))
    assert len(canonical_partitions) == 15


def test_attention_partition_cache_reuses_identical_member_subsets() -> None:
    topologies = attention_partition_topologies(7)
    qk_with_singletons = next(
        topology for topology in topologies if topology.variant == "partition-qk-v-o"
    )
    qk_with_vo = next(topology for topology in topologies if topology.variant == "partition-qk-vo")
    first_group = next(group for group in qk_with_singletons.groups if group.label == "qk")
    second_group = next(group for group in qk_with_vo.groups if group.label == "qk")

    assert _group_cache_key(qk_with_singletons, first_group, 586) == _group_cache_key(
        qk_with_vo,
        second_group,
        586,
    )


def test_mlp_partitions_enumerate_bell_three_once() -> None:
    topologies = mlp_partition_topologies(7)

    assert [topology.variant for topology in topologies] == [
        "partition-g-u-d",
        "partition-gu-d",
        "partition-gd-u",
        "partition-ud-g",
        "partition-gud",
    ]
    canonical_partitions = set()
    for topology in topologies:
        groups = tuple(
            sorted(tuple(sorted(member.label for member in group.members)) for group in topology.groups)
        )
        canonical_partitions.add(groups)
        covered = sorted(member for group in groups for member in group)
        assert covered == ["7:down^T", "7:gate", "7:up"]
    assert len(canonical_partitions) == 5


def test_load_objective_profiles_reads_resident_artifact(tmp_path: Path) -> None:
    artifact_id = "sha256-" + "a" * 64
    artifact_directory = tmp_path / "artifacts" / "aa" / artifact_id
    artifact_directory.mkdir(parents=True)
    save_file(
        {
            "block_7.mlp.gate_proj.input_importance": torch.tensor([1.0, 2.0]),
            "block_7.mlp.gate_proj.output_importance": torch.tensor([3.0, 4.0, 5.0]),
        },
        artifact_directory / "tensors.safetensors",
    )
    objectives_directory = tmp_path / "artifacts" / "bb" / ("sha256-" + "b" * 64)
    objectives_directory.mkdir(parents=True)
    reference = {"artifact": {"artifact_id": artifact_id}}
    objectives = [
        {
            "layer": {"block": {"index": 7}, "path": "mlp.gate_proj"},
            "input_importance": reference | {"key": "block_7.mlp.gate_proj.input_importance"},
            "output_importance": reference | {"key": "block_7.mlp.gate_proj.output_importance"},
        }
    ]
    objectives_path = objectives_directory / "objectives.json"
    objectives_path.write_text(json.dumps(objectives), encoding="utf-8")

    profiles = load_objective_profiles(objectives_path)

    input_importance, output_importance = profiles["block.7.mlp.gate_proj"]
    assert torch.equal(input_importance, torch.tensor([1.0, 2.0]))
    assert torch.equal(output_importance, torch.tensor([3.0, 4.0, 5.0]))


def test_reciprocal_group_charges_distinct_fisher_input_profiles() -> None:
    v = MemberSpec(7, "v")
    o_transpose = MemberSpec(7, "o", True)
    group = GroupSpec("v-o-transpose", (v, o_transpose))
    profiles = {
        v.calibration_path: (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])),
        o_transpose.calibration_path: (torch.tensor([6.0, 7.0]), torch.tensor([8.0, 9.0, 10.0])),
    }

    assert _unique_input_profile_count(group, profiles) == 2


def test_shared_qk_group_reuses_equivalent_fisher_input_profile() -> None:
    q = MemberSpec(7, "q")
    k = MemberSpec(7, "k")
    group = GroupSpec("qk", (q, k))
    profiles = {
        q.calibration_path: (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])),
        k.calibration_path: (torch.tensor([2.0, 4.0, 6.0]), torch.tensor([7.0])),
    }

    assert _unique_input_profile_count(group, profiles) == 1


def test_materialized_reconstructions_restore_transposed_member_orientation() -> None:
    group = GroupSpec(
        "q-o",
        (
            MemberSpec(7, "q"),
            MemberSpec(7, "o", True),
        ),
    )
    stacked = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    q, o = _materialized_member_reconstructions(group, stacked, (2, 4))

    assert q.weight.shape == (2, 3)
    assert torch.equal(q.weight, stacked[:2])
    assert o.weight.shape == (3, 4)
    assert torch.equal(o.weight, stacked[2:].mT)


def test_adjacent_down_shares_the_output_axis_by_transposing_members() -> None:
    current, shared = adjacent_topologies("down", 3, 4)

    assert all(member.transpose for group in current.groups for member in group.members)
    assert [member.label for member in shared.groups[0].members] == ["3:down^T", "4:down^T"]


def test_requested_topologies_expands_selected_arms_only() -> None:
    topologies = requested_topologies(
        ("attention-reciprocal", "adjacent-up"),
        (5,),
        ((10, 11),),
    )

    assert [topology.key for topology in topologies] == [
        "attention-reciprocal|5|current-qkv-plus-o",
        "attention-reciprocal|5|candidate-qk-plus-vo",
        "adjacent-up|10-11|current-separate",
        "adjacent-up|10-11|candidate-shared",
    ]


def test_requested_topologies_expands_all_attention_partitions() -> None:
    topologies = requested_topologies(
        ("attention-partitions",),
        (5,),
        (),
    )

    assert len(topologies) == 15
    assert all(topology.comparison == "attention-partitions" for topology in topologies)


def test_requested_topologies_filters_attention_partitions() -> None:
    topologies = requested_topologies(
        ("attention-partitions",),
        (5,),
        (),
        attention_partition_variants=("partition-qkv-o", "partition-qv-ko"),
    )

    assert [topology.variant for topology in topologies] == [
        "partition-qkv-o",
        "partition-qv-ko",
    ]


def test_requested_topologies_filters_mlp_partitions() -> None:
    topologies = requested_topologies(
        ("mlp-partitions",),
        (),
        (),
        mlp_blocks=(5,),
        mlp_partition_variants=("partition-g-u-d", "partition-gd-u"),
    )

    assert [topology.variant for topology in topologies] == [
        "partition-g-u-d",
        "partition-gd-u",
    ]


def test_summarize_topology_aggregates_energy_and_physical_bits() -> None:
    topology = attention_topologies(2)[0]
    groups = (
        {
            "group_key": "a",
            "scale_fitted": {"error_energy": 3.0, "target_energy": 12.0},
            "original_space": {"error_energy": 12.0, "target_energy": 12.0},
            "source_elements": 10,
            "target_bits": 10,
            "bit_cost": {"binary_factor_bits": 6, "scale_bits": 2, "padding_bits": 1},
            "wall_seconds": 1.5,
            "peak_device_bytes": 100,
        },
        {
            "group_key": "b",
            "scale_fitted": {"error_energy": 1.0, "target_energy": 4.0},
            "original_space": {"error_energy": 4.0, "target_energy": 4.0},
            "source_elements": 6,
            "target_bits": 6,
            "bit_cost": {"binary_factor_bits": 3, "scale_bits": 1, "padding_bits": 0},
            "wall_seconds": 2.0,
            "peak_device_bytes": 80,
        },
    )

    result = summarize_topology(topology, groups)

    assert result["normalized_rmse"] == 0.5
    assert result["original_normalized_rmse"] == 1.0
    assert result["actual_bits"] == 13
    assert result["actual_bpw"] == 13 / 16
    assert result["unused_target_bits"] == 3
    assert result["wall_seconds"] == 3.5
    assert result["peak_device_bytes"] == 100
