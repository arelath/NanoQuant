from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from tools.compare_block_trajectories import (
    compare_rank_allocations,
    compare_trajectories,
    load_legacy_rank_csv,
    load_legacy_trajectory,
    load_rewrite_trajectory,
    render_markdown,
)


def _commit_block(root: Path, block: int, loss: float, identity: dict[str, str]) -> dict[str, object]:
    artifacts = LocalArtifactStore(root / "artifacts")
    with artifacts.begin_write("block-result") as writer:
        (writer.path / "block-result.json").write_text(
            json.dumps(
                {
                    "block": {"index": block},
                    "identity": identity,
                    "layers": [
                        {
                            "actual_bit_cost": {"binary_factor_bits": 10 + block, "scale_bits": 2},
                            "frozen_state": {"rank": block + 1},
                            "layer": {"block": {"index": block}, "path": "mlp.gate_proj"},
                            "plan": {"source_weight": {"spec": {"shape": [2, 3]}}},
                        }
                    ],
                    "losses": {"final_frozen_pre_kd": loss},
                }
            ),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    return {"kind": "block", "block": block, "artifact_id": descriptor.artifact_id, "identity": identity}


def test_trajectory_comparison_uses_latest_identity_and_aligns_multiple_baselines(tmp_path: Path) -> None:
    old = {"config_hash": "old", "model_hash": "model", "plan_hash": "plan"}
    active = {"config_hash": "new", "model_hash": "model", "plan_hash": "plan"}
    records = [
        _commit_block(tmp_path, 0, 99.0, old),
        _commit_block(tmp_path, 0, 1.0, active),
        _commit_block(tmp_path, 1, 2.0, active),
    ]
    state = tmp_path / "state"
    state.mkdir()
    (state / "journal.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    contemporary = tmp_path / "contemporary.log"
    contemporary.write_text(
        "Post-block scale refit summary: 2 -> 1.25\n"
        "Post-block scale refit summary: 3 -> 2.5e+00\n",
        encoding="utf-8",
    )
    historical = tmp_path / "historical.log"
    historical.write_text("Post-block scale refit summary: 2 -> 0.8\n", encoding="utf-8")

    rewrite = load_rewrite_trajectory(tmp_path)
    result = compare_trajectories(
        rewrite,
        (
            ("contemporary", contemporary, load_legacy_trajectory(contemporary)),
            ("historical", historical, load_legacy_trajectory(historical)),
        ),
    )

    assert rewrite.identity == active
    assert rewrite.losses == (1.0, 2.0)
    assert [(value.block, value.rank, value.actual_bits) for value in rewrite.layer_budgets] == [
        (0, 1, 12),
        (1, 2, 13),
    ]
    assert result["rewrite_block_count"] == 2
    baselines = {value["name"]: value for value in result["baselines"]}
    assert baselines["contemporary"]["rewrite_lower_count"] == 2
    assert baselines["historical"]["paired_block_count"] == 1
    blocks = result["blocks"]
    assert blocks[0]["baselines"]["contemporary"]["percent_delta"] == pytest.approx(-20.0)
    assert blocks[1]["baselines"]["historical"] is None
    rendered = render_markdown(result)
    assert "rewrite lower at 2/2" in rendered
    assert "-20.00%" in rendered


def test_rank_comparison_aligns_committed_prefix_and_distinguishes_total_bits(tmp_path: Path) -> None:
    identity = {"config_hash": "new", "model_hash": "model", "plan_hash": "plan"}
    records = [_commit_block(tmp_path, 0, 1.0, identity), _commit_block(tmp_path, 1, 2.0, identity)]
    state = tmp_path / "state"
    state.mkdir()
    (state / "journal.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    ranks = tmp_path / "rank-utility.csv"
    ranks.write_text(
        "block,layer,rank,binary_bits\n"
        "1,mlp.gate_proj,1,10\n"
        "2,mlp.gate_proj,3,11\n"
        "3,mlp.gate_proj,99,999\n",
        encoding="utf-8",
    )

    rewrite = load_rewrite_trajectory(tmp_path)
    result = compare_rank_allocations(
        rewrite,
        (("legacy", ranks, load_legacy_rank_csv(ranks)),),
    )[0]

    assert result["paired_layer_count"] == 2
    assert result["rank_mismatch_count"] == 1
    assert result["rewrite_rank_sum"] == 3
    assert result["legacy_rank_sum"] == 4
    assert result["rewrite_source_parameters"] == 12
    assert result["paired_source_parameters"] == 12
    assert result["rewrite_actual_bits"] == 25
    assert result["rewrite_effective_bpw"] == pytest.approx(25 / 12)
    assert result["legacy_rank_dependent_bits"] == 21
    assert result["legacy_rank_dependent_bpw"] == pytest.approx(21 / 12)
    assert result["rank_mismatches"] == [
        {"block": 1, "layer": "mlp.gate_proj", "rewrite_rank": 2, "legacy_rank": 3, "rank_delta": -1}
    ]
    comparison = compare_trajectories(rewrite, (("legacy", tmp_path / "legacy.log", (1.0, 2.0)),))
    comparison["rank_baselines"] = [result]
    assert "1 rank mismatches" in render_markdown(comparison)

    with pytest.raises(ValueError, match="unique"):
        compare_rank_allocations(
            rewrite,
            (
                ("legacy", ranks, load_legacy_rank_csv(ranks)),
                ("legacy", ranks, load_legacy_rank_csv(ranks)),
            ),
        )


def test_rewrite_trajectory_rejects_noncontiguous_active_prefix(tmp_path: Path) -> None:
    identity = {"config_hash": "new", "model_hash": "model", "plan_hash": "plan"}
    records = [_commit_block(tmp_path, 1, 2.0, identity)]
    state = tmp_path / "state"
    state.mkdir()
    (state / "journal.jsonl").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contiguous"):
        load_rewrite_trajectory(tmp_path)


def test_rewrite_trajectory_does_not_fall_back_to_stale_identity(tmp_path: Path) -> None:
    old = {"config_hash": "old", "model_hash": "model", "plan_hash": "plan"}
    active = {"config_hash": "new", "model_hash": "model", "plan_hash": "plan"}
    old_block = _commit_block(tmp_path, 0, 1.0, old)
    active_layer = {
        "kind": "layer",
        "block": 0,
        "layer": "mlp.gate_proj",
        "artifact_id": "sha256-" + "f" * 64,
        "identity": active,
    }
    state = tmp_path / "state"
    state.mkdir()
    (state / "journal.jsonl").write_text(
        json.dumps(old_block) + "\n" + json.dumps(active_layer) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="active journal identity"):
        load_rewrite_trajectory(tmp_path)


def test_rewrite_trajectory_validates_block_artifact_hashes(tmp_path: Path) -> None:
    identity = {"config_hash": "new", "model_hash": "model", "plan_hash": "plan"}
    record = _commit_block(tmp_path, 0, 1.0, identity)
    state = tmp_path / "state"
    state.mkdir()
    (state / "journal.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    block_path = artifacts.path_for(str(record["artifact_id"])) / "block-result.json"
    block_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactCorruptionError, match="ART001"):
        load_rewrite_trajectory(tmp_path)


def test_rewrite_trajectory_rejects_block_payload_from_another_identity(tmp_path: Path) -> None:
    active = {"config_hash": "new", "model_hash": "model", "plan_hash": "plan"}
    stale = {"config_hash": "old", "model_hash": "model", "plan_hash": "plan"}
    record = _commit_block(tmp_path, 0, 1.0, stale)
    record["identity"] = active
    state = tmp_path / "state"
    state.mkdir()
    (state / "journal.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid block result"):
        load_rewrite_trajectory(tmp_path)
