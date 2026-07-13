from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.compare_block_trajectories import (
    compare_trajectories,
    load_legacy_trajectory,
    load_rewrite_trajectory,
    render_markdown,
)


def _commit_block(root: Path, block: int, loss: float, identity: dict[str, str], suffix: str) -> dict[str, object]:
    artifact_id = "sha256-" + suffix * 64
    artifact = root / "artifacts" / artifact_id[7:9] / artifact_id
    artifact.mkdir(parents=True)
    (artifact / "block-result.json").write_text(
        json.dumps({"block": {"index": block}, "losses": {"final_frozen_pre_kd": loss}}),
        encoding="utf-8",
    )
    return {"kind": "block", "block": block, "artifact_id": artifact_id, "identity": identity}


def test_trajectory_comparison_uses_latest_identity_and_aligns_multiple_baselines(tmp_path: Path) -> None:
    old = {"config_hash": "old", "model_hash": "model", "plan_hash": "plan"}
    active = {"config_hash": "new", "model_hash": "model", "plan_hash": "plan"}
    records = [
        _commit_block(tmp_path, 0, 99.0, old, "a"),
        _commit_block(tmp_path, 0, 1.0, active, "b"),
        _commit_block(tmp_path, 1, 2.0, active, "c"),
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


def test_rewrite_trajectory_rejects_noncontiguous_active_prefix(tmp_path: Path) -> None:
    identity = {"config_hash": "new", "model_hash": "model", "plan_hash": "plan"}
    records = [_commit_block(tmp_path, 1, 2.0, identity, "d")]
    state = tmp_path / "state"
    state.mkdir()
    (state / "journal.jsonl").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contiguous"):
        load_rewrite_trajectory(tmp_path)


def test_rewrite_trajectory_does_not_fall_back_to_stale_identity(tmp_path: Path) -> None:
    old = {"config_hash": "old", "model_hash": "model", "plan_hash": "plan"}
    active = {"config_hash": "new", "model_hash": "model", "plan_hash": "plan"}
    old_block = _commit_block(tmp_path, 0, 1.0, old, "e")
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
