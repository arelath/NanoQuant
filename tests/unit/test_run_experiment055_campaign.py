import json
from pathlib import Path

from tools.run_experiment055_campaign import _completed_blocks, _next_slice_index


def test_completed_blocks_counts_only_candidate_run(tmp_path: Path) -> None:
    run = tmp_path / "candidate"
    state = run / "state"
    state.mkdir(parents=True)
    records = (
        {"kind": "layer"},
        {"kind": "block", "block": 0},
        {"kind": "block", "block": 1},
    )
    (state / "journal.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    unrelated = tmp_path / "control" / "state"
    unrelated.mkdir(parents=True)
    (unrelated / "journal.jsonl").write_text(
        json.dumps({"kind": "block", "block": 0}) + "\n",
        encoding="utf-8",
    )

    assert _completed_blocks(run) == 2


def test_next_slice_index_preserves_existing_logs(tmp_path: Path) -> None:
    (tmp_path / "campaign-slice-002.stdout.log").touch()
    (tmp_path / "campaign-slice-014.stderr.log").touch()
    (tmp_path / "campaign-slice-not-an-index.log").touch()

    assert _next_slice_index(tmp_path) == 15
