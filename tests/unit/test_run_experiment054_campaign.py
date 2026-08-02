import json
from pathlib import Path

from tools.run_experiment054_campaign import _completed_blocks, _next_slice_index


def test_completed_blocks_counts_all_campaign_run_journals(tmp_path: Path) -> None:
    control = tmp_path / "control" / "state"
    candidate = tmp_path / "candidate" / "state"
    control.mkdir(parents=True)
    candidate.mkdir(parents=True)
    control.joinpath("journal.jsonl").write_text(
        "\n".join(
            json.dumps({"kind": kind})
            for kind in ("layer", "block", "layer", "block")
        ),
        encoding="utf-8",
    )
    candidate.joinpath("journal.jsonl").write_text(
        json.dumps({"kind": "block"}),
        encoding="utf-8",
    )

    assert _completed_blocks(tmp_path) == 3


def test_next_slice_index_preserves_existing_campaign_logs(tmp_path: Path) -> None:
    tmp_path.joinpath("campaign-slice-001.stdout.log").write_text("first", encoding="utf-8")
    tmp_path.joinpath("campaign-slice-001.stderr.log").write_text("", encoding="utf-8")
    tmp_path.joinpath("campaign-slice-004.stdout.log").write_text("fourth", encoding="utf-8")
    tmp_path.joinpath("unrelated.log").write_text("ignored", encoding="utf-8")

    assert _next_slice_index(tmp_path) == 5
