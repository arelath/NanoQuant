import json
import runpy
from pathlib import Path
from typing import Any


def _write_blocks(run_root: Path, count: int) -> None:
    journal = run_root / "state" / "journal.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        "".join(json.dumps({"kind": "block", "index": index}) + "\n" for index in range(count)),
        encoding="utf-8",
    )


def test_experiment060_campaign_counts_control_and_candidate_progress(tmp_path: Path) -> None:
    namespace: dict[str, Any] = runpy.run_path("tools/run_experiment060_campaign.py")
    _write_blocks(tmp_path / "060-d2-uniform-control-gemma-3-1b-it", 26)
    _write_blocks(
        tmp_path / "060-best-methods-2bpw-compress-and-benchmark-gemma-3-1b-it",
        3,
    )

    assert namespace["_campaign_progress"](tmp_path) == 29
