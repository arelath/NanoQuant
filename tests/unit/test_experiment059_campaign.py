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


def test_experiment059_campaign_counts_control_and_candidate_progress(tmp_path: Path) -> None:
    namespace: dict[str, Any] = runpy.run_path("tools/run_experiment059_campaign.py")
    _write_blocks(tmp_path / "059-d2-uniform-control-gemma-3-270m-it", 18)
    _write_blocks(
        tmp_path / "059-best-methods-2bpw-compress-and-benchmark-gemma-3-270m-it",
        3,
    )

    assert namespace["_campaign_progress"](tmp_path) == 21
