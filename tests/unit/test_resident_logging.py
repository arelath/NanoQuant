import json
from pathlib import Path

import pytest

from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.resident_quantization import _logged_operation


def test_logged_operation_preserves_failure_and_records_location(tmp_path: Path) -> None:
    sink = JsonlEventSink(tmp_path / "events.jsonl", "fixture")

    with pytest.raises(RuntimeError, match="injected"):
        with _logged_operation(
            sink,
            "factorized_tuning",
            block=3,
            layer="mlp.down_proj",
            epochs=8,
        ):
            raise RuntimeError("injected failure")

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["name"] for event in events] == [
        "factorized_tuning.started",
        "factorized_tuning.failed",
    ]
    failure = events[-1]
    assert failure["severity"] == "error"
    assert failure["fields"]["block"] == 3
    assert failure["fields"]["layer"] == "mlp.down_proj"
    assert failure["fields"]["epochs"] == 8
    assert failure["fields"]["error_type"] == "RuntimeError"
    assert failure["fields"]["wall_seconds"] >= 0
