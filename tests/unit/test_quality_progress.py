import io

import pytest

from nanoquant.infrastructure.quality_progress import QualityProgressLogger


def _logger(now: list[float], output: io.StringIO) -> QualityProgressLogger:
    return QualityProgressLogger(interval_seconds=30.0, clock=lambda: now[0], stream=output)


def test_quality_progress_prints_phases_and_rate_limits_batches() -> None:
    now = [0.0]
    output = io.StringIO()
    progress = _logger(now, output)
    task = {"model": "base", "task": "hellaswag", "task_index": 5, "task_count": 6}

    progress("model_load_started", {"model": "base"})
    progress("task_started", {**task, "sample_count": 100})
    for completed, moment in ((1, 2.0), (2, 10.0), (3, 40.0), (4, 41.0), (10, 50.0)):
        now[0] = moment
        progress("task_batch_completed", {**task, "completed_batches": completed, "total_batches": 10})
    progress(
        "task_completed",
        {**task, "metric": "acc_norm", "value": 0.5, "elapsed_seconds": 50.0},
    )
    progress(
        "model_evaluation_completed",
        {"model": "base", "elapsed_seconds": 61.0, "peak_device_bytes": 3 * 1024**3, "peak_host_bytes": None},
    )

    lines = output.getvalue().splitlines()
    assert all(line.startswith("[quality] ") for line in lines)
    assert lines[0] == "[quality] loading base model"
    assert lines[1] == "[quality] base task hellaswag (5/6) started, 100 samples"
    batches = [line for line in lines if ": batch " in line]
    # First batch, the first batch after the 30 s interval, and the final batch.
    assert [line.split("batch ")[1].split(" ")[0] for line in batches] == ["1/10", "3/10", "10/10"]
    assert "hellaswag (5/6): batch 3/10 (30%), elapsed=00:00:40" in batches[1]
    assert batches[2].endswith("eta=00:00:00")
    assert lines[-2] == "[quality] base task hellaswag acc_norm=0.5000 in 00:00:50"
    assert lines[-1] == "[quality] base model evaluated in 00:01:01, peak device=3.00 GiB, peak host=?"


def test_quality_progress_tracks_batch_phases_independently() -> None:
    now = [0.0]
    output = io.StringIO()
    progress = _logger(now, output)

    progress("wikitext_started", {"model": "base"})
    progress("wikitext_batch_completed", {"model": "base", "completed_batches": 1, "total_batches": 4})
    progress("reasoning_started", {"model": "base", "mode": "thinking", "sample_count": 8})
    progress(
        "reasoning_batch_completed",
        {"model": "base", "mode": "thinking", "completed_batches": 1, "total_batches": 2},
    )
    progress("unexpected_event", {"detail": 3})

    text = output.getvalue()
    assert "base wikitext: batch 1/4 (25%)" in text
    assert "base reasoning thinking: batch 1/2 (50%)" in text
    assert "[quality] unexpected_event: detail=3" in text


def test_quality_progress_rejects_negative_interval() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        QualityProgressLogger(interval_seconds=-1.0)
