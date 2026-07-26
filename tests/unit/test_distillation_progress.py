import io

import pytest

from nanoquant.infrastructure.distillation_progress import DistillationProgressLogger


def test_distillation_progress_logs_first_periodic_and_terminal_updates() -> None:
    now = [0.0]
    output = io.StringIO()
    progress = DistillationProgressLogger(
        interval_seconds=30.0,
        clock=lambda: now[0],
        stream=output,
    )

    progress(
        "teacher_cache.epoch_started",
        {"epoch": 0, "epochs": 2, "total_batches": 10},
    )
    now[0] = 2.0
    progress(
        "teacher_cache.batch_completed",
        {
            "epoch": 0,
            "epochs": 2,
            "completed_batches": 1,
            "total_batches": 10,
            "selected_tokens": 512,
            "cache_bytes": 1024 * 1024,
        },
    )
    now[0] = 10.0
    progress(
        "teacher_cache.batch_completed",
        {
            "epoch": 0,
            "epochs": 2,
            "completed_batches": 2,
            "total_batches": 10,
            "selected_tokens": 512,
            "cache_bytes": 2 * 1024 * 1024,
        },
    )
    now[0] = 33.0
    progress(
        "teacher_cache.batch_completed",
        {
            "epoch": 0,
            "epochs": 2,
            "completed_batches": 3,
            "total_batches": 10,
            "selected_tokens": 512,
            "cache_bytes": 3 * 1024 * 1024,
        },
    )
    progress(
        "teacher_cache.epoch_completed",
        {
            "epoch": 0,
            "epochs": 2,
            "completed_batches": 10,
            "cache_bytes": 10 * 1024 * 1024,
        },
    )

    lines = output.getvalue().splitlines()
    assert len(lines) == 4
    assert "teacher cache epoch 1/2 started (10 batches)" in lines[0]
    assert "batch 1/10" in lines[1]
    assert "elapsed=00:00:02, eta=00:00:18" in lines[1]
    assert "batch 3/10" in lines[2]
    assert "10 batches, cache=10.0 MiB" in lines[3]


def test_distillation_training_progress_includes_loss_lr_resume_and_eta() -> None:
    now = [100.0]
    output = io.StringIO()
    progress = DistillationProgressLogger(
        interval_seconds=30.0,
        clock=lambda: now[0],
        stream=output,
    )

    progress(
        "training.started",
        {
            "epochs": 8,
            "completed_steps": 100,
            "total_steps": 500,
            "selected_parameters": 12,
        },
    )
    progress(
        "training.epoch_started",
        {"epoch": 2, "epochs": 8, "total_batches": 50},
    )
    now[0] = 4.0 + 100.0
    progress(
        "training.batch_completed",
        {
            "epoch": 2,
            "epochs": 8,
            "completed_batches": 1,
            "total_batches": 50,
            "completed_steps": 101,
            "total_steps": 500,
            "batch_loss": 2.75,
            "epoch_mean_loss": 2.75,
            "learning_rate": 9.5e-6,
        },
    )
    progress(
        "training.epoch_completed",
        {
            "epoch": 2,
            "epochs": 8,
            "completed_steps": 150,
            "total_steps": 500,
            "epoch_mean_loss": 2.5,
        },
    )
    progress(
        "training.completed",
        {
            "completed_steps": 500,
            "total_steps": 500,
            "final_epoch_loss": 2.25,
        },
    )

    rendered = output.getvalue()
    assert "training started at step 100/500" in rendered
    assert "training epoch 3/8 started (50 batches)" in rendered
    assert "step 101/500, loss=2.750000, epoch_mean=2.750000, lr=9.500e-06" in rendered
    assert "elapsed=00:00:04, eta=00:26:36" in rendered
    assert "mean_loss=2.500000" in rendered
    assert "final_epoch_loss=2.250000" in rendered


def test_distillation_progress_rejects_negative_interval() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        DistillationProgressLogger(interval_seconds=-1.0)
