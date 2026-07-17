from nanoquant.infrastructure.compression_progress import CompressionProgress
from nanoquant.ports.event_sink import Event


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _event(name: str, **fields: object) -> Event:
    return Event(1, "2026-01-01T00:00:00+00:00", "run", 1, "resident", "info", name, fields)


def test_progress_tracks_layers_and_learns_block_eta() -> None:
    clock = Clock()
    progress = CompressionProgress(clock)

    assert progress.observe(
        _event(
            "compression.progress_initialized",
            total_blocks=3,
            completed_blocks=0,
            completed_wall_seconds=0.0,
            mean_block_seconds=None,
        )
    ) == "checkpoint"
    assert "0/3 [00:00:00<?, ?s/block]" in progress.render()

    assert progress.observe(_event("block.started", block=0, layers=2, completed_blocks=0)) == "checkpoint"
    assert progress.observe(_event("layer.started", layer="self_attn.q_proj", position=0)) == "refresh"
    clock.now = 5.0
    assert progress.observe(_event("layer.completed", layer="self_attn.q_proj")) == "checkpoint"
    assert "16.7%" in progress.render()
    assert "layer=1/2 self_attn.q_proj" in progress.render()

    assert progress.observe(_event("block.completed", block=0, wall_seconds=12.0)) == "checkpoint"
    rendered = progress.render()
    assert "1/3 [00:00:12<00:00:24, 12.0s/block]" in rendered


def test_progress_tracks_calibration_batches_before_compression() -> None:
    clock = Clock()
    progress = CompressionProgress(clock)

    assert progress.observe(_event("calibration.progress_initialized", total_batches=4)) == "checkpoint"
    assert "Calibrating:" in progress.render()
    assert "0/4 [00:00:00<?, ?s/batch]" in progress.render()

    clock.now = 2.0
    assert progress.observe(
        _event("calibration.progress_updated", completed_batches=1, total_batches=4)
    ) == "refresh"
    assert "25.0%" in progress.render()
    assert "1/4 [00:00:02<00:00:06, 2.0s/batch]" in progress.render()

    clock.now = 8.0
    assert progress.observe(
        _event("calibration.progress_completed", completed_batches=4, total_batches=4)
    ) == "finish"
    assert "100.0%" in progress.render()
    assert "4/4" in progress.render()

    assert progress.observe(
        _event(
            "compression.progress_initialized",
            total_blocks=2,
            completed_blocks=0,
        )
    ) == "checkpoint"
    assert "Compressing Layers:" in progress.render()


def test_progress_initialization_seeds_resume_eta() -> None:
    progress = CompressionProgress(lambda: 0.0)

    assert progress.observe(
        _event(
            "compression.progress_initialized",
            total_blocks=4,
            completed_blocks=2,
            completed_wall_seconds=40.0,
            mean_block_seconds=20.0,
        )
    ) == "checkpoint"
    assert "50.0%" in progress.render()
    assert "2/4 [00:00:40<00:00:40, 20.0s/block]" in progress.render()

    assert progress.observe(_event("run.interrupted")) == "finish"
    assert "status=interrupted" in progress.render()


def test_progress_ignores_events_until_valid_initialization() -> None:
    progress = CompressionProgress(lambda: 0.0)

    assert progress.observe(_event("block.started", block=0, layers=2)) == "none"
    assert progress.observe(_event("compression.progress_initialized", total_blocks=0)) == "none"
    assert progress.active is False
