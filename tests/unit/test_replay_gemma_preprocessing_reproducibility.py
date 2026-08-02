from __future__ import annotations

import sys
from pathlib import Path

import tools.replay_gemma_preprocessing_reproducibility as replay


def test_replay_launches_each_run_in_a_fresh_python_process(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[list[str], bool]] = []

    def run(command: list[str], *, check: bool) -> None:
        captured.append((command, check))

    monkeypatch.setattr(replay.subprocess, "run", run)
    launcher = tmp_path / "experiment.py"
    output_root = tmp_path / "replay"
    output = output_root / "run-a"

    replay._run_in_independent_process(launcher, output_root, output, 0.75)

    assert captured == [
        (
            [
                sys.executable,
                str(Path(replay.__file__).resolve()),
                "--launcher",
                str(launcher),
                "--output-root",
                str(output_root),
                "--maximum-wddm-shared-gib",
                "0.75",
                "--single-run-output",
                str(output),
            ],
            True,
        )
    ]
