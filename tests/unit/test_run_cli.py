import json
from pathlib import Path

from nanoquant.cli.main import main
from nanoquant.config.schema import ModelConfig, RunConfig
from nanoquant.domain.runs import RunStatus
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.runs import RunDirectory, initial_manifest, launcher_provenance, transition


def _managed_run(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "runs"
    launcher = tmp_path / "019_fixture.py"
    launcher.write_text("# fixture\n", encoding="utf-8")
    manifest = initial_manifest(
        RunConfig(ModelConfig("fixture")),
        launcher_provenance(launcher, 19),
        {},
        run_id="run_cli_fixture",
    )
    directory = RunDirectory(root, manifest.run_id)
    running = transition(manifest, RunStatus.RUNNING)
    completed = transition(running, RunStatus.COMPLETED)
    directory.write_manifest(completed)
    with JsonlEventSink(directory.events_path, manifest.run_id) as events:
        events.emit("run", "info", "run.started")
        events.emit("run", "warning", "run.warning", message="first\nsecond")
        events.emit("run", "info", "run.completed")
    return root, manifest.run_id


def test_runs_list_show_and_path_use_live_manifests(tmp_path: Path, capsys) -> None:
    root, run_id = _managed_run(tmp_path)

    assert main(["runs", "list", "--run-root", str(root), "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["run_id"] == run_id
    assert listed[0]["status"] == "completed"

    assert main(["runs", "show", "latest", "--run-root", str(root), "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["manifest"]["launcher"]["experiment_number"] == 19
    assert shown["events_integrity"] == "ok"

    assert main(["runs", "path", "exp:19", "--kind", "journal", "--run-root", str(root)]) == 0
    assert capsys.readouterr().out.strip() == str((root / run_id / "state" / "journal.jsonl").resolve())


def test_logs_render_filter_save_and_follow_completed_run(tmp_path: Path, capsys) -> None:
    root, run_id = _managed_run(tmp_path)

    assert main(["logs", run_id, "--run-root", str(root), "--level", "warning", "--save"]) == 0
    output = capsys.readouterr().out
    assert "run.warning" in output
    assert "run.started" not in output
    rendered = (root / run_id / "run.log").read_text(encoding="utf-8")
    assert len(rendered.splitlines()) == 3
    assert 'message="first\\nsecond"' in rendered

    assert main(["logs", "latest", "--run-root", str(root), "--follow", "--poll-seconds", "0.01"]) == 0
    captured = capsys.readouterr()
    assert "run.completed" in captured.out
    assert "status=completed" in captured.err


def test_unmanaged_path_is_read_only_and_missing_selector_has_stable_code(tmp_path: Path, capsys) -> None:
    unmanaged = tmp_path / "legacy"
    with JsonlEventSink(unmanaged / "events.jsonl", "legacy") as events:
        events.emit("run", "info", "run.started")

    assert main(["logs", "--path", str(unmanaged)]) == 0
    assert "run.started" in capsys.readouterr().out
    assert not (unmanaged / "manifest.json").exists()

    assert main(["runs", "show", "missing", "--run-root", str(tmp_path / "runs")]) == 3
    assert "no run matches" in capsys.readouterr().err


def test_runs_vram_folds_samples_and_block_windows_without_writing(tmp_path: Path, capsys) -> None:
    run = tmp_path / "vram-run"
    with JsonlEventSink(run / "events.jsonl", "vram-fixture") as events:
        events.emit(
            "run",
            "info",
            "run.started",
            **{
                "cuda.allocated_bytes": 100,
                "cuda.reserved_bytes": 150,
                "cuda.device_total_bytes": 1_000,
            },
        )
        events.emit(
            "resource",
            "info",
            "resource.sample",
            **{"cuda.allocated_bytes": 300, "cuda.reserved_bytes": 500},
        )
        events.emit(
            "resource",
            "info",
            "resource.sample",
            **{"cuda.allocated_bytes": 250, "cuda.reserved_bytes": 350},
        )
        events.emit(
            "resident",
            "info",
            "block.completed",
            block=2,
            planned_device_bytes=400,
            **{
                "cuda.window_peak_allocated_bytes": 325,
                "cuda.window_peak_reserved_bytes": 500,
            },
        )

    assert main(["runs", "vram", "--path", str(run), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["sample_count"] == 2
    assert summary["peak_reserved_bytes"] == 500
    assert summary["empty_cache_sawtooth_count"] == 1
    assert summary["block_peaks"] == [
        {
            "block": 2,
            "budget_utilization": 1.25,
            "planned_device_bytes": 400,
            "window_peak_allocated_bytes": 325,
            "window_peak_reserved_bytes": 500,
        }
    ]
    assert not (run / "manifest.json").exists()
