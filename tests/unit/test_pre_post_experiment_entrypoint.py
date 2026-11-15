from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from recipes import _experiment as entry


@pytest.mark.parametrize("flag", ["--plan-only", "--dry-run"])
def test_preview_adds_pre_kd_plan_without_running_export(monkeypatch, capsys, tmp_path: Path, flag: str) -> None:
    launcher = tmp_path / "experiments" / "077-example.py"
    monkeypatch.setattr(sys, "argv", [str(launcher), flag])
    child = Mock(return_value=SimpleNamespace(stdout='{"export": {"use_active_global_tuning": false}}'))
    monkeypatch.setattr(entry.subprocess, "run", child)

    def runner() -> int:
        print('{"config": {"seed": 0}}')
        return 0

    with pytest.raises(SystemExit) as stopped:
        entry.experiment_pre_post_callable_main("__main__", launcher, runner, uniform_control=True)
    assert stopped.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"] == {"seed": 0}
    assert payload["pre_global_tuning"]["export"]["use_active_global_tuning"] is False
    command = child.call_args.args[0]
    assert command[-2:] == ["--uniform-control", "--plan-only"]
    assert "--refresh-quality" not in command


def test_post_failure_never_starts_pre_export(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["launcher"])
    child = Mock()
    monkeypatch.setattr(entry.subprocess, "run", child)
    with pytest.raises(SystemExit) as stopped:
        entry.experiment_pre_post_callable_main("__main__", tmp_path / "experiments/x.py", lambda: 7)
    assert stopped.value.code == 7
    child.assert_not_called()


@pytest.mark.parametrize("exit_code", [0, 9])
def test_pre_export_runs_only_after_success_and_propagates_failure(monkeypatch, tmp_path: Path, exit_code: int) -> None:
    monkeypatch.setattr(sys, "argv", ["launcher"])
    phases: list[str] = []

    def runner() -> int:
        phases.append("post-kd")
        return 0

    def child(command, **kwargs):
        assert phases == ["post-kd"]
        assert command[0] == sys.executable
        assert command[-2:] == ["--uniform-control", "--refresh-quality"]
        phases.append("pre-kd")
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(entry.subprocess, "run", child)
    with pytest.raises(SystemExit) as stopped:
        entry.experiment_pre_post_callable_main(
            "__main__", tmp_path / "experiments/x.py", runner, uniform_control=True,
        )
    assert stopped.value.code == exit_code
    assert phases == ["post-kd", "pre-kd"]


def test_importing_launcher_does_not_run_any_phase(monkeypatch, tmp_path: Path) -> None:
    runner, child = Mock(), Mock()
    monkeypatch.setattr(entry.subprocess, "run", child)
    entry.experiment_pre_post_callable_main("imported_launcher", tmp_path / "experiments/x.py", runner)
    runner.assert_not_called()
    child.assert_not_called()
