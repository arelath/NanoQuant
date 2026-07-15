from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoquant.application.report import render_run_report, summarize_run_payloads


def _manifest(
    status: str,
    *,
    parent_run_id: str | None = None,
    forked_from_stage: str | None = None,
    failure: dict[str, object] | None = None,
    conclusion: str | None = None,
    recommended_next_action: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "run_fixture",
        "status": status,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:10:00Z",
        "config_hash": "sha256:config",
        "resolved_config": {
            "intent": {
                "experiment_number": 19,
                "name": "summary-fixture",
                "purpose": "exercise terminal summaries",
                "hypothesis": "structured state is sufficient",
                "baseline_run": "baseline",
            }
        },
        "launcher": {
            "kind": "numbered_runfile",
            "experiment_number": 19,
            "repository_relative_path": "experiments/019_summary_fixture.py",
            "content_hash": "sha256:launcher",
            "revision": "abc123",
            "arguments": [],
        },
        "environment": {
            "python": "3.12.0",
            "platform": "fixture-platform",
            "machine": "x86_64",
            "packages": {"torch": "2.fixture"},
            "environment": {"CUDA_VISIBLE_DEVICES": "0"},
        },
        "parent_run_id": parent_run_id,
        "forked_from_stage": forked_from_stage,
        "artifacts": ["sha256-artifact"] if status == "completed" else [],
        "failure": failure,
        "conclusion": conclusion,
        "recommended_next_action": recommended_next_action,
    }


def _event(
    sequence: int,
    name: str,
    severity: str = "info",
    stage: str = "run",
    **fields: object,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "timestamp": f"2026-01-01T00:00:{sequence:02d}Z",
        "run_id": "run_fixture",
        "sequence": sequence,
        "stage": stage,
        "severity": severity,
        "name": name,
        "fields": fields,
    }


@pytest.mark.parametrize(
    ("status", "terminal", "severity", "failure"),
    [
        ("completed", "run.completed", "info", None),
        ("failed", "run.failed", "error", {"type": "RuntimeError", "message": "boom"}),
        ("interrupted", "run.interrupted", "warning", None),
    ],
)
def test_terminal_run_summaries_cover_completed_failed_and_interrupted(
    tmp_path: Path,
    status: str,
    terminal: str,
    severity: str,
    failure: dict[str, object] | None,
) -> None:
    manifest = _manifest(status, failure=failure)
    events = (_event(1, "run.started"), _event(2, terminal, severity, reason=status))
    summary = summarize_run_payloads(
        manifest,
        events,
    )

    assert summary.status == status
    assert summary.terminal_event == terminal
    assert summary.attempt_count == 1
    assert summary.resume_count == 0
    assert summary.consistency_warnings == ()
    assert summary.warning_count == (1 if severity == "warning" else 0)
    assert summary.error_count == (1 if severity == "error" else 0)
    assert bool(summary.failure) == (failure is not None)
    assert summary.experiment_number == 19
    assert summary.zero_argument_runfile
    assert summary.launcher_path == "experiments/019_summary_fixture.py"
    assert summary.launcher_content_hash == "sha256:launcher"
    assert summary.cost.elapsed_seconds == 600.0
    assert dict(summary.environment)["machine"] == "x86_64"

    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    report = render_run_report(tmp_path)
    assert f"Status: `{status}`" in report
    assert f"Terminal event: `{terminal}`" in report
    assert "## Outcome" in report
    assert "## Launcher" in report
    assert "Zero-argument numbered runfile: **yes**" in report
    assert "`experiments/019_summary_fixture.py`" in report
    assert "`sha256:launcher`" in report
    assert "## Environment" in report
    assert '"CUDA_VISIBLE_DEVICES": "0"' in report
    assert "## Cost" in report
    assert "Manifest elapsed seconds: 600.0" in report
    assert "Recommended next action:" in report
    assert f'"reason": "{status}"' in report
    if failure is not None:
        assert '"message": "boom"' in report


def test_resumed_summary_preserves_attempt_history_and_final_outcome(tmp_path: Path) -> None:
    events = (
        _event(1, "run.started"),
        _event(2, "run.interrupted", "warning", reason="keyboard"),
        _event(3, "run.resumed", stored_config_hash="sha256:old"),
        _event(4, "run.completed", artifact_count=1),
    )
    manifest = _manifest("completed")
    summary = summarize_run_payloads(manifest, events)

    assert summary.resumed
    assert summary.attempt_count == 2
    assert summary.resume_count == 1
    assert summary.lifecycle_events == (
        "run.started",
        "run.interrupted",
        "run.resumed",
        "run.completed",
    )
    assert summary.terminal_event == "run.completed"

    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    report = render_run_report(tmp_path)
    assert "Status: `completed`" in report
    assert "Resume attempts: 1" in report
    assert "run.interrupted, run.resumed, run.completed" in report
    assert "`sha256-artifact`" in report


def test_forked_summary_preserves_parent_and_invalidation_boundary(tmp_path: Path) -> None:
    manifest = _manifest(
        "completed",
        parent_run_id="run_parent",
        forked_from_stage="quantize",
    )
    events = (_event(1, "run.started"), _event(2, "run.completed"))
    summary = summarize_run_payloads(manifest, events)

    assert summary.forked
    assert summary.parent_run_id == "run_parent"
    assert summary.forked_from_stage == "quantize"

    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    report = render_run_report(tmp_path)
    assert "Parent run: `run_parent`" in report
    assert "Forked from stage: `quantize`" in report


def test_summary_reports_lifecycle_and_event_integrity_mismatches() -> None:
    events = (
        _event(2, "run.started"),
        {**_event(1, "run.failed", "error", code="RUN999"), "run_id": "foreign"},
    )
    summary = summarize_run_payloads(_manifest("completed"), events)

    assert summary.terminal_event == "run.failed"
    assert summary.issues[0].code == "RUN999"
    assert summary.consistency_warnings == (
        "event sequences are not strictly increasing",
        "event stream contains 1 foreign run IDs",
        "manifest status completed does not match terminal event run.failed",
    )


def test_summary_preserves_explicit_conclusion_action_and_structured_cost(tmp_path: Path) -> None:
    manifest = _manifest(
        "completed",
        conclusion="Candidate satisfies the quick promotion gate.",
        recommended_next_action="Run the full evaluation tier.",
    )
    events = (
        _event(1, "run.started"),
        _event(
            2,
            "quantization.completed",
            stage="quantization",
            wall_seconds=12.5,
            factorization_wall_seconds=10.0,
            gpu_peak_bytes=7_000,
            host_peak_bytes=8_000,
            temporary_disk_bytes=9_000,
        ),
        _event(3, "run.completed"),
    )
    summary = summarize_run_payloads(manifest, events)

    assert summary.conclusion == "Candidate satisfies the quick promotion gate."
    assert summary.recommended_next_action == "Run the full evaluation tier."
    assert summary.cost.peak_device_bytes == 7_000
    assert summary.cost.peak_host_bytes == 8_000
    assert summary.cost.peak_temporary_disk_bytes == 9_000
    assert [(item.metric, item.value) for item in summary.cost.observations] == [
        ("factorization_wall_seconds", 10.0),
        ("gpu_peak_bytes", 7_000),
        ("host_peak_bytes", 8_000),
        ("temporary_disk_bytes", 9_000),
        ("wall_seconds", 12.5),
    ]

    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    report = render_run_report(tmp_path)
    assert "Conclusion: Candidate satisfies the quick promotion gate." in report
    assert "Recommended next action: Run the full evaluation tier." in report
    assert "| `quantization` | `quantization.completed` | `gpu_peak_bytes` | 7000 | bytes |" in report


def test_summary_flags_numbered_runfile_arguments_and_experiment_mismatch() -> None:
    manifest = _manifest("completed")
    launcher = dict(manifest["launcher"])
    launcher["experiment_number"] = 20
    launcher["arguments"] = ["--rank", "32"]
    manifest["launcher"] = launcher

    summary = summarize_run_payloads(
        manifest,
        (_event(1, "run.started"), _event(2, "run.completed")),
    )

    assert not summary.zero_argument_runfile
    assert summary.experiment_number == 20
    assert summary.consistency_warnings == (
        "intent experiment 19 differs from launcher experiment 20",
        "numbered runfile recorded experiment-defining arguments",
    )
