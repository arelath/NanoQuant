"""Run summaries rendered exclusively from structured manifests and events."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_LIFECYCLE_EVENTS = {
    "run.started",
    "run.resumed",
    "run.completed",
    "run.failed",
    "run.interrupted",
}
_EXPECTED_TERMINAL_EVENT = {
    "completed": "run.completed",
    "failed": "run.failed",
    "interrupted": "run.interrupted",
}


@dataclass(frozen=True, slots=True)
class RunIssue:
    sequence: int
    severity: str
    stage: str
    name: str
    code: str | None
    fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class RunCostObservation:
    sequence: int
    stage: str
    event: str
    metric: str
    value: int | float
    unit: str


@dataclass(frozen=True, slots=True)
class RunCost:
    elapsed_seconds: float | None
    observations: tuple[RunCostObservation, ...]
    peak_device_bytes: int | None
    peak_host_bytes: int | None
    peak_temporary_disk_bytes: int | None


@dataclass(frozen=True, slots=True)
class _SummaryEvent:
    run_id: str
    timestamp: str
    sequence: int
    stage: str
    severity: str
    name: str
    fields: dict[str, object]


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    status: str
    created_at: str
    updated_at: str
    config_hash: str
    experiment_number: int | None
    run_name: str
    purpose: str
    hypothesis: str
    baseline_run: str | None
    launcher_kind: str
    launcher_path: str | None
    launcher_content_hash: str
    launcher_revision: str | None
    launcher_arguments: tuple[str, ...]
    environment: tuple[tuple[str, object], ...]
    cost: RunCost
    event_count: int
    warning_count: int
    error_count: int
    stage_event_counts: tuple[tuple[str, int], ...]
    attempt_count: int
    resume_count: int
    lifecycle_events: tuple[str, ...]
    terminal_event: str | None
    terminal_fields: tuple[tuple[str, object], ...]
    parent_run_id: str | None
    forked_from_stage: str | None
    artifacts: tuple[str, ...]
    failure: tuple[tuple[str, object], ...]
    issues: tuple[RunIssue, ...]
    conclusion: str
    recommended_next_action: str
    consistency_warnings: tuple[str, ...]

    @property
    def resumed(self) -> bool:
        return self.resume_count > 0

    @property
    def forked(self) -> bool:
        return self.parent_run_id is not None

    @property
    def zero_argument_runfile(self) -> bool:
        return self.launcher_kind == "numbered_runfile" and not self.launcher_arguments


def _object(payload: object, path: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object")
    return {str(key): value for key, value in payload.items()}


def _string(payload: object, path: str) -> str:
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"{path} must be a non-empty string")
    return payload


def _optional_string(payload: object, path: str) -> str | None:
    if payload is None:
        return None
    return _string(payload, path)


def _optional_int(payload: object, path: str) -> int | None:
    if payload is None:
        return None
    if type(payload) is not int or payload < 0:
        raise ValueError(f"{path} must be a non-negative integer or null")
    return payload


def _string_list(payload: object, path: str) -> tuple[str, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be an array")
    return tuple(_string(value, f"{path}[{index}]") for index, value in enumerate(payload))


def _elapsed_seconds(created_at: str, updated_at: str, warnings: list[str]) -> float | None:
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        elapsed = (updated - created).total_seconds()
    except (TypeError, ValueError):
        warnings.append("manifest timestamps cannot be converted to elapsed run cost")
        return None
    if elapsed < 0:
        warnings.append("manifest updated_at precedes created_at")
        return None
    return elapsed


def _cost_observations(
    events: tuple[_SummaryEvent, ...], elapsed_seconds: float | None, warnings: list[str]
) -> RunCost:
    observations: list[RunCostObservation] = []
    device_peaks: list[int] = []
    host_peaks: list[int] = []
    disk_peaks: list[int] = []
    for event in events:
        for metric, value in sorted(event.fields.items()):
            is_seconds = metric.endswith("_seconds")
            is_cost_bytes = metric.endswith("_bytes") and any(
                marker in metric
                for marker in (
                    "peak",
                    "temporary_disk",
                    "workspace",
                    "allocated",
                    "reserved",
                    "working_set",
                    "private",
                )
            )
            if not is_seconds and not is_cost_bytes:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                warnings.append(f"event {event.sequence} has invalid cost metric {metric}")
                continue
            unit = "seconds" if is_seconds else "bytes"
            if unit == "bytes" and type(value) is not int:
                warnings.append(f"event {event.sequence} cost metric {metric} is not an integer byte count")
                continue
            observations.append(RunCostObservation(event.sequence, event.stage, event.name, metric, value, unit))
            if unit == "bytes":
                integer = int(value)
                if "disk" in metric:
                    disk_peaks.append(integer)
                elif any(marker in metric for marker in ("host", "cpu", "working_set", "private")):
                    host_peaks.append(integer)
                elif any(
                    marker in metric
                    for marker in (
                        "gpu",
                        "device",
                        "cuda",
                        "allocated",
                        "reserved",
                        "workspace",
                    )
                ):
                    device_peaks.append(integer)
    return RunCost(
        elapsed_seconds,
        tuple(observations),
        max(device_peaks, default=None),
        max(host_peaks, default=None),
        max(disk_peaks, default=None),
    )


def _explicit_text(manifest: dict[str, object], terminal: _SummaryEvent | None, field: str) -> str | None:
    value = manifest.get(field)
    if value is None and terminal is not None:
        value = terminal.fields.get(field)
    return _optional_string(value, f"manifest/event.{field}")


def _default_conclusion(
    status: str,
    warning_count: int,
    failure: tuple[tuple[str, object], ...],
) -> str:
    if status == "completed":
        suffix = f" with {warning_count} warning events" if warning_count else " successfully"
        return f"Run completed{suffix}."
    if status == "failed":
        details = dict(failure)
        kind = details.get("type", "unknown failure")
        message = details.get("message")
        return f"Run failed: {kind}{f': {message}' if message else ''}."
    if status == "interrupted":
        return "Run was interrupted before completion."
    if status == "running":
        return "Run is still in progress."
    return "Run has been created but has not started."


def _default_next_action(status: str, warning_count: int) -> str:
    if status == "completed" and warning_count:
        return "Review warning diagnostics before applying promotion gates."
    if status == "completed":
        return "Compare results with the named baseline and apply the predefined promotion gates."
    if status == "failed":
        return "Diagnose the recorded failure, then fork or rerun from the earliest invalid stage."
    if status == "interrupted":
        return "Validate committed state and resume from the latest durable checkpoint."
    if status == "running":
        return "Continue monitoring structured progress and resource events."
    return "Start the run after validating its resolved configuration and resource plan."


def _event(payload: object, index: int) -> _SummaryEvent:
    value = _object(payload, f"events[{index}]")
    sequence = value.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise ValueError(f"events[{index}].sequence must be a non-negative integer")
    return _SummaryEvent(
        _string(value.get("run_id"), f"events[{index}].run_id"),
        _string(value.get("timestamp"), f"events[{index}].timestamp"),
        sequence,
        _string(value.get("stage"), f"events[{index}].stage"),
        _string(value.get("severity"), f"events[{index}].severity"),
        _string(value.get("name"), f"events[{index}].name"),
        _object(value.get("fields"), f"events[{index}].fields"),
    )


def summarize_run_payloads(manifest_payload: object, event_payloads: tuple[object, ...]) -> RunSummary:
    manifest = _object(manifest_payload, "manifest")
    run_id = _string(manifest.get("run_id"), "manifest.run_id")
    status = _string(manifest.get("status"), "manifest.status")
    if status not in {"created", "running", "completed", "failed", "interrupted"}:
        raise ValueError(f"manifest.status is unsupported: {status}")
    created_at = _string(manifest.get("created_at"), "manifest.created_at")
    updated_at = _string(manifest.get("updated_at"), "manifest.updated_at")
    resolved = _object(manifest.get("resolved_config"), "manifest.resolved_config")
    intent = _object(resolved.get("intent", {}), "manifest.resolved_config.intent")
    intent_experiment = _optional_int(
        intent.get("experiment_number"), "manifest.resolved_config.intent.experiment_number"
    )
    run_name = str(intent.get("name") or "unnamed")
    purpose = str(intent.get("purpose") or "Not provided")
    hypothesis = str(intent.get("hypothesis") or "Not provided")
    baseline_run = _optional_string(intent.get("baseline_run"), "manifest.resolved_config.intent.baseline_run")
    launcher = _object(manifest.get("launcher"), "manifest.launcher")
    launcher_kind = _string(launcher.get("kind"), "manifest.launcher.kind")
    launcher_experiment = _optional_int(launcher.get("experiment_number"), "manifest.launcher.experiment_number")
    launcher_path = _optional_string(
        launcher.get("repository_relative_path"), "manifest.launcher.repository_relative_path"
    )
    launcher_content_hash = _string(launcher.get("content_hash"), "manifest.launcher.content_hash")
    launcher_revision = _optional_string(launcher.get("revision"), "manifest.launcher.revision")
    launcher_arguments = _string_list(launcher.get("arguments", []), "manifest.launcher.arguments")
    environment = tuple(sorted(_object(manifest.get("environment"), "manifest.environment").items()))
    events = tuple(_event(payload, index) for index, payload in enumerate(event_payloads))
    warnings: list[str] = []
    if intent_experiment is not None and launcher_experiment is not None and intent_experiment != launcher_experiment:
        warnings.append(f"intent experiment {intent_experiment} differs from launcher experiment {launcher_experiment}")
    if launcher_kind == "numbered_runfile" and launcher_arguments:
        warnings.append("numbered runfile recorded experiment-defining arguments")
    if launcher_kind == "numbered_runfile" and launcher_path is None:
        warnings.append("numbered runfile has no repository-relative launcher path")
    sequences = tuple(event.sequence for event in events)
    if any(right <= left for left, right in zip(sequences, sequences[1:], strict=False)):
        warnings.append("event sequences are not strictly increasing")
    foreign = tuple(event.run_id for event in events if event.run_id != run_id)
    if foreign:
        warnings.append(f"event stream contains {len(foreign)} foreign run IDs")

    lifecycle = tuple(event.name for event in events if event.name in _LIFECYCLE_EVENTS)
    terminal_candidates = tuple(
        event for event in events if event.name in {"run.completed", "run.failed", "run.interrupted"}
    )
    terminal = terminal_candidates[-1] if terminal_candidates else None
    terminal_name = None if terminal is None else terminal.name
    expected_terminal = _EXPECTED_TERMINAL_EVENT.get(status)
    if expected_terminal is not None and terminal_name != expected_terminal:
        warnings.append(f"manifest status {status} does not match terminal event {terminal_name or 'missing'}")
    if status in {"created", "running"} and terminal_name is not None:
        warnings.append(f"non-terminal manifest status {status} has terminal event {terminal_name}")

    issues = tuple(
        RunIssue(
            event.sequence,
            event.severity,
            event.stage,
            event.name,
            str(event.fields.get("code")) if event.fields.get("code") is not None else None,
            tuple(sorted(event.fields.items())),
        )
        for event in events
        if event.severity in {"warning", "error"}
    )
    artifacts_payload = manifest.get("artifacts", [])
    if not isinstance(artifacts_payload, list):
        raise ValueError("manifest.artifacts must be a list")
    artifacts = tuple(_string(value, f"manifest.artifacts[{index}]") for index, value in enumerate(artifacts_payload))
    failure_payload = manifest.get("failure")
    failure = () if failure_payload is None else tuple(sorted(_object(failure_payload, "manifest.failure").items()))
    cost = _cost_observations(events, _elapsed_seconds(created_at, updated_at, warnings), warnings)
    stage_counts = Counter(event.stage for event in events)
    resume_count = lifecycle.count("run.resumed")
    attempt_count = lifecycle.count("run.started") + resume_count
    warning_count = sum(event.severity == "warning" for event in events)
    conclusion = _explicit_text(manifest, terminal, "conclusion") or _default_conclusion(status, warning_count, failure)
    recommended_next_action = _explicit_text(manifest, terminal, "recommended_next_action") or _default_next_action(
        status, warning_count
    )
    return RunSummary(
        run_id,
        status,
        created_at,
        updated_at,
        _string(manifest.get("config_hash"), "manifest.config_hash"),
        launcher_experiment if launcher_experiment is not None else intent_experiment,
        run_name,
        purpose,
        hypothesis,
        baseline_run,
        launcher_kind,
        launcher_path,
        launcher_content_hash,
        launcher_revision,
        launcher_arguments,
        environment,
        cost,
        len(events),
        warning_count,
        sum(event.severity == "error" for event in events),
        tuple(sorted(stage_counts.items())),
        attempt_count,
        resume_count,
        lifecycle,
        terminal_name,
        () if terminal is None else tuple(sorted(terminal.fields.items())),
        _optional_string(manifest.get("parent_run_id"), "manifest.parent_run_id"),
        _optional_string(manifest.get("forked_from_stage"), "manifest.forked_from_stage"),
        artifacts,
        failure,
        issues,
        conclusion,
        recommended_next_action,
        tuple(warnings),
    )


def summarize_run(run_directory: str | Path) -> tuple[RunSummary, dict[str, object]]:
    root = Path(run_directory)
    manifest = _object(json.loads((root / "manifest.json").read_text(encoding="utf-8")), "manifest")
    event_payloads = tuple(
        json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if line
    )
    return summarize_run_payloads(manifest, event_payloads), manifest


def _render_fields(fields: tuple[tuple[str, object], ...]) -> str:
    return json.dumps(dict(fields), sort_keys=True, ensure_ascii=False)


def render_run_report(run_directory: str | Path) -> str:
    summary, _ = summarize_run(run_directory)
    experiment = summary.experiment_number if summary.experiment_number is not None else "none"
    elapsed = summary.cost.elapsed_seconds if summary.cost.elapsed_seconds is not None else "n/a"
    device_peak = summary.cost.peak_device_bytes if summary.cost.peak_device_bytes is not None else "n/a"
    host_peak = summary.cost.peak_host_bytes if summary.cost.peak_host_bytes is not None else "n/a"
    disk_peak = summary.cost.peak_temporary_disk_bytes if summary.cost.peak_temporary_disk_bytes is not None else "n/a"
    lines = [
        f"# Run {summary.run_id}",
        "",
        f"- Status: `{summary.status}`",
        f"- Experiment: `{experiment}` — {summary.run_name}",
        f"- Purpose: {summary.purpose}",
        f"- Hypothesis: {summary.hypothesis}",
        f"- Baseline: `{summary.baseline_run or 'none'}`",
        f"- Config hash: `{summary.config_hash}`",
        "",
        "## Outcome",
        "",
        f"- Conclusion: {summary.conclusion}",
        f"- Recommended next action: {summary.recommended_next_action}",
        "",
        "## Launcher",
        "",
        f"- Kind: `{summary.launcher_kind}`",
        f"- Zero-argument numbered runfile: **{'yes' if summary.zero_argument_runfile else 'no'}**",
        f"- Repository-relative path: `{summary.launcher_path or 'unavailable'}`",
        f"- Content hash: `{summary.launcher_content_hash}`",
        f"- Code revision: `{summary.launcher_revision or 'unavailable'}`",
        f"- Arguments: `{json.dumps(summary.launcher_arguments)}`",
        "",
        "## Execution",
        "",
        f"- Created: `{summary.created_at}`",
        f"- Updated: `{summary.updated_at}`",
        f"- Attempts: {summary.attempt_count}",
        f"- Resume attempts: {summary.resume_count}",
        f"- Events: {summary.event_count} ({summary.warning_count} warnings, {summary.error_count} errors)",
        f"- Lifecycle: `{', '.join(summary.lifecycle_events) or 'none'}`",
        f"- Terminal event: `{summary.terminal_event or 'none'}`",
    ]
    if summary.terminal_fields:
        lines.append(f"- Terminal context: `{_render_fields(summary.terminal_fields)}`")
    if summary.failure:
        lines.append(f"- Failure: `{_render_fields(summary.failure)}`")
    lines.extend(
        (
            "",
            "## Environment",
            "",
            "```json",
            json.dumps(dict(summary.environment), sort_keys=True, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Cost",
            "",
            f"- Manifest elapsed seconds: {elapsed}",
            f"- Peak device bytes: {device_peak}",
            f"- Peak host bytes: {host_peak}",
            f"- Peak temporary disk bytes: {disk_peak}",
            "",
        )
    )
    if summary.cost.observations:
        lines.extend(
            (
                "| Sequence | Stage | Event | Metric | Value | Unit |",
                "| ---: | --- | --- | --- | ---: | --- |",
            )
        )
        lines.extend(
            f"| {item.sequence} | `{item.stage}` | `{item.event}` | `{item.metric}` | {item.value} | {item.unit} |"
            for item in summary.cost.observations
        )
    else:
        lines.append("No structured cost observations were emitted.")
    lines.extend(("", "## Lineage", ""))
    if summary.forked:
        lines.extend(
            (
                f"- Parent run: `{summary.parent_run_id}`",
                f"- Forked from stage: `{summary.forked_from_stage or 'unknown'}`",
            )
        )
    else:
        lines.append("No parent run; this is a root run.")
    lines.extend(("", "## Issues", ""))
    if summary.issues:
        lines.extend(
            f"- `{issue.severity}` `{issue.code or 'no-code'}` `{issue.stage}` `{issue.name}` "
            f"{_render_fields(issue.fields)}"
            for issue in summary.issues
        )
    else:
        lines.append("No warning or error events.")
    if summary.consistency_warnings:
        lines.extend(("", "### Summary consistency warnings", ""))
        lines.extend(f"- {warning}" for warning in summary.consistency_warnings)
    lines.extend(("", "## Artifacts", ""))
    lines.extend(f"- `{artifact}`" for artifact in summary.artifacts)
    if not summary.artifacts:
        lines.append("No committed artifacts.")
    return "\n".join(lines) + "\n"
