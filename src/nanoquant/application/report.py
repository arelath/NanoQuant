"""Reports rendered exclusively from structured manifests and events."""

from __future__ import annotations

import json
from pathlib import Path


def render_run_report(run_directory: str | Path) -> str:
    root = Path(run_directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    intent = manifest["resolved_config"]["intent"]
    warnings = [event for event in events if event["severity"] == "warning"]
    failures = [event for event in events if event["severity"] == "error"]
    lines = [
        f"# Run {manifest['run_id']}",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Experiment: `{intent['experiment_number']}` — {intent['name']}",
        f"- Purpose: {intent['purpose'] or 'Not provided'}",
        f"- Hypothesis: {intent['hypothesis'] or 'Not provided'}",
        f"- Baseline: `{intent['baseline_run'] or 'none'}`",
        f"- Config hash: `{manifest['config_hash']}`",
        "",
        "## Execution",
        "",
        f"Recorded {len(events)} structured events, {len(warnings)} warnings, and {len(failures)} errors.",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{artifact}`" for artifact in manifest.get("artifacts", []))
    if not manifest.get("artifacts"):
        lines.append("No committed artifacts.")
    return "\n".join(lines) + "\n"
