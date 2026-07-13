"""Compare two NanoQuant profile artifacts by stable phase path."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class ProfilePhase:
    path: str
    count: int
    wall_seconds: float
    self_seconds: float
    wall_p50: float
    self_p50: float


@dataclass(frozen=True, slots=True)
class LoadedProfile:
    path: Path
    run_id: str
    fingerprint: str
    coverage: float
    wall_total_seconds: float
    phases: dict[str, ProfilePhase]


def load_profile(path: str | Path) -> LoadedProfile:
    source = Path(path)
    payload: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported profile schema: {source}")
    environment = payload.get("environment")
    coverage = payload.get("coverage")
    phase_values = payload.get("phases")
    if not isinstance(environment, dict) or not isinstance(coverage, dict) or not isinstance(phase_values, list):
        raise ValueError(f"malformed profile: {source}")
    phases: dict[str, ProfilePhase] = {}
    for value in phase_values:
        if not isinstance(value, dict):
            raise ValueError(f"malformed phase in profile: {source}")
        phase = ProfilePhase(
            str(value["path"]),
            int(value["count"]),
            float(value["wall_seconds"]),
            float(value["self_seconds"]),
            float(value["p50"]),
            float(value["self_p50"]),
        )
        if phase.path in phases:
            raise ValueError(f"duplicate phase path in profile: {phase.path}")
        phases[phase.path] = phase
    return LoadedProfile(
        source,
        str(payload["run_id"]),
        str(environment.get("runtime_fingerprint", "")),
        float(coverage["fraction"]),
        float(coverage["wall_total_seconds"]),
        phases,
    )


def compare_profiles(
    baseline: LoadedProfile,
    candidate: LoadedProfile,
    *,
    metric: str = "self",
    minimum_seconds: float = 1.0,
    threshold_percent: float = 5.0,
) -> dict[str, object]:
    if metric not in {"self", "wall"}:
        raise ValueError("profile comparison metric must be 'self' or 'wall'")
    if minimum_seconds < 0 or threshold_percent < 0:
        raise ValueError("profile comparison thresholds cannot be negative")
    metric_field = f"{metric}_seconds"
    median_field = f"{metric}_p50"
    comparable_environment = bool(baseline.fingerprint) and baseline.fingerprint == candidate.fingerprint
    rows: list[dict[str, object]] = []
    regressions = 0
    for path in sorted(set(baseline.phases) | set(candidate.phases)):
        baseline_phase = baseline.phases.get(path)
        candidate_phase = candidate.phases.get(path)
        baseline_seconds = 0.0 if baseline_phase is None else float(getattr(baseline_phase, metric_field))
        candidate_seconds = 0.0 if candidate_phase is None else float(getattr(candidate_phase, metric_field))
        delta_seconds = candidate_seconds - baseline_seconds
        delta_percent = None if baseline_seconds == 0 else 100.0 * delta_seconds / baseline_seconds
        baseline_median = 0.0 if baseline_phase is None else float(getattr(baseline_phase, median_field))
        candidate_median = 0.0 if candidate_phase is None else float(getattr(candidate_phase, median_field))
        median_delta_percent = (
            None
            if baseline_median == 0
            else 100.0 * (candidate_median - baseline_median) / baseline_median
        )
        if baseline_phase is None:
            status = "new"
        elif candidate_phase is None:
            status = "missing"
        elif baseline_phase.count != candidate_phase.count:
            status = "count_mismatch"
        elif baseline_seconds < minimum_seconds:
            status = "below_minimum"
        elif (
            delta_percent is not None
            and median_delta_percent is not None
            and delta_percent > threshold_percent
            and median_delta_percent > threshold_percent
        ):
            status = "regression"
            regressions += 1
        elif (
            delta_percent is not None
            and median_delta_percent is not None
            and delta_percent < -threshold_percent
            and median_delta_percent < -threshold_percent
        ):
            status = "improvement"
        elif delta_percent is not None and abs(delta_percent) > threshold_percent:
            status = "noisy"
        else:
            status = "stable"
        rows.append(
            {
                "path": path,
                "baseline_count": 0 if baseline_phase is None else baseline_phase.count,
                "candidate_count": 0 if candidate_phase is None else candidate_phase.count,
                "baseline_seconds": baseline_seconds,
                "candidate_seconds": candidate_seconds,
                "delta_seconds": delta_seconds,
                "delta_percent": delta_percent,
                "baseline_median_seconds": baseline_median,
                "candidate_median_seconds": candidate_median,
                "median_delta_percent": median_delta_percent,
                "status": status,
            }
        )
    return {
        "schema_version": 1,
        "metric": metric,
        "minimum_seconds": minimum_seconds,
        "threshold_percent": threshold_percent,
        "comparable_environment": comparable_environment,
        "baseline": {
            "path": str(baseline.path),
            "run_id": baseline.run_id,
            "coverage": baseline.coverage,
            "wall_total_seconds": baseline.wall_total_seconds,
            "runtime_fingerprint": baseline.fingerprint,
        },
        "candidate": {
            "path": str(candidate.path),
            "run_id": candidate.run_id,
            "coverage": candidate.coverage,
            "wall_total_seconds": candidate.wall_total_seconds,
            "runtime_fingerprint": candidate.fingerprint,
        },
        "regression_count": regressions,
        "actionable_regression_count": regressions if comparable_environment else 0,
        "phases": rows,
    }


def render_markdown(comparison: dict[str, object]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    phases = comparison["phases"]
    if not isinstance(baseline, dict) or not isinstance(candidate, dict) or not isinstance(phases, list):
        raise TypeError("malformed profile comparison")
    comparable = bool(comparison["comparable_environment"])
    rows = [value for value in phases if isinstance(value, dict)]
    rows.sort(key=lambda value: abs(float(value["delta_seconds"])), reverse=True)
    lines = [
        "# Profile comparison",
        "",
        f"- Metric: `{comparison['metric']}_seconds`",
        f"- Environment comparable: **{'yes' if comparable else 'no'}**",
        f"- Baseline coverage: {float(baseline['coverage']):.2%}",
        f"- Candidate coverage: {float(candidate['coverage']):.2%}",
        f"- Observed regressions above threshold: {comparison['regression_count']}",
        f"- Actionable regressions: {comparison['actionable_regression_count']}",
        "",
        "| Phase | Baseline (s) | Candidate (s) | Delta (s) | Delta | Median delta | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for value in rows:
        percent = value["delta_percent"]
        rendered_percent = "n/a" if percent is None else f"{float(percent):+.2f}%"
        median_percent = value["median_delta_percent"]
        rendered_median = "n/a" if median_percent is None else f"{float(median_percent):+.2f}%"
        lines.append(
            f"| `{value['path']}` | {float(value['baseline_seconds']):.6f} | "
            f"{float(value['candidate_seconds']):.6f} | {float(value['delta_seconds']):+.6f} | "
            f"{rendered_percent} | {rendered_median} | {value['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metric", choices=("self", "wall"), default="self")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparison = compare_profiles(
        load_profile(args.baseline),
        load_profile(args.candidate),
        metric=args.metric,
        minimum_seconds=args.min_seconds,
        threshold_percent=args.threshold_pct,
    )
    rendered = (
        json.dumps(comparison, sort_keys=True, indent=2)
        if args.format == "json"
        else render_markdown(comparison)
    )
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
    regressions = cast(int, comparison["actionable_regression_count"])
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
