from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.compare_profiles import compare_profiles, load_profile, render_markdown


def _write_profile(
    path: Path,
    *,
    fingerprint: str = "same",
    coverage: float = 0.95,
    phases: tuple[tuple[str, int, float, float], ...],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": path.stem,
                "environment": {"runtime_fingerprint": fingerprint},
                "coverage": {"fraction": coverage, "wall_total_seconds": 10.0},
                "phases": [
                    {
                        "path": name,
                        "count": count,
                        "wall_seconds": wall,
                        "self_seconds": self_seconds,
                        "p50": wall / count,
                        "self_p50": self_seconds / count,
                    }
                    for name, count, wall, self_seconds in phases
                ],
            }
        ),
        encoding="utf-8",
    )


def test_compare_profiles_classifies_stable_improvement_regression_and_protocol_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_profile(
        baseline_path,
        phases=(("run/a", 2, 10.0, 8.0), ("run/b", 1, 4.0, 2.0), ("run/c", 1, 0.5, 0.5)),
    )
    _write_profile(
        candidate_path,
        phases=(
            ("run/a", 2, 8.0, 6.0),
            ("run/b", 1, 5.0, 2.5),
            ("run/c", 2, 0.6, 0.6),
            ("run/new", 1, 1.0, 1.0),
        ),
    )

    result = compare_profiles(
        load_profile(baseline_path),
        load_profile(candidate_path),
        minimum_seconds=1.0,
        threshold_percent=10.0,
    )
    phases = {str(value["path"]): value for value in result["phases"] if isinstance(value, dict)}

    assert result["comparable_environment"] is True
    assert result["regression_count"] == 1
    assert phases["run/a"]["status"] == "improvement"
    assert phases["run/b"]["status"] == "regression"
    assert phases["run/c"]["status"] == "count_mismatch"
    assert phases["run/new"]["status"] == "new"
    assert result["actionable_regression_count"] == 1
    assert "Observed regressions above threshold: 1" in render_markdown(result)


def test_environment_mismatch_is_informational_and_bad_schema_is_rejected(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_profile(baseline_path, fingerprint="one", phases=(("run/a", 1, 2.0, 2.0),))
    _write_profile(candidate_path, fingerprint="two", phases=(("run/a", 1, 4.0, 4.0),))
    result = compare_profiles(load_profile(baseline_path), load_profile(candidate_path))
    assert result["comparable_environment"] is False
    assert result["regression_count"] == 1
    assert result["actionable_regression_count"] == 0

    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_profile(bad)


def test_duplicate_phase_paths_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    _write_profile(path, phases=(("run/a", 1, 1.0, 1.0), ("run/a", 1, 1.0, 1.0)))
    with pytest.raises(ValueError, match="duplicate"):
        load_profile(path)


def test_total_delta_without_median_confirmation_is_classified_as_noise(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_profile(baseline_path, phases=(("run/a", 2, 10.0, 8.0),))
    _write_profile(candidate_path, phases=(("run/a", 2, 12.0, 9.0),))
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_payload["phases"][0]["self_p50"] = baseline_payload["phases"][0]["self_p50"]
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

    result = compare_profiles(
        load_profile(baseline_path),
        load_profile(candidate_path),
        minimum_seconds=1.0,
        threshold_percent=10.0,
    )
    phase = next(value for value in result["phases"] if value["path"] == "run/a")

    assert phase["status"] == "noisy"
    assert result["regression_count"] == 0
