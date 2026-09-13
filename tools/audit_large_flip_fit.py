"""Read-only float64 audit of the best completed 5% perturbation attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_hour_basin_search import neighborhood_audit
from probe_hour_factor_fit import score, validate_state
from safetensors.torch import load_file

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file


def audit(source: Path, device: str):
    report = json.loads((source / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("search must finish before the independent audit")
    attempt = min((a for a in report["attempts"] if a.get("campaign") == "large-5pct"),
                  key=lambda a: a["final_error"])
    row = next(r for r in report["records"] if r["name"] == attempt["final"])
    if not row["within_budget"] or hash_file(source / row["state_file"]) != row["state_sha256"]:
        raise ValueError("selected completed checkpoint is invalid")
    fixture_path = Path(report["source"]) / "fixture.safetensors"
    if hash_file(fixture_path) != report["inputs"]["fixture.safetensors"]:
        raise ValueError("fixture hash mismatch")
    torch.set_num_threads(4)
    with acquire_device_lease(device):
        fixture = load_file(str(fixture_path), device=device)
        state = load_file(str(source / row["state_file"]), device=device)
        validate_state(state, tuple(fixture["target"].shape), state["scale_mid"].numel())
        metrics = score(fixture, state, double=True)
        result = {"state_file": row["state_file"], "state_sha256": row["state_sha256"],
                  "fixture_sha256": report["inputs"]["fixture.safetensors"],
                  "search_report_sha256": hash_file(source / "report.json"),
                  "audit_runner_sha256": hash_file(Path(__file__)),
                  "metrics": metrics, "neighborhood": neighborhood_audit(fixture, state),
                  "energy_gap_above_overall_best_percent": 100 * (
                      metrics["weighted_error"] / report["comparison"]["best"]["weighted_error"] - 1),
                  "selection_fp32_relative_error_delta": abs(
                      metrics["weighted_error"] - row["weighted_error"]) / metrics["weighted_error"]}
        atomic_write_json(source / "large-flip-audit.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("evidence/067/basin-search"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    audit(args.source.resolve(), args.device)
