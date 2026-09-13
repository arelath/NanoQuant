"""Close a stopped Experiment 068 with read-only CPU scoring, without more search."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import _paths  # noqa: E402, F401
import numpy as np  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def finalize(source):
    source = source.resolve()
    cancellation = json.loads((source / "cancellation.json").read_text(encoding="utf-8-sig"))
    report = json.loads((source / "report.json").read_text())
    if report["status"] not in {"searching", "cancelled"}:
        raise ValueError("only an explicitly cancelled search can be closed")
    snapshots = report.setdefault("provenance_snapshots", {})
    for path, expected in {**report["inputs"], **report["provenance"]}.items():
        checked = Path(path)
        if checked.name == "probe_hour_admm_perturb.py" and (source / "search-runner.py").exists():
            checked = source / "search-runner.py"
            snapshots[path] = "search-runner.py"
        if hash_file(checked) != expected:
            raise ValueError(f"source hash mismatch: {path}")
    for row in report["records"]:
        for filename, expected in ((row["state_file"], row["state_sha256"]),
                                   (row["initial_file"], row["initial_sha256"])):
            if hash_file(source / filename) != expected:
                raise ValueError("checkpoint hash mismatch")
    fixture_name = next(k for k in report["inputs"] if Path(k).name == "fixture.safetensors")
    fixture = load_file(fixture_name)
    target = fixture["target"].astype(np.float64)
    weights = (fixture["output_importance"].astype(np.float64)[:, None]
               * fixture["input_importance"].astype(np.float64)[None, :])
    target_energy = float((np.square(target) * weights).sum())
    raw_energy = float(np.square(target).sum())

    def score(row):
        if hash_file(source / row["state_file"]) != row["state_sha256"]:
            raise ValueError("comparison hash mismatch")
        state = load_file(str(source / row["state_file"]))
        left = state["left_binary"].astype(np.float64) * state["scale_post"].astype(np.float64)[:, None]
        right = state["right_binary"].astype(np.float64) * state["scale_mid"].astype(np.float64)[:, None]
        right *= state["scale_pre"].astype(np.float64)[None, :]
        difference = left @ right - target
        error = float((np.square(difference) * weights).sum())
        raw = float(np.square(difference).sum())
        return {"weighted_error": error, "weighted_nmse": error / target_energy,
                "weighted_nrmse": (error / target_energy) ** 0.5, "weighted_target_energy": target_energy,
                "raw_error": raw, "raw_nrmse": (raw / raw_energy) ** 0.5,
                "state_file": row["state_file"], "state_sha256": row["state_sha256"]}

    baseline = {"state_file": "baseline.safetensors", "state_sha256": next(
        v for k, v in report["inputs"].items() if Path(k).name == "baseline.safetensors")}
    comparators = {"baseline": baseline, **{k: report[k] for k in ("inherited", "control", "best_perturbed", "best")}}
    report["comparison"] = {k: score(row) for k, row in comparators.items()}
    report["status"] = "cancelled"
    report["cancellation"] = cancellation
    report["search_elapsed_seconds"] = datetime.fromisoformat(cancellation["requested_utc"]).timestamp()
    report["search_elapsed_seconds"] -= report["started_utc"]
    report["last_completed_seconds"] = report["records"][-1]["elapsed_seconds"]
    if report.get("pending"):
        report["interrupted"] = report.pop("pending")
    report["neighborhood_audit"] = {}
    report["audit_scope"] = "CPU FP64 comparison and all completed checkpoint hashes; no new single-bit audit"
    report["peak_cuda_allocated"] = None
    report["peak_cuda_reserved"] = None
    report["cuda_memory_scope"] = "Allocator peaks unavailable because the user stopped the worker"
    report["closeout_source_sha256"] = hash_file(Path(__file__))
    report["all_completed_checkpoint_hashes_verified"] = True
    shutil.copy2(source / report["best"]["state_file"], source / "best.safetensors")
    report["best_sha256"] = hash_file(source / "best.safetensors")
    report["saved_state_bytes"] = sum(p.stat().st_size for p in source.glob("*.safetensors"))
    atomic_write_json(source / "report.json", report)
    print(json.dumps({"status": report["status"], "search_seconds": report["search_elapsed_seconds"],
                      "comparison": report["comparison"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "evidence/068/admm-perturb")
    finalize(parser.parse_args().source)
