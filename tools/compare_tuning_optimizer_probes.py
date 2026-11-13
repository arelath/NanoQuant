"""Compare complete paired optimizer replays and enforce a held-out MSE gate."""

import argparse
import json
import math
from pathlib import Path

PROJECTIONS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.o_proj")


def compare_pair(baseline_path, candidate_path, maximum_error_ratio=1.02):
    if not math.isfinite(maximum_error_ratio) or maximum_error_ratio < 1:
        raise ValueError("maximum error ratio must be finite and at least one")
    protocols = [json.loads((path / "protocol.json").read_text()) for path in (baseline_path, candidate_path)]
    results = [json.loads((path / "compiled.json").read_text()) for path in (baseline_path, candidate_path)]
    if any(len(rows) != 1 or rows[0]["kind"] != "factorized" for rows in results):
        raise ValueError("comparison requires one factorized result per probe")
    baseline, candidate = [rows[0] for rows in results]
    for row, protocol, expected in zip(
        (baseline, candidate), protocols, ("parity", "torch-fused-scales-fp32"), strict=True,
    ):
        if row["optimizer"] != expected or protocol["optimizer"] != expected:
            raise ValueError("unexpected optimizer in paired comparison")
    for key in ("snapshot", "block_artifact", "calibration_artifact", "input_shape", "tuning_config",
                "seed", "initialization", "initialization_artifacts", "remaining_projections", "projection"):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError(f"paired protocols differ: {key}")
    for key in ("dataset", "split", "fingerprint", "bos_token_id", "shape", "token_sha256"):
        if protocols[0]["heldout"][key] != protocols[1]["heldout"][key]:
            raise ValueError(f"paired held-out protocols differ: {key}")
    if baseline["initial_evaluated_loss"] != candidate["initial_evaluated_loss"]:
        raise ValueError("paired initial losses differ")
    changes = {}
    for key in ("weighted_mse", "unweighted_mse"):
        old, new = baseline["heldout"][key], candidate["heldout"][key]
        if not all(math.isfinite(value) and value > 0 for value in (old, new)):
            raise ValueError("held-out losses must be finite and positive")
        changes[key] = new / old - 1
    if not all(math.isfinite(row["seconds"]) and row["seconds"] > 0 for row in (baseline, candidate)):
        raise ValueError("timings must be finite and positive")
    old_error, new_error = (row["projection_relative_frobenius_error"] for row in (baseline, candidate))
    if not all(math.isfinite(value) and value > 0 for value in (old_error, new_error)):
        raise ValueError("projection errors must be finite and positive")
    heldout_passes = all(value <= maximum_error_ratio - 1 for value in changes.values())
    return {
        "projection": protocols[0]["projection"], "seed": protocols[0]["seed"],
        "baseline_path": str(baseline_path), "candidate_path": str(candidate_path),
        "baseline": baseline, "candidate": candidate,
        "time_reduction_fraction": 1 - candidate["seconds"] / baseline["seconds"],
        "heldout_error_change_fraction": changes,
        "projection_error_change_fraction": new_error / old_error - 1,
        "passes_heldout_gate": heldout_passes,
        "passes_quality_gate": heldout_passes and new_error <= maximum_error_ratio * old_error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--maximum-error-ratio", type=float, default=1.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [compare_pair(
        args.root / f"optimizer-heldout-{projection}-parity-seed{seed}",
        args.root / f"optimizer-heldout-{projection}-torch-fused-scales-fp32-seed{seed}",
        args.maximum_error_ratio,
    ) for seed in args.seeds for projection in PROJECTIONS]
    result = {
        "maximum_error_ratio": args.maximum_error_ratio,
        "all_pass": all(row["passes_quality_gate"] for row in rows), "comparisons": rows,
        "passed_projections": [projection for projection in PROJECTIONS if all(
            row["passes_quality_gate"] for row in rows if row["projection"] == projection
        )],
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in rows:
        print(row["projection"], "seed", row["seed"],
              "time reduction", f'{100 * row["time_reduction_fraction"]:.2f}%',
              "heldout changes", row["heldout_error_change_fraction"],
              "PASS" if row["passes_quality_gate"] else "FAIL")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
