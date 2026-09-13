"""Experiment 068: ten-percent sign perturbations and candidate-initialized ADMM.

The research adapter reuses production solve/projection kernels and mirrors its
iteration/export order. Production entry points and random initialization stay
unchanged. CPU parity tests bind the mirrored path to the production algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_hour_basin_search import neighborhood_audit
from probe_hour_factor_fit import STATE_KEYS, BudgetExpired, save_state, score, validate_state
from safetensors.torch import load_file

from nanoquant.domain.factorization import _rank_one_sign_projection, _solve, _svid_components, cubic_schedule
from nanoquant.domain.functional_binary_population import canonical_binary_hash
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

ROOT = Path(__file__).resolve().parents[1]


def lift_candidate(fixture, state, dtype=torch.float32):
    """Lift an exported fit into weighted projected factors with unit left columns.

ADMM exports (L / column_norm(L)) @ R in weighted coordinates. Thus the
column norm removed from L must be placed into R when lifting an existing fit.
Duals are reset separately; this is a fresh warm start, not resumed ADMM state.
"""
    rank = state["left_binary"].shape[1]
    validate_state(state, tuple(fixture["target"].shape), rank)
    wi = fixture["input_importance"].float().sqrt().clamp_min(1e-12)
    wo = fixture["output_importance"].float().sqrt().clamp_min(1e-12)
    left = state["left_binary"].float() * (state["scale_post"].float() * wo)[:, None]
    norms = left.norm(dim=0).clamp_min(1e-12)
    right = state["right_binary"].float() * (state["scale_mid"].float() * norms)[:, None]
    right = right * (state["scale_pre"].float() * wi)[None, :]
    return (left / norms).to(dtype), right.to(dtype)


def perturb(state, seed, probability=0.10):
    generator = torch.Generator(device=state["left_binary"].device).manual_seed(seed)
    candidate = {k: v.clone() for k, v in state.items()}
    counts = {}
    for key in STATE_KEYS[:2]:
        mask = torch.rand(candidate[key].shape, generator=generator, device=candidate[key].device) < probability
        candidate[key][mask] *= -1
        counts[key] = int(mask.sum())
    return candidate, counts


def admm_fit(fixture, rank, seed, *, initial=None, iterations=800, dtype=torch.bfloat16, deadline=math.inf):
    """Baseline 5-inner-step, cubic-rho, 0.03-ridge ADMM with optional projected start."""
    torch.set_float32_matmul_precision("highest")
    generator = torch.Generator(device=fixture["target"].device).manual_seed(seed)
    target = fixture["target"].to(dtype)
    wi = fixture["input_importance"].float().sqrt().clamp_min(1e-12)
    wo = fixture["output_importance"].float().sqrt().clamp_min(1e-12).reshape(-1, 1)
    normalized = target * wi.reshape(1, -1) * wo

    def project(value):
        return _rank_one_sign_projection(value, 5, generator, 1e-12, "power")

    if initial is None:
        left = torch.randn((target.shape[0], rank), dtype=dtype, device=target.device, generator=generator)
        right = torch.randn((rank, target.shape[1]), dtype=dtype, device=target.device, generator=generator)
        lp, rp = project(left), project(right)
        ld, rd = left - lp, right - rp
    else:
        left, right = lift_candidate(fixture, initial, dtype)
        if left.shape[1] != rank:
            raise ValueError("candidate rank mismatch")
        lp, rp = project(left), project(right)
        ld, rd = left - lp, right - rp
    trace = []
    for iteration in range(iterations):
        if time.perf_counter() >= deadline:
            raise BudgetExpired
        rho = cubic_schedule(iteration / max(1, iterations))
        right_norm = rp.norm(dim=1).clamp_min(1e-12)
        left = _solve(rp.mT / right_norm, normalized.mT, lp.mT, ld.mT, rho, 0.03, 1e-12).mT
        left_norm = lp.norm(dim=0).clamp_min(1e-12)
        right = _solve(lp / left_norm, normalized, rp, rd, rho, 0.03, 1e-12)
        previous_left, previous_right = lp, rp
        lp, rp = project(left + ld), project(right + rd)
        ld.add_(left - lp)
        rd.add_(right - rp)
        if iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == iterations:
            trace.append({
                "iteration": iteration + 1, "rho": rho,
                "primal_residual": float((left - lp).norm() + (right - rp).norm()),
                "dual_residual": float(rho * ((lp - previous_left).norm() + (rp - previous_right).norm())),
            })
    # Same export ordering and dtype conversions as factorize_admm.
    lu, ru = lp / wo, rp / wi
    balance = (ru.norm().clamp_min(1e-12) / lu.norm().clamp_min(1e-12)).sqrt()
    le, re = lu * balance, ru / balance
    le = le * lp.norm(dim=0).clamp_min(1e-12).reciprocal()
    right_u, pre, right_binary = _svid_components(re.float(), 5, generator, 1e-12, "power")
    left_u, post, left_sign = _svid_components(le.mT.float(), 5, generator, 1e-12, "power")
    state = dict(zip(STATE_KEYS, (
        left_sign.mT.to(dtype).contiguous().float(), right_binary.to(dtype).contiguous().float(),
        pre.to(dtype).float(), (right_u * left_u).to(dtype).float(), post.to(dtype).float(),
    ), strict=True))
    return state, trace


def scale_candidate(fixture, state):
    scaled = fit_scales(
        fixture["target"], *(state[k] for k in STATE_KEYS),
        fixture["input_importance"], fixture["output_importance"], alternating_passes=2,
    )
    return {**state, **{k: getattr(scaled, k) for k in STATE_KEYS[2:]}}


def eligible_improvement(error, best_error, elapsed, budget):
    return elapsed <= budget and error < best_error


def run(args):
    if (args.output.resolve() / "cancellation.json").exists():
        print("This experiment was cancelled. Its completed checkpoints are preserved; use a new output for a new run.")
        return
    output, source, fixture_source = args.output.resolve(), args.source.resolve(), args.fixture.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads((source / "report.json").read_text())
    metadata = json.loads((source / "fixture.json").read_text())
    if prior["status"] != "complete":
        raise ValueError("source search is incomplete")
    inputs = {
        str(fixture_source): metadata["fixture_sha256"],
        str(source / "baseline.safetensors"): prior["inputs"]["baseline.safetensors"],
        str(source / "best.safetensors"): prior["best_sha256"],
        str(source / "report.json"): hash_file(source / "report.json"),
    }
    for filename, expected in inputs.items():
        if hash_file(Path(filename)) != expected:
            raise ValueError(f"input hash mismatch: {filename}")
    files = [Path(__file__), ROOT / "tools/probe_hour_factor_fit.py", ROOT / "tools/probe_hour_basin_search.py"]
    files += [ROOT / f"src/nanoquant/domain/{name}.py" for name in (
        "factorization", "scale_fit", "linear_math", "functional_binary_population",
    )]
    files.append(ROOT / "Docs/103-hour-admm-perturbations.md")
    provenance = {str(p): hash_file(p) for p in files}
    path = output / "report.json"
    if path.exists():
        report = json.loads(path.read_text())
        if report["inputs"] != inputs or report["provenance"] != provenance or report["budget_seconds"] != args.seconds:
            raise ValueError("resume identity, source, or budget mismatch")
        if report["status"] == "complete":
            print(json.dumps(report["comparison"], indent=2))
            return
    else:
        shutil.copy2(source / "baseline.safetensors", output / "baseline.safetensors")
        shutil.copy2(source / "best.safetensors", output / "inherited.safetensors")
        shutil.copy2(source / "fixture.json", output / "fixture.json")
        shutil.copy2(ROOT / "Docs/103-hour-admm-perturbations.md", output / "protocol.md")
        report = {
            "experiment": 68, "schema_version": 1, "status": "preparing", "inputs": inputs,
            "provenance": provenance, "budget_seconds": args.seconds, "records": [], "next_attempt": 0,
            "max_attempt_seconds": 15.0, "process_history": [],
            "policy": {"flip_probability": 0.1, "outer_iterations": 800, "inner_iterations": 5,
                       "scale_passes": 2, "arithmetic": "BF16 projected factors, IEEE FP32 solves",
                       "parent": "best eligible state", "dual": "initial projection residual, no inherited dual",
                       "rho": "cubic reset from zero",
                       "seed": "68000 + attempt index; control 67999"},
        }
        atomic_write_json(path, report)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    with acquire_device_lease(args.device):
        torch.use_deterministic_algorithms(True)
        fixture = load_file(str(fixture_source), device=args.device)

        def load(row):
            p = output / row["state_file"]
            if hash_file(p) != row["state_sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {p}")
            return {k: v.float() for k, v in load_file(str(p), device=args.device).items()}

        def event(data):
            with (output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps({"utc": time.time(), **data}) + "\n")
            print(json.dumps(data), flush=True)

        def write():
            atomic_write_json(path, report)

        report["process_history"].append({"pid": os.getpid(), "utc": time.time()})
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(), "threads": torch.get_num_threads()}
        if "started_utc" not in report:
            inherited = {"state_file": "inherited.safetensors", "state_sha256": prior["best_sha256"]}
            report["inherited"] = {**inherited, **score(fixture, load(inherited)), "elapsed_seconds": 0.0}
            report["best"] = report["inherited"]
            torch.cuda.synchronize()
            report["started_utc"] = time.time()
            report["deadline_utc"] = report["started_utc"] + args.seconds
        deadline = time.perf_counter() + max(0.0, report["deadline_utc"] - time.time())
        torch.cuda.reset_peak_memory_stats()
        report["status"] = "searching"
        if report.get("pending"):
            event({"event": "skipped_interrupted_attempt", **report.pop("pending")})
        write()
        while time.perf_counter() + report["max_attempt_seconds"] * 1.2 + 2 < deadline:
            started = time.perf_counter()
            control = "control" not in report
            index = report["next_attempt"]
            if control:
                name, seed, probability = "control", 67999, 0.0
            else:
                name, seed, probability = f"candidate-{index:04d}", 68000 + index, 0.10
                report["next_attempt"] += 1
            parent = report["inherited"] if control else report["best"]
            report["pending"] = {"name": name, "seed": seed, "parent": parent["state_file"]}
            write()
            parent_state = load(parent)
            candidate, flips = perturb(parent_state, seed, probability)
            proposal_error = score(fixture, candidate)["weighted_error"]
            proposal_sha = save_state(output / f"{name}-initial.safetensors", candidate)
            try:
                state, trace = admm_fit(fixture, metadata["rank"], seed, initial=candidate, deadline=deadline)
            except BudgetExpired:
                event({"event": "deadline_interruption", "name": name})
                report["interrupted"] = report.pop("pending")
                break
            raw_error = score(fixture, state)["weighted_error"]
            state = scale_candidate(fixture, state)
            validate_state(state, tuple(metadata["shape"]), metadata["rank"])
            metrics = score(fixture, state)
            sha = save_state(output / f"{name}.safetensors", state)
            sign_hash = canonical_binary_hash(state["left_binary"], state["right_binary"])
            distance = {k: int((state[k] != parent_state[k]).sum()) for k in STATE_KEYS[:2]}
            torch.cuda.synchronize()
            elapsed = time.time() - report["started_utc"]
            seconds = time.perf_counter() - started
            row = {
                "name": name, "kind": "control" if control else "10_percent", "seed": seed,
                "parent": parent["state_file"], "parent_sha256": parent["state_sha256"],
                "initial_file": f"{name}-initial.safetensors", "initial_sha256": proposal_sha,
                "initial_flips": flips, "initial_error": proposal_error, "admm_error_before_scale_fit": raw_error,
                "state_file": f"{name}.safetensors", "state_sha256": sha, "canonical_hash": sign_hash,
                "final_distance_from_parent": distance, "elapsed_seconds": elapsed, "seconds": seconds,
                "within_budget": elapsed <= args.seconds, "iterations_completed": 800, "trace": trace, **metrics,
            }
            row["new_best"] = eligible_improvement(metrics["weighted_error"], report["best"]["weighted_error"],
                                                   elapsed, args.seconds)
            report["records"].append(row)
            if row["new_best"]:
                report["best"] = row
            if control:
                report["control"] = row
            elif row["within_budget"] and metrics["weighted_error"] < report.get(
                    "best_perturbed", {"weighted_error": math.inf})["weighted_error"]:
                report["best_perturbed"] = row
            report["max_attempt_seconds"] = max(report["max_attempt_seconds"], seconds)
            report.pop("pending")
            write()
            event({k: row[k] for k in ("name", "elapsed_seconds", "weighted_error", "initial_error", "new_best")})
        report["search_elapsed_seconds"] = time.time() - report["started_utc"]
        report["status"] = "auditing"
        report["peak_cuda_allocated"] = torch.cuda.max_memory_allocated()
        report["peak_cuda_reserved"] = torch.cuda.max_memory_reserved()
        report["cuda_memory_scope"] = "Current worker; process_history identifies any prior workers"
        write()
        for row in report["records"]:
            for file_key, hash_key in (("state_file", "state_sha256"), ("initial_file", "initial_sha256")):
                if hash_file(output / row[file_key]) != row[hash_key]:
                    raise ValueError("completed checkpoint hash mismatch")
        for filename, expected in {**inputs, **provenance}.items():
            if hash_file(Path(filename)) != expected:
                raise ValueError("input or numerical source changed during search")
        baseline = {"state_file": "baseline.safetensors",
                    "state_sha256": inputs[str(source / "baseline.safetensors")]}
        comparators = {"baseline": baseline, "inherited": report["inherited"], "best": report["best"]}
        comparators.update({k: report[k] for k in ("control", "best_perturbed") if k in report})
        report["comparison"] = {k: {**score(fixture, load(row), double=True),
                                    "state_file": row["state_file"], "state_sha256": row["state_sha256"]}
                                for k, row in comparators.items()}
        audit_keys = ("best", "best_perturbed") if "best_perturbed" in comparators else ("best",)
        report["neighborhood_audit"] = {k: neighborhood_audit(fixture, load(comparators[k])) for k in audit_keys}
        report["best_sha256"] = save_state(output / "best.safetensors", load(report["best"]))
        report["saved_state_bytes"] = sum(p.stat().st_size for p in output.glob("*.safetensors"))
        report["status"] = "complete"
        write()
        event({"event": "complete", "search_seconds": report["search_elapsed_seconds"],
               "comparison": report["comparison"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/068/admm-perturb")
    parser.add_argument("--source", type=Path, default=ROOT / "evidence/067/basin-search")
    parser.add_argument("--fixture", type=Path, default=ROOT / "evidence/066/hour-fit/fixture.safetensors")
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    run(args)
