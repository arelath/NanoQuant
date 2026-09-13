"""Experiment 067: checkpointed basin hopping around the retained hour-fit winner."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from probe_hour_factor_fit import STATE_KEYS, save_state, score, state_of, validate_state
from safetensors.torch import load_file

from nanoquant.domain.binary_factor_search import refine_binary_factors_separable
from nanoquant.domain.functional_binary_population import canonical_binary_hash
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

ROOT = Path(__file__).resolve().parents[1]
SEED = 67000


def flip_costs(fixture: dict[str, torch.Tensor], state: dict[str, torch.Tensor], *, double: bool = False):
    """Exact fixed-scale one-bit energy changes; no hypothetical scale refitting."""
    torch.set_float32_matmul_precision("highest")
    dtype = torch.float64 if double else torch.float32
    left, right, pre, mid, post = (state[key].to(dtype) for key in STATE_KEYS)
    wi, wo = (fixture[key].to(dtype) for key in ("input_importance", "output_importance"))
    residual = (left * post[:, None]) @ (right * mid[:, None] * pre[None, :]) - fixture["target"].to(dtype)
    left_cross = (residual * wi[None, :]) @ (right * pre[None, :]).mT
    right_cross = (left * post[:, None]).mT @ (residual * wo[:, None])
    left_cost = 4 * wo[:, None] * (
        post.square()[:, None] * mid.square()[None, :] * (pre.square() * wi).sum()
        - left * post[:, None] * mid[None, :] * left_cross
    )
    right_cost = 4 * wi[None, :] * (
        mid.square()[:, None] * pre.square()[None, :] * (post.square() * wo).sum()
        - right * mid[:, None] * pre[None, :] * right_cross
    )
    return left_cost, right_cost


def neighborhood_audit(fixture, state) -> dict[str, Any]:
    costs = flip_costs(fixture, state, double=True)
    tolerance = 1e-10
    return {
        "scope": "All individual sign flips with all scales and the opposite factor fixed; float64",
        "absolute_energy_tolerance": tolerance,
        "bits_examined": sum(x.numel() for x in costs),
        "improving_bits": sum(int((x < -tolerance).sum()) for x in costs),
        "best_single_flip_energy_change": min(float(x.min()) for x in costs),
    }


def refine(fixture, state, kind: str):
    torch.set_float32_matmul_precision("highest")
    result = refine_binary_factors_separable(
        fixture["target"], *(state[k] for k in STATE_KEYS),
        fixture["input_importance"], fixture["output_importance"],
        outer_passes=1, scale_passes=8, continuous_candidates=False,
        one_bit_passes=32 if kind == "descent" else 0,
        variable_depth_passes=2 if kind == "descent" else 0,
        variable_depth_length=64,
        tabu_passes=2 if kind == "tabu" else 0, tabu_steps=256,
        tabu_tenure=8, tabu_tenure_jitter=4,
        pair_passes=1 if kind == "wide" else 0, pair_pool_size=8,
        block_passes=0, component_passes=1 if kind == "wide" else 0,
        component_limit=4, component_alternating_steps=4,
    )
    stats = {k: int(getattr(result, k)) for k in (
        "one_bit_updates", "variable_depth_updates", "tabu_updates", "pair_updates", "component_updates"
    )}
    return state_of(result), stats


def proposal_spec(index: int, *, large: bool = False) -> tuple[str, float, int]:
    if large:
        return "uniform", 0.05, 0
    return (
        ("weak", 32, 0), ("weak", 256, 0), ("uniform", 0.0001, 0),
        ("component", 0.02, 2), ("weak", 2048, 0), ("uniform", 0.001, 0),
        ("component", 0.10, 8), ("weak", 8192, 0), ("wide", 0, 0),
        ("elite", 0, 0), ("component", 0.25, 2), ("uniform", 0.005, 0),
    )[index % 12]


def perturb(fixture, state, index: int, *, large: bool = False):
    kind, radius, count = proposal_spec(index, large=large)
    generator = torch.Generator(device=state["left_binary"].device).manual_seed(SEED + index)
    candidate = {k: v.clone() for k, v in state.items()}
    components: list[int] = []
    if kind == "weak":
        # Randomly choose among low exact-cost flips, retaining diversity rather than a fixed tie order.
        costs = flip_costs(fixture, state)
        for key, cost in zip(STATE_KEYS[:2], costs, strict=True):
            n = min(int(radius), cost.numel())
            pool = cost.flatten().topk(min(16 * n, cost.numel()), largest=False).indices
            selected = pool[torch.randperm(pool.numel(), generator=generator, device=pool.device)[:n]]
            candidate[key].view(-1)[selected] *= -1
    elif kind == "uniform":
        for key in STATE_KEYS[:2]:
            mask = torch.rand(state[key].shape, generator=generator, device=state[key].device) < radius
            candidate[key][mask] *= -1
    elif kind == "component":
        rank = state["left_binary"].shape[1]
        components = torch.randperm(rank, generator=generator, device=state["left_binary"].device)[:count].tolist()
        for c in components:
            for vector in (candidate["left_binary"][:, c], candidate["right_binary"][c]):
                mask = torch.rand(vector.shape, generator=generator, device=vector.device) < radius
                vector[mask] *= -1
    return candidate, {
        "proposal": kind, "radius": radius, "components": components, "seed": SEED + index,
        "left_flips": int((candidate["left_binary"] != state["left_binary"]).sum()),
        "right_flips": int((candidate["right_binary"] != state["right_binary"]).sum()),
    }


def accept_walker(candidate_error: float, walker_error: float, best_error: float, fraction: float, seed: int):
    temperature = best_error * (0.001 * (1 - fraction) + 0.00005 * fraction)
    probability = math.exp(min(0.0, (walker_error - candidate_error) / temperature))
    return random.Random(seed).random() < probability


def run(args):
    output, source = args.output.resolve(), args.source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((source / "report.json").read_text())
    metadata = json.loads((source / "fixture.json").read_text())
    if old["status"] != "complete":
        raise ValueError("previous search must be complete")
    inputs = {
        "fixture.safetensors": metadata["fixture_sha256"],
        "baseline.safetensors": old["baseline"]["state_sha256"],
        "best.safetensors": old["best_sha256"],
    }
    for name, expected in inputs.items():
        if hash_file(source / name) != expected:
            raise ValueError(f"source checkpoint hash mismatch: {name}")
    files = [Path(__file__), ROOT / "tools/probe_hour_factor_fit.py"] + [
        ROOT / f"src/nanoquant/domain/{name}.py"
        for name in ("binary_factor_search", "functional_binary_population", "scale_fit")
    ]
    provenance = {str(p): hash_file(p) for p in files}
    path = output / "report.json"
    if path.exists():
        report = json.loads(path.read_text())
        if args.amend_large_flips and not report.get("large_flips"):
            if report["status"] != "searching" or time.time() >= report["deadline_utc"]:
                raise ValueError("only a live unfinished search can be amended")
            # Preserve the full earlier receipt and source before accepting the user's protocol change.
            frozen = output / "pre-large-flip-runner.py"
            previous = output / "pre-large-flip-report.json"
            if json.loads(previous.read_text()) != report:
                raise ValueError("pre-amendment report snapshot mismatch")
            for name, expected in report["provenance"].items():
                checked = frozen if Path(name) == Path(__file__) else Path(name)
                if hash_file(checked) != expected:
                    raise ValueError("pre-amendment numerical provenance mismatch")
            report["protocol_changes"] = [{
                "utc": time.time(), "elapsed_seconds": time.time() - report["started_utc"],
                "reason": "User requested much larger 5% sign flips, then refinement",
                "previous_provenance": report["provenance"],
                "previous_report_sha256": hash_file(previous),
                "previous_runner_sha256": hash_file(frozen),
                "completed_attempts_before_change": len(report["attempts"]),
                "best_before_change": report["best"],
                "policy": "5% uniform flips per factor; up to 64 refinement passes; at least 12 before pruning",
            }]
            report["large_flips"] = True
            report["provenance"] = provenance
            report["walker"] = report["best"]
            report["phase"] = "hopping"
            atomic_write_json(path, report)
        if (report["provenance"] != provenance or report["inputs"] != inputs
                or report["budget_seconds"] != args.seconds or report["source"] != str(source)):
            raise ValueError("resume identity or budget changed")
        if report["status"] == "complete":
            print(json.dumps(report["comparison"], indent=2), flush=True)
            return
    else:
        shutil.copy2(source / "fixture.json", output / "fixture.json")
        for src, dest in (("baseline.safetensors", "baseline.safetensors"),
                          ("best.safetensors", "previous-best.safetensors")):
            shutil.copy2(source / src, output / dest)
        report = {
            "experiment": 67, "status": "preparing", "source": str(source), "inputs": inputs,
            "provenance": provenance, "budget_seconds": args.seconds, "records": [],
            "next_attempt": 0, "control_steps": 0, "control_stalls": 0, "seen_proposals": [],
            "phase": "control", "timings": {}, "attempts": [],
        }
        atomic_write_json(path, report)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    with acquire_device_lease(args.device):
        torch.use_deterministic_algorithms(True)
        fixture = load_file(str(source / "fixture.safetensors"), device=args.device)

        def load(row):
            p = output / row["state_file"]
            if hash_file(p) != row["state_sha256"]:
                raise ValueError(f"state hash mismatch: {p}")
            return {k: v.float() for k, v in load_file(str(p), device=args.device).items()}

        def write():
            atomic_write_json(path, report)

        def event(data):
            data = {"utc": time.time(), **data}
            with (output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(data) + "\n")
            print(json.dumps(data), flush=True)

        if "started_utc" in report:
            report.setdefault("process_history", []).append(report["environment"])
            report["environment"] = {**report["environment"], "pid": os.getpid(), "resumed_utc": time.time()}
            report["cuda_memory_scope"] = "Allocator peak of the resumed worker; first-worker counters unavailable"
            write()
        if "started_utc" not in report:
            state = {k: v.float() for k, v in load_file(str(output / "previous-best.safetensors"),
                                                      device=args.device).items()}
            validate_state(state, tuple(metadata["shape"]), metadata["rank"])
            report["best"] = {"state_file": "previous-best.safetensors", "state_sha256": inputs["best.safetensors"],
                              **score(fixture, state), "name": "previous-best"}
            report["walker"] = report["best"]
            report["control_best"] = report["best"]
            report["environment"] = {"pid": os.getpid(), "torch": torch.__version__, "cuda": torch.version.cuda,
                                      "gpu": torch.cuda.get_device_name(args.device)}
            torch.cuda.synchronize()
            report["started_utc"] = time.time()
            report["deadline_utc"] = report["started_utc"] + args.seconds
            report["status"] = "searching"
            write()
        deadline = time.perf_counter() + max(0.0, report["deadline_utc"] - time.time())

        def remaining():
            return deadline - time.perf_counter()

        def affordable(kind):
            return remaining() > max(3.0, 1.4 * report["timings"].get(kind, 15.0)) + 1.0

        def commit(state, details):
            validate_state(state, tuple(metadata["shape"]), metadata["rank"])
            metrics = score(fixture, state)
            name = f"state-{len(report['records']):04d}"
            sha = save_state(output / f"{name}.safetensors", state)
            row = {"name": name, "state_file": f"{name}.safetensors", "state_sha256": sha,
                   "canonical_hash": canonical_binary_hash(state["left_binary"], state["right_binary"]),
                   **details, **metrics, "elapsed_seconds": time.time() - report["started_utc"]}
            row["new_best"] = metrics["weighted_error"] < report["best"]["weighted_error"]
            # A bounded operation that overruns the cap is retained as evidence, never selected.
            row["within_budget"] = remaining() >= 0
            report["records"].append(row)
            if row["new_best"] and row["within_budget"]:
                report["best"] = row
            write()
            event({"event": "completed", **row})
            return row

        def step(state, kind, details):
            start = time.perf_counter()
            candidate, stats = refine(fixture, state, kind)
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            report["timings"][kind] = max(duration, report["timings"].get(kind, 0.0))
            return candidate, commit(candidate, {"move": kind, "seconds": duration, **details, **stats})

        # Freeze a continued-local-refinement comparator before any external kick.
        while report["phase"] == "control" and affordable("descent") and affordable("tabu"):
            i = report["control_steps"]
            if (time.time() - report["started_utc"] >= min(240, args.seconds * 0.1)
                    or report["control_stalls"] >= 4):
                report["phase"] = "hopping"
                report["walker"] = report["best"]
                write()
                break
            report["control_steps"] += 1
            write()
            parent = report["control_best"]
            _, row = step(load(parent), ("descent", "tabu")[i % 2],
                          {"family": "continued-local-control", "parent": parent["name"]})
            gain = 1 - row["weighted_error"] / parent["weighted_error"]
            report["control_stalls"] = report["control_stalls"] + 1 if gain < 1e-6 else 0
            if row["within_budget"] and gain > 0:
                report["control_best"] = row
            write()

        elite_rows = sorted((r for r in old["records"] if r.get("family") == "joint-admm"),
                            key=lambda r: r["weighted_error"])
        elites, groups = [], set()
        for row in elite_rows:
            group = (row["mode"], row["seed"])
            if group not in groups:
                elites.append(row)
                groups.add(group)

        while report["phase"] == "hopping" and remaining() > min(120, args.seconds * 0.1):
            if not affordable("descent"):
                break
            index = report["next_attempt"]
            report["next_attempt"] += 1
            if index % 18 == 0:
                report["walker"] = report["best"]
            parent = report["walker"] if index % 3 == 1 else report["best"]
            write()  # Interrupted attempts are skipped on resume; the original deadline is unchanged.
            original = load(parent)
            large = report.get("large_flips", False)
            state, details = perturb(fixture, original, index, large=large)
            details["campaign"] = "large-5pct" if large else "original-local"
            if details["proposal"] == "elite":
                elite = elites[(index // 12) % len(elites)]
                if hash_file(source / elite["state_file"]) != elite["state_sha256"]:
                    raise ValueError("elite checkpoint hash mismatch")
                state = {k: v.float() for k, v in load_file(str(source / elite["state_file"]),
                                                          device=args.device).items()}
                details.update({"elite_source": elite["state_file"], "elite_sha256": elite["state_sha256"]})
            proposal_hash = canonical_binary_hash(state["left_binary"], state["right_binary"])
            details.update({"attempt": index, "family": "basin-search", "parent": parent["name"],
                            "parent_sha256": parent["state_sha256"], "proposal_hash": proposal_hash,
                            "proposal_error": score(fixture, state)["weighted_error"]})
            # A repeated wide/elite proposal may have different scales; do not deduplicate by signs alone.
            if details["proposal"] not in ("wide", "elite") and proposal_hash in report["seen_proposals"]:
                event({"event": "duplicate-proposal", "attempt": index})
                continue
            report["seen_proposals"].append(proposal_hash)
            event({"event": "proposal", **details})
            row = None
            for local_step in range(64 if large else 8):
                kind = "wide" if details["proposal"] == "wide" and local_step == 0 else (
                    "tabu" if local_step % 3 == 2 else "descent")
                if not affordable(kind):
                    break
                before = score(fixture, state)["weighted_error"]
                state, row = step(state, kind, {**details, "local_step": local_step})
                if (local_step >= (11 if large else 1)
                        and row["weighted_error"] > report["best"]["weighted_error"] * (1.01 if large else 1.003)):
                    break
                if local_step >= (11 if large else 2) and 1 - row["weighted_error"] / before < 1e-7:
                    break
            if row is not None:
                fraction = min(1.0, (time.time() - report["started_utc"]) / args.seconds)
                accepted = row["within_budget"] and accept_walker(
                    row["weighted_error"], report["walker"]["weighted_error"],
                    report["best"]["weighted_error"], fraction, SEED + index,
                )
                uphill = accepted and row["weighted_error"] > report["walker"]["weighted_error"]
                if accepted:
                    report["walker"] = row
                report["attempts"].append({**details, "final": row["name"], "final_error": row["weighted_error"],
                                            "walker_accepted": accepted, "walker_uphill": uphill,
                                            "final_hash": row["canonical_hash"]})
                write()

        # Remaining budget refines the incumbent; audit outside the clock never changes it.
        report["phase"] = "finish"
        i = 0
        while affordable("descent"):
            kind = ("descent", "tabu", "wide")[i % 3]
            if not affordable(kind):
                kind = "descent"
            parent = report["best"]
            step(load(parent), kind, {"family": "final-polish", "parent": parent["name"]})
            i += 1
        if "finished_utc" not in report:
            report["finished_utc"] = time.time()
            report["search_seconds"] = report["finished_utc"] - report["started_utc"]
        report["peak_allocated_bytes"] = max(report.get("peak_allocated_bytes", 0), torch.cuda.max_memory_allocated())
        report["peak_reserved_bytes"] = max(report.get("peak_reserved_bytes", 0), torch.cuda.max_memory_reserved())
        report["status"] = "validating"
        write()
        baseline = {"state_file": "baseline.safetensors", "state_sha256": inputs["baseline.safetensors"]}
        previous = {"state_file": "previous-best.safetensors", "state_sha256": inputs["best.safetensors"]}
        comparisons, audits = {}, {}
        for name, row in (("baseline", baseline), ("previous_best", previous),
                          ("continued_local", report["control_best"]), ("best", report["best"])):
            state = load(row)
            comparisons[name] = score(fixture, state, double=True)
            audits[name] = neighborhood_audit(fixture, state)
        for name in ("baseline", "previous_best", "continued_local"):
            comparisons[f"energy_reduction_vs_{name}_percent"] = 100 * (
                1 - comparisons["best"]["weighted_error"] / comparisons[name]["weighted_error"])
            comparisons[f"nrmse_reduction_vs_{name}_percent"] = 100 * (
                1 - comparisons["best"]["weighted_nrmse"] / comparisons[name]["weighted_nrmse"])
        for row in report["records"]:
            if hash_file(output / row["state_file"]) != row["state_sha256"]:
                raise ValueError("final checkpoint audit failed")
        if provenance != {str(p): hash_file(p) for p in files}:
            raise ValueError("numerical source changed during search")
        report["best_sha256"] = save_state(output / "best.safetensors", load(report["best"]))
        report["comparison"], report["neighborhood_audits"] = comparisons, audits
        report["status"] = "complete"
        write()
        event({"event": "complete", "comparison": comparisons, "audits": audits})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "evidence/066/hour-fit")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/067/basin-search")
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amend-large-flips", action="store_true",
                        help="Apply the user-authorized 5%% protocol amendment to a preserved unfinished run")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    run(args)
