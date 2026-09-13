"""A timed, checkpointed full-matrix binary-factor fitting experiment.

Reuses the production ADMM, scale fit, and binary search implementations.
Only this analysis runner chooses arithmetic, restart, and horizon policies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import _load_cases
from safetensors.torch import load_file, save_file

from nanoquant.domain.binary_factor_search import refine_binary_factors_separable
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

ROOT = Path(__file__).resolve().parents[1]
REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
STATE_KEYS = ("left_binary", "right_binary", "scale_pre", "scale_mid", "scale_post")
MODES = {
    "bf16-ieee": (torch.bfloat16, "highest"),
    "bf16-tf32": (torch.bfloat16, "high"),
    "fp32-ieee": (torch.float32, "highest"),
    "fp32-tf32": (torch.float32, "high"),
    "fp32-medium": (torch.float32, "medium"),
}


class BudgetExpired(Exception):
    """The timed search stopped at an ADMM iteration boundary."""


class DeadlineRecorder:
    def __init__(self, deadline: float, progress: Any = None) -> None:
        self.deadline = deadline
        self.progress = progress
        self.last_progress = time.perf_counter()
        self.iterations = 0

    def phase(self, name: str, /, **attributes: object) -> Any:
        if name == "iteration":
            if time.perf_counter() >= self.deadline:
                raise BudgetExpired
            if self.progress is not None and time.perf_counter() - self.last_progress >= 30:
                self.progress(self.iterations)
                self.last_progress = time.perf_counter()
        return nullcontext()

    def add(self, counter: str, value: float, /, **attributes: object) -> None:
        if counter == "admm.iterations":
            self.iterations += int(value)

    def mark(self, name: str, /, **attributes: object) -> None:
        pass


def state_of(result: Any) -> dict[str, torch.Tensor]:
    return {key: getattr(result, key).detach().float().contiguous() for key in STATE_KEYS}


def validate_state(state: dict[str, torch.Tensor], shape: tuple[int, int], rank: int) -> None:
    expected = ((shape[0], rank), (rank, shape[1]), (shape[1],), (rank,), (shape[0],))
    for key, dimensions in zip(STATE_KEYS, expected, strict=True):
        if tuple(state[key].shape) != dimensions or not bool(torch.isfinite(state[key]).all()):
            raise ValueError(f"invalid factor state: {key}")
    for key in STATE_KEYS[:2]:
        if not bool((state[key].abs() == 1).all()):
            raise ValueError(f"nonbinary signs: {key}")


def score(
    fixture: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    double: bool = False,
) -> dict[str, float]:
    """One common IEEE score; float64 reduction, optionally float64 GEMM too."""
    torch.set_float32_matmul_precision("highest")
    dtype = torch.float64 if double else torch.float32
    left, right, pre, mid, post = (state[key].to(dtype) for key in STATE_KEYS)
    prediction = (left * post[:, None]) @ (right * mid[:, None] * pre[None, :])
    difference = prediction - fixture["target"].to(dtype)
    weights = fixture["output_importance"].to(dtype)[:, None] * fixture["input_importance"].to(dtype)[None, :]
    error = float((difference.square() * weights).sum(dtype=torch.float64))
    energy = float((fixture["target"].to(dtype).square() * weights).sum(dtype=torch.float64))
    raw_error = float(difference.square().sum(dtype=torch.float64))
    raw_energy = float(fixture["target"].to(dtype).square().sum(dtype=torch.float64))
    if not math.isfinite(error) or energy <= 0:
        raise ValueError("nonfinite or empty scoring objective")
    return {
        "weighted_error": error,
        "weighted_target_energy": energy,
        "weighted_nmse": error / energy,
        "weighted_nrmse": math.sqrt(error / energy),
        "raw_error": raw_error,
        "raw_nrmse": math.sqrt(raw_error / raw_energy),
    }


def save_state(path: Path, state: dict[str, torch.Tensor]) -> str:
    payload = {
        key: value.cpu().to(torch.int8 if key in STATE_KEYS[:2] else torch.float32).contiguous()
        for key, value in state.items()
    }
    temporary = path.with_suffix(".tmp.safetensors")
    save_file(payload, str(temporary))
    temporary.replace(path)
    return hash_file(path)


def prepare(output: Path, source: Path, model: Path) -> dict[str, Any]:
    path = output / "fixture.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["fixture_sha256"] != hash_file(output / "fixture.safetensors"):
            raise ValueError("fixture hash mismatch")
        if result["source_run"] != str(source.resolve()) or result["model"] != str(model.resolve()):
            raise ValueError("fixture source mismatch")
        return result
    identity, cases = _load_cases(source, model, (12,), ("mlp.down_proj",), 26)
    case = cases[0]
    if case.patch is not None:
        raise ValueError("this experiment requires a pure binary-plus-fixed-outlier owner")
    payload = {
        "target": case.target.contiguous(),
        "input_importance": case.input_importance.contiguous(),
        "output_importance": case.output_importance.contiguous(),
        **{f"retained_{key}": value for key, value in state_of(case).items()},
    }
    if case.outlier_indices is not None:
        payload["outlier_indices"] = case.outlier_indices.contiguous()
    if case.outlier_values is not None:
        payload["outlier_values"] = case.outlier_values.contiguous()
    save_file(payload, str(output / "fixture.safetensors"))
    result = {
        "schema_version": 1,
        "experiment": 66,
        "model_revision": REVISION,
        "source_run": str(source.resolve()),
        "model": str(model.resolve()),
        "source_identity": asdict(identity),
        "source_references": case.references,
        "source_key": "model.layers.12.mlp.down_proj.weight",
        "shape": list(case.target.shape),
        "rank": case.left_binary.shape[1],
        "outlier_indices": case.outlier_indices.tolist() if case.outlier_indices is not None else [],
        "fixture_sha256": hash_file(output / "fixture.safetensors"),
        "objective_scope": (
            "Retained Experiment 041 diagonal Fisher residual objective; not Experiment 060's missing calibration"
        ),
    }
    atomic_write_json(path, result)
    return result


def fit(
    fixture: dict[str, torch.Tensor],
    rank: int,
    mode: str,
    seed: int,
    iterations: int,
    deadline: float = math.inf,
    progress: Any = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    dtype, precision = MODES[mode]
    torch.set_float32_matmul_precision(precision)
    torch.cuda.synchronize()
    started = time.perf_counter()
    recorder = DeadlineRecorder(deadline, progress)
    result = factorize_admm_with_parameters(
        fixture["target"].to(dtype),
        fixture["input_importance"],
        fixture["output_importance"],
        rank,
        torch.Generator(device=fixture["target"].device).manual_seed(seed),
        AdmmParameters(outer_iterations=iterations, transpose_wide=False),
        recorder=recorder,
    )
    torch.cuda.synchronize()
    admm_seconds = time.perf_counter() - started
    if time.perf_counter() >= deadline:
        raise BudgetExpired
    # Every arm gets the production two-pass scale fit at identical IEEE precision.
    torch.set_float32_matmul_precision("highest")
    scaled = fit_scales(
        fixture["target"],
        result.left_binary,
        result.right_binary,
        result.scale_pre,
        result.scale_mid,
        result.scale_post,
        fixture["input_importance"],
        fixture["output_importance"],
        alternating_passes=2,
    )
    state = state_of(result)
    state.update({key: getattr(scaled, key) for key in STATE_KEYS[2:]})
    metrics = score(fixture, state)
    torch.cuda.synchronize()
    return state, {
        "mode": mode,
        "seed": seed,
        "outer_iterations": iterations,
        "inner_iterations": 5,
        "observed_outer_iterations": result.iterations_completed,
        "scale_passes": 2,
        "admm_seconds": admm_seconds,
        "wall_seconds": time.perf_counter() - started,
        **metrics,
    }


def polish(
    fixture: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    tabu: bool,
) -> dict[str, torch.Tensor]:
    # Bounded existing exact-objective search, one checkpointable outer pass at a time.
    torch.set_float32_matmul_precision("highest")
    result = refine_binary_factors_separable(
        fixture["target"],
        *(state[key] for key in STATE_KEYS),
        fixture["input_importance"],
        fixture["output_importance"],
        outer_passes=1,
        scale_passes=8,
        continuous_candidates=False,
        one_bit_passes=0 if tabu else 16,
        variable_depth_passes=0 if tabu else 2,
        variable_depth_length=64,
        tabu_passes=2 if tabu else 0,
        tabu_steps=256,
        tabu_tenure=8,
        tabu_tenure_jitter=4,
        pair_passes=0,
        block_passes=0,
        component_passes=0,
    )
    return state_of(result)


def choose_mode(pilots: list[dict[str, Any]]) -> str:
    reference = next(item for item in pilots if item["mode"] == "fp32-ieee")
    eligible = [item for item in pilots if item["weighted_error"] <= 1.05 * reference["weighted_error"]]
    return str(min(eligible, key=lambda item: item["admm_seconds"])["mode"])


def schedule(index: int, fastest: str) -> tuple[str, int, int]:
    """Alternate deeper joint fits and new basins; keep the recipe predeclared."""
    # Pair speeds with latent precision: BF16 rounding limits can be invisible
    # in the short speed pilot but material over a long optimization trajectory.
    alternate = "fp32-tf32" if fastest.startswith("bf16") else "bf16-tf32"
    initial = [("bf16-ieee", 1, 800), ("bf16-ieee", 2, 800)] + [
        (mode, 0, horizon) for horizon in (800, 1600, 3200, 6400, 12800, 25600, 51200) for mode in (fastest, alternate)
    ]
    if index < len(initial):
        return initial[index]
    offset = index - len(initial)
    return (fastest, alternate)[offset % 2], 1 + offset // 6, (6400, 25600, 51200)[(offset // 2) % 3]


def budgeted_iterations(planned: int, seconds_left: float, seconds_per_iteration: float) -> int:
    """Reserve scoring/polish time and avoid discarding a long unfinished fit."""
    affordable = int(max(0, seconds_left - 20) / (seconds_per_iteration * 1.2))
    return max(
        (n for n in (128, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200) if n <= min(planned, affordable)),
        default=0,
    )


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = prepare(output, args.source_run, args.model)
    if args.prepare_only:
        print(json.dumps(metadata, indent=2), flush=True)
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    with acquire_device_lease(args.device):
        torch.use_deterministic_algorithms(True)
        fixture = load_file(str(output / "fixture.safetensors"), device=args.device)
        rank = int(metadata["rank"])
        shape = tuple(metadata["shape"])
        report_path = output / "report.json"
        source_hash = hash_file(Path(__file__))
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["runner_sha256"] != source_hash or report["fixture_sha256"] != metadata["fixture_sha256"]:
                raise ValueError("resume identity changed")
            if report["budget_seconds"] != args.seconds:
                raise ValueError("resume budget changed")
            if report["status"] == "complete":
                print(json.dumps(report["comparison"], indent=2), flush=True)
                return
        else:
            report = {
                "schema_version": 1,
                "status": "preparing",
                "fixture_sha256": metadata["fixture_sha256"],
                "runner_sha256": source_hash,
                "budget_seconds": args.seconds,
                "pilots": [],
                "records": [],
                "baseline": None,
                "incumbent": None,
                "next_candidate": 0,
                "protocol": {
                    "baseline": (
                        "800x5 ADMM BF16 factors/FP32 solves, cubic, ridge .03, "
                        "native orientation, two IEEE scale passes"
                    ),
                    "fixed": "full target, Fisher weights, rank 1152, seven removed columns, representation and score",
                    "selection": "lowest full-precision weighted residual error; raw error is diagnostic only",
                    "pilot": "64-step same-seed modes; fastest within 5% error energy of FP32 IEEE pilot",
                    "budget_scope": (
                        "Configured additional search seconds after baseline and speed pilots; "
                        "includes search scoring/checkpoint I/O"
                    ),
                    "no_quality_claim": (
                        "No model tuning, held-out data, perplexity, or full-model parity gate is evaluated"
                    ),
                },
                "environment": {
                    "torch": str(torch.__version__),
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(),
                    "pid": os.getpid(),
                },
            }

        def event(kind: str, **details: Any) -> None:
            payload = {"utc_epoch": time.time(), "kind": kind, **details}
            with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, allow_nan=False) + "\n")
            print(json.dumps(payload, allow_nan=False), flush=True)

        def persist() -> None:
            atomic_write_json(report_path, report)

        def record_state(name: str, state: dict[str, torch.Tensor], record: dict[str, Any]) -> dict[str, Any]:
            validate_state(state, shape, rank)
            record = {"name": name, **record, "completed_utc": time.time()}
            record["state_file"] = f"{name}.safetensors"
            record["state_sha256"] = save_state(output / record["state_file"], state)
            report["records"].append(record)
            if report["incumbent"] is None or record["weighted_error"] < report["incumbent"]["weighted_error"]:
                report["incumbent"] = record
                event("incumbent", **record)
            persist()
            return record

        def load_record(record: dict[str, Any]) -> dict[str, torch.Tensor]:
            path = output / record["state_file"]
            if hash_file(path) != record["state_sha256"]:
                raise ValueError("saved factor hash mismatch")
            return {key: value.float() for key, value in load_file(str(path), device=args.device).items()}

        if report["baseline"] is None:
            event("baseline_started")
            baseline_state, baseline = fit(
                fixture,
                rank,
                "bf16-ieee",
                0,
                800,
                progress=lambda n: event("baseline_progress", iterations=n),
            )
            report["baseline"] = record_state("baseline", baseline_state, baseline)
            retained = {key: fixture[f"retained_{key}"].float() for key in STATE_KEYS}
            # Historical tuned factors are a contextual comparator, never a free search seed.
            report["retained_context"] = score(fixture, retained)
            persist()
            event("baseline_complete", **baseline)
        completed_modes = {row["mode"] for row in report["pilots"]}
        for mode in MODES:
            if mode in completed_modes:
                continue
            fit(fixture, rank, mode, 102, 8)  # Untimed per-mode warmup, never a search candidate.
            _, pilot = fit(fixture, rank, mode, 101, 64)
            report["pilots"].append(pilot)
            persist()
            event("pilot", **pilot)
        fastest = choose_mode(report["pilots"])
        report["fastest_mode"] = fastest
        if "search_started_utc" not in report:
            report["search_started_utc"] = time.time()
            report["search_deadline_utc"] = time.time() + args.seconds
            report["status"] = "searching"
            persist()
            event("search_started", fastest_mode=fastest, deadline_utc=report["search_deadline_utc"])
        deadline = time.perf_counter() + max(0.0, report["search_deadline_utc"] - time.time())
        torch.cuda.reset_peak_memory_stats()
        # First isolate local-search headroom from additional joint ADMM work.
        if not report.get("initial_polish_complete"):
            state = load_record(report["baseline"])
            for pass_index in range(16):
                existing = next(
                    (row for row in report["records"] if row["name"] == f"baseline-polish-{pass_index:02d}"),
                    None,
                )
                if existing is not None:
                    state = load_record(existing)
                    continue
                if time.perf_counter() >= deadline - 10:
                    break
                started = time.perf_counter()
                state = polish(fixture, state, tabu=pass_index >= 8)
                record_state(
                    f"baseline-polish-{pass_index:02d}",
                    state,
                    {
                        "family": "local-search",
                        "tabu": pass_index >= 8,
                        "wall_seconds": time.perf_counter() - started,
                        **score(fixture, state),
                    },
                )
            report["initial_polish_complete"] = True
            persist()
        while time.perf_counter() < deadline - 10:
            index = report["next_candidate"]
            mode, seed, iterations = schedule(index, fastest)
            timings = [
                row["admm_seconds"] / row["outer_iterations"]
                for row in [*report["pilots"], *report["records"]]
                if row.get("mode") == mode and "admm_seconds" in row
            ]
            planned = iterations
            iterations = budgeted_iterations(iterations, deadline - time.perf_counter(), max(timings))
            if not iterations:
                break
            name = f"candidate-{index:03d}-{mode}-s{seed}-n{iterations}"
            event("candidate_started", name=name, mode=mode, seed=seed, iterations=iterations)
            try:
                state, result = fit(
                    fixture,
                    rank,
                    mode,
                    seed,
                    iterations,
                    deadline - 10,
                    progress=lambda n, name=name: event(
                        "candidate_progress",
                        name=name,
                        iterations=n,
                        remaining_seconds=max(0.0, deadline - time.perf_counter()),
                    ),
                )
            except BudgetExpired:
                report["unfinished_candidate"] = name
                event("candidate_budget_expired", name=name)
                break
            result["family"] = "joint-admm"
            result["planned_outer_iterations"] = planned
            record_state(name, state, result)
            # Polish only promising starts; no held-out information participates.
            if result["weighted_error"] <= report["incumbent"]["weighted_error"] * 1.01:
                for pass_index in range(4):
                    if time.perf_counter() >= deadline - 10:
                        break
                    started = time.perf_counter()
                    state = polish(fixture, state, tabu=pass_index >= 2)
                    record_state(
                        f"{name}-polish-{pass_index}",
                        state,
                        {
                            "family": "joint-admm-local-search",
                            "parent": name,
                            "wall_seconds": time.perf_counter() - started,
                            **score(fixture, state),
                        },
                    )
            report["next_candidate"] += 1
            persist()
        report["search_finished_utc"] = time.time()
        report["search_wall_seconds"] = report["search_finished_utc"] - report["search_started_utc"]
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["status"] = "validating"
        persist()
        best = load_record(report["incumbent"])
        baseline = load_record(report["baseline"])
        # A separate final FP64 GEMM audit does not select or improve the result.
        event("final_fp64_audit_started")
        exact_best, exact_baseline = score(fixture, best, double=True), score(fixture, baseline, double=True)
        for state in (best, baseline):
            validate_state(state, shape, rank)
        report["comparison"] = {
            "baseline": exact_baseline,
            "best": exact_best,
            "best_name": report["incumbent"]["name"],
            "error_reduction_percent": 100 * (1 - exact_best["weighted_error"] / exact_baseline["weighted_error"]),
            "nrmse_reduction_percent": 100 * (1 - exact_best["weighted_nrmse"] / exact_baseline["weighted_nrmse"]),
            "search_seconds": report["search_wall_seconds"],
            "baseline_seconds": report["baseline"]["wall_seconds"],
            "fp64_audit_relative_error_delta": abs(
                exact_best["weighted_error"] / report["incumbent"]["weighted_error"] - 1
            ),
        }
        save_state(output / "best.safetensors", best)
        report["best_sha256"] = hash_file(output / "best.safetensors")
        report["status"] = "complete"
        persist()
        event("complete", **report["comparison"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/066/hour-fit")
    parser.add_argument("--source-run", type=Path, default=ROOT / "evidence/041/experiment040-production-replay")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home()
        / (f".cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/{REVISION}/model.safetensors"),
    )
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or not math.isfinite(args.seconds):
        parser.error("seconds must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
