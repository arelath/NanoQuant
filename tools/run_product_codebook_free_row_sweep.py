"""Run a resumable per-layer product-codebook free-row reconstruction sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from safetensors import safe_open

from nanoquant.infrastructure.io_utils import atomic_write_json

DEFAULT_FREE_ROW_COUNTS = (576, 608, 640, 672, 704)
DEFAULT_BLOCKS = tuple(range(26))


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_fixed_outliers(logical_weights: Path, block: int) -> tuple[int, ...]:
    shard = logical_weights / f"block-{block:05d}.safetensors"
    key = f"blocks.{block}.mlp.down_proj.outlier_indices"
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise ValueError(f"missing exact outlier indices: {shard}:{key}")
        indices = tuple(int(value) for value in handle.get_tensor(key).tolist())
    if not indices:
        raise ValueError(f"empty exact outlier index tensor: {shard}:{key}")
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError(f"invalid exact outlier indices: {shard}:{key}")
    return indices


def _candidate_key(
    free_rows: int,
    outlier_count: int,
    *,
    adaptive_free_rows: bool,
) -> str:
    adaptive = "_adaptive" if adaptive_free_rows else ""
    return (
        f"right_product_codebook_k16_free{free_rows}{adaptive}"
        f"_outliers{outlier_count}"
    )


def build_probe_command(
    *,
    python: Path,
    probe: Path,
    model: Path,
    calibration_state: Path,
    output: Path,
    block: int,
    fixed_outliers: tuple[int, ...],
    free_row_counts: tuple[int, ...],
    baseline_rank: int,
    candidate_rank: int,
    outer_iterations: int,
    seed: int,
    device: str,
    adaptive_free_rows: bool = False,
    adaptive_free_row_refit_passes: int = 4,
    codebook_warmup_fraction: float = 0.0,
) -> list[str]:
    command = [
        str(python),
        str(probe),
        "--model",
        str(model),
        "--calibration-state",
        str(calibration_state),
        "--output",
        str(output),
        "--block",
        str(block),
        "--projection",
        "down",
        "--baseline-rank",
        str(baseline_rank),
        "--candidate-rank",
        str(candidate_rank),
        "--candidate-outlier-columns",
        str(len(fixed_outliers)),
        "--fixed-outlier-indices",
        ",".join(str(index) for index in fixed_outliers),
        "--right-free-row-counts",
        ",".join(str(count) for count in free_row_counts),
        "--index-widths",
        "16",
        "--outer-iterations",
        str(outer_iterations),
        "--codebook-mode",
        "product-right",
        "--assignment-batch-words",
        "8192",
        "--binary-search",
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if adaptive_free_rows:
        command.extend(
            (
                "--adaptive-free-rows",
                "--adaptive-free-row-refit-passes",
                str(adaptive_free_row_refit_passes),
                "--codebook-warmup-fraction",
                str(codebook_warmup_fraction),
            )
        )
    return command


def _summarize_probe(
    output: Path,
    free_row_counts: tuple[int, ...],
    outlier_count: int,
    *,
    adaptive_free_rows: bool,
) -> dict[str, Any]:
    payload = json.loads(output.read_text(encoding="utf-8"))
    baseline = payload["results"]["free_words"]
    candidates: list[dict[str, Any]] = []
    for free_rows in free_row_counts:
        key = _candidate_key(
            free_rows,
            outlier_count,
            adaptive_free_rows=adaptive_free_rows,
        )
        result = payload["results"][key]
        candidates.append(
            {
                "key": key,
                "right_free_rows": free_rows,
                "actual_bpw": result["actual_bpw"],
                "weighted_normalized_rmse": result["metrics"][
                    "weighted_normalized_rmse"
                ],
                "weighted_rmse_change_fraction": result[
                    "comparison_to_free_words"
                ]["weighted_rmse_change_fraction"],
                "wall_seconds": result["wall_seconds"],
                "adaptive_free_rows": result.get("adaptive_free_rows"),
            }
        )
    best = min(candidates, key=lambda item: item["weighted_normalized_rmse"])
    return {
        "baseline": {
            "rank": baseline["rank"],
            "actual_bpw": baseline["actual_bpw"],
            "weighted_normalized_rmse": baseline["metrics"][
                "weighted_normalized_rmse"
            ],
            "wall_seconds": baseline["wall_seconds"],
        },
        "candidates": candidates,
        "best_right_free_rows": best["right_free_rows"],
        "best_weighted_rmse_change_fraction": best[
            "weighted_rmse_change_fraction"
        ],
    }


def run(args: argparse.Namespace) -> int:
    blocks = args.blocks
    free_row_counts = args.free_row_counts
    if len(blocks) != len(set(blocks)) or any(block < 0 for block in blocks):
        raise ValueError("blocks must be unique and non-negative")
    if (
        len(free_row_counts) != len(set(free_row_counts))
        or any(count <= 0 or count % 32 for count in free_row_counts)
        or any(count >= args.candidate_rank for count in free_row_counts)
    ):
        raise ValueError(
            "free-row counts must be unique positive multiples of 32 below rank"
        )
    if args.baseline_rank <= 0 or args.candidate_rank <= 0:
        raise ValueError("ranks must be positive")
    if args.outer_iterations <= 0:
        raise ValueError("outer iterations must be positive")
    if args.adaptive_free_rows and not 0 < args.codebook_warmup_fraction < 1:
        raise ValueError("adaptive free rows require a positive warmup fraction")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    layer_results: dict[str, Any] = {}
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        layer_results.update(previous.get("layers", {}))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "blocks": list(blocks),
        "free_row_counts": list(free_row_counts),
        "baseline_rank": args.baseline_rank,
        "candidate_rank": args.candidate_rank,
        "outer_iterations": args.outer_iterations,
        "seed": args.seed,
        "adaptive_free_rows": args.adaptive_free_rows,
        "adaptive_free_row_refit_passes": (
            args.adaptive_free_row_refit_passes
        ),
        "codebook_warmup_fraction": args.codebook_warmup_fraction,
        "model": str(args.model.resolve()),
        "calibration_state": str(args.calibration_state.resolve()),
        "logical_weights": str(args.logical_weights.resolve()),
        "layers": layer_results,
    }
    atomic_write_json(summary_path, summary)

    repo_root = Path(__file__).resolve().parents[1]
    probe = Path(__file__).resolve().with_name("probe_sign_word_codebook.py")
    python = Path(sys.executable).resolve()
    try:
        for block in blocks:
            fixed_outliers = _read_fixed_outliers(args.logical_weights, block)
            output = args.output_dir / f"block-{block:02d}.json"
            summary["current_block"] = block
            summary["current_output"] = str(output.resolve())
            summary["updated_at"] = _utc_now()
            atomic_write_json(summary_path, summary)
            print(
                f"starting block={block} free_rows={free_row_counts} "
                f"outliers={fixed_outliers}",
                flush=True,
            )
            command = build_probe_command(
                python=python,
                probe=probe,
                model=args.model.resolve(),
                calibration_state=args.calibration_state.resolve(),
                output=output.resolve(),
                block=block,
                fixed_outliers=fixed_outliers,
                free_row_counts=free_row_counts,
                baseline_rank=args.baseline_rank,
                candidate_rank=args.candidate_rank,
                outer_iterations=args.outer_iterations,
                seed=args.seed,
                device=args.device,
                adaptive_free_rows=args.adaptive_free_rows,
                adaptive_free_row_refit_passes=(
                    args.adaptive_free_row_refit_passes
                ),
                codebook_warmup_fraction=args.codebook_warmup_fraction,
            )
            subprocess.run(command, cwd=repo_root, check=True)
            layer_results[str(block)] = _summarize_probe(
                output,
                free_row_counts,
                len(fixed_outliers),
                adaptive_free_rows=args.adaptive_free_rows,
            )
            summary["completed_blocks"] = sorted(
                int(item) for item in layer_results
            )
            summary["updated_at"] = _utc_now()
            atomic_write_json(summary_path, summary)
            print(
                f"completed block={block} best_free_rows="
                f"{layer_results[str(block)]['best_right_free_rows']}",
                flush=True,
            )
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["updated_at"] = _utc_now()
        atomic_write_json(summary_path, summary)
        raise

    summary["status"] = "completed"
    summary.pop("current_block", None)
    summary.pop("current_output", None)
    summary["completed_at"] = _utc_now()
    summary["updated_at"] = summary["completed_at"]
    atomic_write_json(summary_path, summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--logical-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=_parse_ints, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--free-row-counts",
        type=_parse_ints,
        default=DEFAULT_FREE_ROW_COUNTS,
    )
    parser.add_argument("--baseline-rank", type=int, default=970)
    parser.add_argument("--candidate-rank", type=int, default=1152)
    parser.add_argument("--outer-iterations", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adaptive-free-rows", action="store_true")
    parser.add_argument("--adaptive-free-row-refit-passes", type=int, default=4)
    parser.add_argument("--codebook-warmup-fraction", type=float, default=1 / 12)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
