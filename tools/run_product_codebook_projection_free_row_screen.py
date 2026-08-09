"""Screen shape-specific product-codebook free-row ladders by projection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from safetensors import safe_open

from nanoquant.infrastructure.io_utils import atomic_write_json

DEFAULT_BLOCKS = (0, 12, 24)


@dataclass(frozen=True, slots=True)
class ProjectionSweepConfig:
    projection: str
    outlier_layer: str
    baseline_rank: int
    candidate_rank: int
    free_row_counts: tuple[int, ...]
    transpose_matrix: bool = False


PROJECTION_CONFIGS = {
    "gate": ProjectionSweepConfig(
        "gate", "mlp.gate_proj", 970, 1152, (576, 608, 640, 672, 704), True
    ),
    "up": ProjectionSweepConfig(
        "up", "mlp.up_proj", 970, 1152, (576, 608, 640, 672, 704), True
    ),
    "q": ProjectionSweepConfig(
        "q", "self_attn.attn_qkv", 522, 576, (256, 288, 320, 352)
    ),
    "k": ProjectionSweepConfig(
        "k", "self_attn.attn_qkv", 191, 224, (32, 64, 96, 128)
    ),
    "v": ProjectionSweepConfig(
        "v", "self_attn.attn_qkv", 191, 224, (32, 64, 96, 128)
    ),
    "o": ProjectionSweepConfig(
        "o", "self_attn.o_proj", 522, 576, (256, 288, 320, 352), True
    ),
}
DEFAULT_PROJECTIONS = tuple(PROJECTION_CONFIGS)


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("list must not be empty")
    return result


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _parse_csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("integer list contains a non-integer") from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_fixed_outliers(
    logical_weights: Path,
    block: int,
    config: ProjectionSweepConfig,
) -> tuple[int, ...]:
    shard = logical_weights / f"block-{block:05d}.safetensors"
    key = f"blocks.{block}.{config.outlier_layer}.outlier_indices"
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise ValueError(f"missing exact outlier indices: {shard}:{key}")
        indices = tuple(int(value) for value in handle.get_tensor(key).tolist())
    if not indices:
        raise ValueError(f"empty exact outlier indices: {shard}:{key}")
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError(f"invalid exact outlier indices: {shard}:{key}")
    return indices


def _candidate_key(free_rows: int, outlier_count: int) -> str:
    return f"right_product_codebook_k16_free{free_rows}_outliers{outlier_count}"


def build_probe_command(
    *,
    python: Path,
    probe: Path,
    model: Path,
    calibration_state: Path,
    output: Path,
    block: int,
    config: ProjectionSweepConfig,
    fixed_outliers: tuple[int, ...],
    outer_iterations: int,
    seed: int,
    device: str,
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
        config.projection,
        "--baseline-rank",
        str(config.baseline_rank),
        "--candidate-rank",
        str(config.candidate_rank),
        "--candidate-outlier-columns",
        str(len(fixed_outliers)),
        "--fixed-outlier-indices",
        ",".join(str(index) for index in fixed_outliers),
        "--right-free-row-counts",
        ",".join(str(count) for count in config.free_row_counts),
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
    if config.transpose_matrix:
        command.append("--transpose-matrix")
    return command


def _summarize_probe(
    output: Path,
    config: ProjectionSweepConfig,
    outlier_count: int,
) -> dict[str, Any]:
    payload = json.loads(output.read_text(encoding="utf-8"))
    baseline = payload["results"]["free_words"]
    candidates: list[dict[str, Any]] = []
    for free_rows in config.free_row_counts:
        key = _candidate_key(free_rows, outlier_count)
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
            }
        )
    improving = [
        item for item in candidates if item["weighted_rmse_change_fraction"] < 0
    ]
    smallest_improving = (
        min(improving, key=lambda item: item["right_free_rows"])
        if improving
        else None
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
        "candidate_rank": config.candidate_rank,
        "transpose_matrix": config.transpose_matrix,
        "candidates": candidates,
        "smallest_improving_free_rows": (
            None
            if smallest_improving is None
            else smallest_improving["right_free_rows"]
        ),
        "best_free_rows": best["right_free_rows"],
        "best_weighted_rmse_change_fraction": best[
            "weighted_rmse_change_fraction"
        ],
    }


def run(args: argparse.Namespace) -> int:
    if len(args.blocks) != len(set(args.blocks)) or any(
        block < 0 for block in args.blocks
    ):
        raise ValueError("blocks must be unique and non-negative")
    if len(args.projections) != len(set(args.projections)) or any(
        projection not in PROJECTION_CONFIGS for projection in args.projections
    ):
        raise ValueError("projections must be unique known projection names")
    if args.outer_iterations <= 0:
        raise ValueError("outer iterations must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    previous: dict[str, Any] = {}
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = previous.get("results", {})
    started_at = previous.get("started_at", _utc_now())
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at,
        "updated_at": _utc_now(),
        "blocks": list(args.blocks),
        "projections": list(args.projections),
        "outer_iterations": args.outer_iterations,
        "seed": args.seed,
        "model": str(args.model.resolve()),
        "calibration_state": str(args.calibration_state.resolve()),
        "logical_weights": str(args.logical_weights.resolve()),
        "down_sweep_summary": str(args.down_sweep_summary.resolve()),
        "results": results,
    }
    atomic_write_json(summary_path, summary)

    repo_root = Path(__file__).resolve().parents[1]
    probe = Path(__file__).resolve().with_name("probe_sign_word_codebook.py")
    python = Path(sys.executable).resolve()
    try:
        for block in args.blocks:
            block_results = results.setdefault(str(block), {})
            for projection in args.projections:
                config = PROJECTION_CONFIGS[projection]
                output = args.output_dir / f"block-{block:02d}-{projection}.json"
                if projection in block_results and output.is_file():
                    print(
                        f"skipping completed block={block} projection={projection}",
                        flush=True,
                    )
                    continue
                fixed_outliers = _read_fixed_outliers(
                    args.logical_weights,
                    block,
                    config,
                )
                summary["current_block"] = block
                summary["current_projection"] = projection
                summary["current_output"] = str(output.resolve())
                summary["updated_at"] = _utc_now()
                atomic_write_json(summary_path, summary)
                print(
                    f"starting block={block} projection={projection} "
                    f"rank={config.candidate_rank} "
                    f"free_rows={config.free_row_counts} outliers={fixed_outliers}",
                    flush=True,
                )
                command = build_probe_command(
                    python=python,
                    probe=probe,
                    model=args.model.resolve(),
                    calibration_state=args.calibration_state.resolve(),
                    output=output.resolve(),
                    block=block,
                    config=config,
                    fixed_outliers=fixed_outliers,
                    outer_iterations=args.outer_iterations,
                    seed=args.seed,
                    device=args.device,
                )
                subprocess.run(command, cwd=repo_root, check=True)
                block_results[projection] = _summarize_probe(
                    output,
                    config,
                    len(fixed_outliers),
                )
                completed = [
                    f"{block_key}:{projection_key}"
                    for block_key, projections in results.items()
                    for projection_key in projections
                    if int(block_key) in args.blocks
                    and projection_key in args.projections
                ]
                summary["completed_arms"] = completed
                summary["updated_at"] = _utc_now()
                atomic_write_json(summary_path, summary)
                print(
                    f"completed block={block} projection={projection} "
                    f"smallest_improving_free_rows="
                    f"{block_results[projection]['smallest_improving_free_rows']}",
                    flush=True,
                )
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["updated_at"] = _utc_now()
        atomic_write_json(summary_path, summary)
        raise

    summary["status"] = "completed"
    summary["completed_arms"] = [
        f"{block}:{projection}"
        for block in args.blocks
        for projection in args.projections
        if projection in results.get(str(block), {})
        and (args.output_dir / f"block-{block:02d}-{projection}.json").is_file()
    ]
    summary.pop("current_block", None)
    summary.pop("current_projection", None)
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
    parser.add_argument("--down-sweep-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=_parse_ints, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--projections",
        type=_parse_csv,
        default=DEFAULT_PROJECTIONS,
    )
    parser.add_argument("--outer-iterations", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
