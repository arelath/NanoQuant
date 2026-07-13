"""Compare a rewrite quantization plan with ranks retained in a legacy checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch


def compare_quantization_plan(plan_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    checkpoint_value = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint_value, dict):
        raise TypeError("legacy checkpoint must contain a tensor state dictionary")
    checkpoint = cast(dict[str, Any], checkpoint_value)
    rows: list[dict[str, Any]] = []
    for block in plan["blocks"]:
        for layer in block["layers"]:
            block_index = int(layer["layer"]["block"]["index"])
            path = str(layer["layer"]["path"])
            prefix = f"model.layers.{block_index}.{path}"
            shape = checkpoint.get(f"{prefix}.U_shape")
            if not isinstance(shape, torch.Tensor) or shape.numel() != 2:
                raise ValueError(f"legacy checkpoint is missing a valid rank shape: {prefix}")
            salient = checkpoint.get(f"{prefix}.salient_idx")
            if salient is not None and not isinstance(salient, torch.Tensor):
                raise ValueError(f"legacy checkpoint has invalid salient indices: {prefix}")
            legacy_rank = int(shape.reshape(-1)[1])
            planned_rank = int(layer["rank"])
            legacy_outlier_count = 0 if salient is None else salient.numel()
            planned_outlier_count = int(layer["outliers"]["count"])
            rows.append(
                {
                    "layer": prefix,
                    "block": block_index,
                    "path": path,
                    "legacy_rank": legacy_rank,
                    "planned_rank": planned_rank,
                    "rank_delta": planned_rank - legacy_rank,
                    "legacy_outlier_count": legacy_outlier_count,
                    "planned_outlier_count": planned_outlier_count,
                    "outlier_count_delta": planned_outlier_count - legacy_outlier_count,
                }
            )
    return {
        "schema_version": 1,
        "plan": str(plan_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "layer_count": len(rows),
        "rank_mismatch_count": sum(row["rank_delta"] != 0 for row in rows),
        "legacy_rank_sum": sum(row["legacy_rank"] for row in rows),
        "planned_rank_sum": sum(row["planned_rank"] for row in rows),
        "absolute_rank_delta_sum": sum(abs(row["rank_delta"]) for row in rows),
        "outlier_count_mismatch_count": sum(row["outlier_count_delta"] != 0 for row in rows),
        "legacy_outlier_count": sum(row["legacy_outlier_count"] for row in rows),
        "planned_outlier_count": sum(row["planned_outlier_count"] for row in rows),
        "layers": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare_quantization_plan(args.plan, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
