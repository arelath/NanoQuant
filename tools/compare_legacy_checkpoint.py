"""Compare retained legacy packed factors with an immutable rewrite run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

import torch

from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.legacy_checkpoint import unpack_binary_gemv


def _indices(value: object) -> set[int]:
    if not isinstance(value, torch.Tensor):
        return set()
    return {int(item) for item in value.reshape(-1).tolist()}


def _agreement(current: torch.Tensor, legacy: torch.Tensor) -> float | None:
    if current.shape != legacy.shape:
        return None
    return float((current.to(device="cpu", dtype=torch.int8) == legacy.to(torch.int8)).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = load_frozen_run(
        args.run_output,
        args.snapshot,
        source_name="google/gemma-3-1b-it",
        revision=args.revision,
        device="cpu",
        backend="factorized",
        use_global_tuning=False,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise TypeError("legacy checkpoint must contain a tensor state dictionary")
    legacy = cast(dict[str, Any], state)
    modules = dict(loaded.model.named_modules())
    rows: list[dict[str, Any]] = []
    for prefix in sorted(key.removesuffix(".U_packed") for key in legacy if key.endswith(".U_packed")):
        module = modules.get(prefix)
        if not isinstance(module, FactorizedReferenceLinear):
            raise TypeError(f"rewrite run is missing factorized layer: {prefix}")
        left_shape = tuple(int(item) for item in legacy[f"{prefix}.U_shape"].tolist())
        right_shape = tuple(int(item) for item in legacy[f"{prefix}.V_shape"].tolist())
        if len(left_shape) != 2 or len(right_shape) != 2:
            raise ValueError(f"legacy factor shape is invalid: {prefix}")
        legacy_left = unpack_binary_gemv(legacy[f"{prefix}.U_packed"], left_shape)
        legacy_right = unpack_binary_gemv(legacy[f"{prefix}.V_packed"], right_shape)
        legacy_outliers = _indices(legacy.get(f"{prefix}.salient_idx"))
        current_outliers = _indices(module.outlier_indices)
        match = re.search(r"\.layers\.(\d+)\.", prefix)
        rows.append(
            {
                "layer": prefix,
                "block": None if match is None else int(match.group(1)),
                "legacy_rank": left_shape[1],
                "rewrite_rank": int(module.left_binary.shape[1]),
                "rank_delta": int(module.left_binary.shape[1]) - left_shape[1],
                "legacy_outliers": sorted(legacy_outliers),
                "rewrite_outliers": sorted(current_outliers),
                "outlier_intersection": len(legacy_outliers & current_outliers),
                "outlier_union": len(legacy_outliers | current_outliers),
                "left_sign_agreement": _agreement(module.left_binary, legacy_left),
                "right_sign_agreement": _agreement(module.right_binary, legacy_right),
            }
        )
    same_rank = [row for row in rows if row["rank_delta"] == 0]
    payload = {
        "schema_version": 1,
        "run_output": str(args.run_output.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "layer_count": len(rows),
        "rank_mismatch_count": sum(row["rank_delta"] != 0 for row in rows),
        "outlier_mismatch_count": sum(row["legacy_outliers"] != row["rewrite_outliers"] for row in rows),
        "legacy_rank_sum": sum(row["legacy_rank"] for row in rows),
        "rewrite_rank_sum": sum(row["rewrite_rank"] for row in rows),
        "same_rank_mean_left_sign_agreement": sum(cast(float, row["left_sign_agreement"]) for row in same_rank)
        / len(same_rank),
        "same_rank_mean_right_sign_agreement": sum(cast(float, row["right_sign_agreement"]) for row in same_rank)
        / len(same_rank),
        "layers": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
