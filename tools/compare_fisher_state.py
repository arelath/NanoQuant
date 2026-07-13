"""Compare a resumable Fisher state with retained per-layer importance vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from nanoquant.application.calibration import materialize_causal_online_state
from nanoquant.infrastructure.calibration_checkpoint import load_causal_calibration_state


def _reference_key(path: str, kind: str) -> str:
    if not path.startswith("block."):
        raise ValueError(f"unsupported Fisher layer path: {path}")
    prefix = "i" if kind == "input" else "o"
    return f"{prefix}.{path.replace('block.', 'model.layers.', 1)}"


def _tensor_metrics(current: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    current = current.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    reference = reference.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if current.shape != reference.shape:
        raise ValueError(
            f"importance shape mismatch: current={tuple(current.shape)}, reference={tuple(reference.shape)}"
        )
    if not torch.isfinite(current).all() or not torch.isfinite(reference).all():
        raise ValueError("importance comparison requires finite tensors")
    absolute = (current - reference).abs()
    relative = absolute / reference.abs().clamp_min(1e-12)
    return {
        "element_count": current.numel(),
        "exact": torch.equal(current, reference),
        "mean_absolute_error": float(absolute.mean()),
        "max_absolute_error": float(absolute.max()),
        "mean_relative_error": float(relative.mean()),
        "max_relative_error": float(relative.max()),
        "l1_relative_error": float(absolute.sum() / reference.abs().sum().clamp_min(1e-12)),
    }


def compare_fisher_state(state_path: Path, reference_path: Path, shrinkage: float) -> dict[str, Any]:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    state_manifest = json.loads((state_path / "manifest.json").read_text(encoding="utf-8"))
    state = load_causal_calibration_state(state_path)
    materialized = materialize_causal_online_state(state, shrinkage=shrinkage)
    layer_rows: list[dict[str, Any]] = [{"path": item.path} for item in materialized]
    expected_keys = {
        _reference_key(item.path, kind)
        for item in materialized
        for kind in ("input", "output")
    }
    summaries: dict[str, Any] = {}
    with safe_open(reference_path, framework="pt", device="cpu") as reference:
        available_keys = set(reference.keys())
        missing_keys = sorted(expected_keys - available_keys)
        if missing_keys:
            raise ValueError(f"reference Fisher statistics are missing keys: {missing_keys[:5]}")
        for kind, attribute in (("input", "input_importance"), ("output", "output_importance")):
            relative_values: list[torch.Tensor] = []
            absolute_values: list[torch.Tensor] = []
            kind_rows: list[dict[str, Any]] = []
            for item, row in zip(materialized, layer_rows, strict=True):
                current = getattr(item, attribute).detach().to(device="cpu", dtype=torch.float32).reshape(-1)
                retained = reference.get_tensor(_reference_key(item.path, kind)).float().reshape(-1)
                metrics = _tensor_metrics(current, retained)
                row[kind] = metrics
                kind_rows.append(metrics)
                absolute = (current - retained).abs()
                absolute_values.append(absolute)
                relative_values.append(absolute / retained.abs().clamp_min(1e-12))
            relative = torch.cat(relative_values)
            absolute = torch.cat(absolute_values)
            summaries[kind] = {
                "element_count": relative.numel(),
                "exact_layer_count": sum(bool(row["exact"]) for row in kind_rows),
                "mean_absolute_error": float(absolute.mean()),
                "max_absolute_error": float(absolute.max()),
                "element_mean_relative_error": float(relative.mean()),
                "layer_mean_relative_error": sum(float(row["mean_relative_error"]) for row in kind_rows)
                / len(kind_rows),
                "layer_mean_l1_relative_error": sum(float(row["l1_relative_error"]) for row in kind_rows)
                / len(kind_rows),
                "max_relative_error": float(relative.max()),
            }
    return {
        "schema_version": 1,
        "state": str(state_path.resolve()),
        "reference": str(reference_path.resolve()),
        "state_schema_version": int(state_manifest["schema_version"]),
        "state_algorithm_version": state.algorithm_version,
        "sample_count": state.sample_count,
        "shrinkage": shrinkage,
        "layer_count": len(materialized),
        "reference_key_count": len(available_keys),
        "unexpected_reference_keys": sorted(available_keys - expected_keys),
        "input": summaries["input"],
        "output": summaries["output"],
        "layers": layer_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shrinkage", type=float, default=0.6)
    args = parser.parse_args()
    payload = compare_fisher_state(args.state, args.reference, args.shrinkage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
