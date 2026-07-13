"""Atomic safetensors checkpoints for resumable online Fisher calibration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.application.calibration import (
    CausalOnlineCalibrationState,
    CausalOnlineLayerSnapshot,
    OnlineAccumulatorSnapshot,
)


def save_causal_calibration_state(path: str | Path, state: CausalOnlineCalibrationState) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    layers = []
    for index, layer in enumerate(state.layers):
        layer_manifest: dict[str, object] = {"path": layer.path}
        for side, snapshot in (("inputs", layer.inputs), ("outputs", layer.outputs)):
            prefix = f"layer_{index}.{side}"
            tensors[f"{prefix}.total"] = snapshot.total.detach().cpu().contiguous()
            tensors[f"{prefix}.global_max"] = (
                torch.empty(0, dtype=torch.float32)
                if snapshot.global_max is None
                else snapshot.global_max.detach().cpu().reshape(1).contiguous()
            )
            layer_manifest[side] = {
                "batch_count": snapshot.batch_count,
                "pre_scale": snapshot.pre_scale,
                "post_scale": snapshot.post_scale,
                "percentile": snapshot.percentile,
            }
        layers.append(layer_manifest)
    manifest = {
        "schema_version": 2,
        "algorithm_version": state.algorithm_version,
        "sample_count": state.sample_count,
        "layers": layers,
    }
    tensor_tmp = root / "state.safetensors.tmp"
    manifest_tmp = root / "manifest.json.tmp"
    save_file(tensors, tensor_tmp)
    manifest_tmp.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tensor_tmp, root / "state.safetensors")
    os.replace(manifest_tmp, root / "manifest.json")


def load_causal_calibration_state(path: str | Path) -> CausalOnlineCalibrationState:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported causal calibration checkpoint schema")
    layers = []
    with safe_open(root / "state.safetensors", framework="pt", device="cpu") as handle:
        for index, layer in enumerate(manifest["layers"]):
            snapshots = []
            for side in ("inputs", "outputs"):
                prefix = f"layer_{index}.{side}"
                values = layer[side]
                global_max = handle.get_tensor(f"{prefix}.global_max")
                snapshots.append(
                    OnlineAccumulatorSnapshot(
                        handle.get_tensor(f"{prefix}.total"),
                        None if global_max.numel() == 0 else global_max.reshape(()),
                        int(values["batch_count"]),
                        float(values["pre_scale"]),
                        float(values["post_scale"]),
                        float(values["percentile"]),
                    )
                )
            layers.append(CausalOnlineLayerSnapshot(str(layer["path"]), snapshots[0], snapshots[1]))
    state = CausalOnlineCalibrationState(
        tuple(layers),
        int(manifest["sample_count"]),
        int(manifest.get("algorithm_version", 1)),
    )
    if state.sample_count != int(manifest["sample_count"]):
        raise ValueError("causal calibration checkpoint sample count disagrees with tensors")
    return state
