"""Hash-validated replacement overlays for existing factorized payload terms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.config.codec import to_dict
from nanoquant.domain.models import ArtifactRef

from .io_utils import hash_file
from .safetensors_io import SAFETENSORS


@dataclass(frozen=True, slots=True)
class FactorizedComponentOverlay:
    root: Path
    manifest: dict[str, Any]
    tensors: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class AppliedFactorizedComponentOverlay:
    tensor_count: int
    layer_count: int
    replaced_bytes: int
    replacement_bytes: int


def load_factorized_component_overlay(
    root: str | Path,
    *,
    frozen_identity: dict[str, str],
    global_tuning: ArtifactRef | None,
) -> FactorizedComponentOverlay:
    root = Path(root)
    manifest_path = root / "manifest.json"
    tensor_path = root / "components.safetensors"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise ValueError("factorized component overlay is incomplete")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("factorized component overlay manifest must be an object")
    manifest = cast(dict[str, Any], payload)
    inventory = manifest.get("tensors")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("semantics") != "replace-existing-factorized-components"
        or manifest.get("frozen_identity") != frozen_identity
        or manifest.get("global_tuning") != (
            None if global_tuning is None else to_dict(global_tuning)
        )
        or manifest.get("tensor_sha256") != hash_file(tensor_path)
        or not isinstance(inventory, dict)
        or manifest.get("tensor_count") != len(inventory)
        or manifest.get("payload_byte_delta") != 0
        or manifest.get("replaced_payload_bytes")
        != manifest.get("replacement_payload_bytes")
    ):
        raise ValueError("factorized component overlay identity or replacement contract is invalid")
    tensors = SAFETENSORS.load(tensor_path)
    if set(tensors) != set(inventory) or any(
        value.ndim not in {1, 2}
        or (
            not value.is_floating_point()
            and name.rsplit(".", maxsplit=1)[-1] != "outlier_values"
        )
        or not torch.isfinite(value).all()
        or list(value.shape) != inventory[name].get("shape")
        or str(value.dtype).removeprefix("torch.") != inventory[name].get("dtype")
        for name, value in tensors.items()
    ):
        raise ValueError("factorized component overlay tensor inventory is invalid")
    return FactorizedComponentOverlay(root, manifest, tensors)


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("model does not expose decoder blocks for a component overlay")
    return layers


def _module_at_path(block: nn.Module, path: str) -> nn.Module:
    current = block
    for part in path.split("."):
        child = current[part] if isinstance(current, nn.ModuleDict) else getattr(current, part, None)
        if not isinstance(child, nn.Module):
            raise KeyError(f"component overlay module path not found: {path}")
        current = child
    return current


def apply_factorized_component_overlay(
    model: nn.Module,
    overlay: FactorizedComponentOverlay,
) -> AppliedFactorizedComponentOverlay:
    grouped: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    prefix = "model.layers."
    for name, value in overlay.tensors.items():
        if not name.startswith(prefix):
            raise ValueError(f"component overlay tensor has a non-model name: {name}")
        logical = name.removeprefix(prefix)
        block_text, remainder = logical.split(".", maxsplit=1)
        path, component = remainder.rsplit(".", maxsplit=1)
        if component not in {
            "scale_pre",
            "scale_post",
            "outlier_values",
            "patch_left",
            "patch_right",
        }:
            raise ValueError(f"component overlay tensor has an unsupported term: {name}")
        try:
            block_index = int(block_text)
        except ValueError as error:
            raise ValueError(f"component overlay block is not an integer: {name}") from error
        components = grouped.setdefault((block_index, path), {})
        if component in components:
            raise ValueError(f"component overlay term is duplicated: {name}")
        components[component] = value

    decoder = _decoder_layers(model)
    replaced_bytes = replacement_bytes = 0
    for (block_index, path), replacements in grouped.items():
        if block_index < 0 or block_index >= len(decoder):
            raise ValueError(f"component overlay block is outside the model: {block_index}")
        module = _module_at_path(decoder[block_index], path)
        if not isinstance(module, FactorizedReferenceLinear):
            raise TypeError(f"component overlay target is not factorized: {block_index}:{path}")
        expected = {"scale_pre", "scale_post"}
        expected.update(
            name
            for name in ("outlier_values", "patch_left", "patch_right")
            if isinstance(getattr(module, name), torch.Tensor)
        )
        if set(replacements) != expected:
            raise ValueError(f"component overlay terms are incomplete for {block_index}:{path}")
        with torch.no_grad():
            for component, replacement in replacements.items():
                current = getattr(module, component)
                if (
                    not isinstance(current, torch.Tensor)
                    or current.shape != replacement.shape
                    or current.dtype != replacement.dtype
                ):
                    raise ValueError(
                        f"component overlay term differs from the frozen payload: "
                        f"{block_index}:{path}.{component}"
                    )
                size = current.numel() * current.element_size()
                replaced_bytes += size
                replacement_bytes += replacement.numel() * replacement.element_size()
                current.copy_(replacement.to(device=current.device))
    if replaced_bytes != overlay.manifest["replaced_payload_bytes"]:
        raise ValueError("component overlay replaced-byte count differs from its manifest")
    return AppliedFactorizedComponentOverlay(
        len(overlay.tensors),
        len(grouped),
        replaced_bytes,
        replacement_bytes,
    )


__all__ = [
    "AppliedFactorizedComponentOverlay",
    "FactorizedComponentOverlay",
    "apply_factorized_component_overlay",
    "load_factorized_component_overlay",
]
