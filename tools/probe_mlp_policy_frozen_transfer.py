"""Evaluate an exported dense MLP policy inside a retained frozen run."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from evaluate_wikitext import _evaluate as _evaluate_wikitext
from evaluate_wikitext import _protocol_tokens
from torch import nn

from nanoquant.application.layers import FrozenReferenceLinear
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.io_utils import (
    atomic_write_json,
    hash_file,
)
from nanoquant.infrastructure.safetensors_io import SAFETENSORS
from nanoquant.kl_budget_workflow import _token_hash

MODEL_SOURCE = "google/gemma-3-1b-it"
PINNED_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
MLP_PATHS = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument(
        "--additional-overlay",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-global-tuning", action="store_true")
    return parser


def _load_overlay(
    overlay: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    manifest_path = overlay / "manifest.json"
    tensor_path = overlay / "weights.safetensors"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise ValueError("MLP policy overlay is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("MLP policy overlay manifest must be an object")
    tensor_inventory = manifest.get("tensors")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("tensor_sha256") != hash_file(tensor_path)
        or not isinstance(tensor_inventory, dict)
        or manifest.get("layer_count") != len(tensor_inventory)
    ):
        raise ValueError("MLP policy overlay identity or hash is invalid")
    tensors = SAFETENSORS.load(tensor_path)
    if set(tensors) != set(tensor_inventory):
        raise ValueError("MLP policy overlay tensor inventory is invalid")
    expected_names = {
        f"model.layers.{block}.{path}.weight"
        for block in manifest.get("blocks", ())
        for path in MLP_PATHS
    }
    if (
        not tensors
        or set(tensors) != expected_names
        or any(
            value.ndim != 2
            or not torch.isfinite(value).all()
            or list(value.shape)
            != tensor_inventory[name].get("shape")
            or str(value.dtype).removeprefix("torch.")
            != tensor_inventory[name].get("dtype")
            for name, value in tensors.items()
        )
    ):
        raise ValueError("MLP policy overlay tensors are invalid")
    return tensors, manifest


def _load_overlays(
    overlays: tuple[Path, ...],
) -> tuple[dict[str, torch.Tensor], tuple[dict[str, Any], ...]]:
    tensors: dict[str, torch.Tensor] = {}
    manifests = []
    for overlay in overlays:
        values, manifest = _load_overlay(overlay)
        overlap = set(tensors) & set(values)
        if overlap:
            raise ValueError(
                f"MLP policy overlays contain duplicate layers: {sorted(overlap)}"
            )
        tensors.update(values)
        manifests.append({"directory": str(overlay), **manifest})
    return tensors, tuple(manifests)


def _module_parent(block: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    current = block
    for part in parts[:-1]:
        child = (
            current[part]
            if isinstance(current, nn.ModuleDict)
            else getattr(current, part, None)
        )
        if not isinstance(child, nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current, parts[-1]


def _install_dense_linear(
    block: nn.Module,
    path: str,
    weight: torch.Tensor,
    *,
    device: str,
) -> None:
    parent, name = _module_parent(block, path)
    original = (
        parent[name]
        if isinstance(parent, nn.ModuleDict)
        else getattr(parent, name, None)
    )
    if not isinstance(original, FrozenReferenceLinear):
        raise TypeError(f"frozen MLP module is not replaceable: {path}")
    bias = original.bias
    replacement = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=bias is not None,
        device=device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(weight.to(device=device))
        if bias is not None:
            assert replacement.bias is not None
            replacement.bias.copy_(
                bias.to(
                    device=device,
                    dtype=replacement.bias.dtype,
                )
            )
    replacement.eval()
    if isinstance(parent, nn.ModuleDict):
        parent[name] = replacement
    else:
        setattr(parent, name, replacement)


def run(args: argparse.Namespace) -> int:
    if args.samples <= 0 or args.sequence_length <= 1:
        raise ValueError("WikiText transfer protocol is invalid")
    overlay_paths = (args.overlay, *args.additional_overlay)
    tensors, manifests = _load_overlays(overlay_paths)
    tokens, dataset_fingerprint, bos_token_id = _protocol_tokens(
        args.snapshot,
        args.samples,
        args.sequence_length,
    )
    with acquire_device_lease(args.device):
        loaded = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name=MODEL_SOURCE,
            revision=args.model_revision,
            device=args.device,
            verify_hashes=False,
            backend="factorized",
            use_global_tuning=args.use_global_tuning,
        )
        loaded.model.eval()
        baseline = _evaluate_wikitext(
            loaded.model,
            tokens,
            args.device,
            1,
        )
        base = getattr(loaded.model, "model", None)
        decoder = getattr(base, "layers", None)
        if not isinstance(decoder, nn.ModuleList):
            raise TypeError("frozen model does not expose decoder blocks")
        for tensor_name, weight in tensors.items():
            prefix = "model.layers."
            suffix = ".weight"
            logical = tensor_name.removeprefix(prefix).removesuffix(suffix)
            block_text, path = logical.split(".", maxsplit=1)
            _install_dense_linear(
                decoder[int(block_text)],
                path,
                weight,
                device=args.device,
            )
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        candidate = _evaluate_wikitext(
            loaded.model,
            tokens,
            args.device,
            1,
        )
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        del loaded
    baseline_nll = float(baseline["mean_negative_log_likelihood"])
    candidate_nll = float(candidate["mean_negative_log_likelihood"])
    baseline_ppl = float(baseline["perplexity"])
    candidate_ppl = float(candidate["perplexity"])
    atomic_write_json(
        args.output,
        {
            "schema_version": 2,
            "status": "completed",
            "role": "analysis-only MLP policy frozen-run transfer",
            "model_revision": args.model_revision,
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "use_global_tuning": args.use_global_tuning,
            "overlays": manifests,
            "protocol": {
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "dataset_fingerprint": dataset_fingerprint,
                "bos_token_id": bos_token_id,
                "token_hash": _token_hash(tokens),
            },
            "baseline": baseline,
            "candidate": candidate,
            "mean_negative_log_likelihood_delta": (
                candidate_nll - baseline_nll
            ),
            "perplexity_delta": candidate_ppl - baseline_ppl,
            "perplexity_relative_delta": (
                candidate_ppl / baseline_ppl - 1
            ),
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
