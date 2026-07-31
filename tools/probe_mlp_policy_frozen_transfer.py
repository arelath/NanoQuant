"""Evaluate an exported dense MLP policy inside a retained frozen run."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from evaluate_wikitext import _evaluate as _evaluate_wikitext
from evaluate_wikitext import _protocol_tokens
from torch import nn

from nanoquant.application.layers import FrozenReferenceLinear
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.factorized_component_overlay import (
    apply_factorized_component_overlay,
    load_factorized_component_overlay,
)
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.io_utils import (
    atomic_write_json,
    hash_file,
)
from nanoquant.infrastructure.safetensors_io import SAFETENSORS
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

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
    parser.add_argument("--component-overlay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--wikitext-offset", type=int, default=0)
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
        f"model.layers.{block}.{path}.weight" for block in manifest.get("blocks", ()) for path in MLP_PATHS
    }
    if (
        not tensors
        or set(tensors) != expected_names
        or any(
            value.ndim != 2
            or not torch.isfinite(value).all()
            or list(value.shape) != tensor_inventory[name].get("shape")
            or str(value.dtype).removeprefix("torch.") != tensor_inventory[name].get("dtype")
            for name, value in tensors.items()
        )
    ):
        raise ValueError("MLP policy overlay tensors are invalid")
    return tensors, manifest


def _evaluate_per_sequence(
    model: nn.Module,
    tokens: torch.Tensor,
    device: str,
) -> dict[str, object]:
    started = time.perf_counter()
    sequences = [_evaluate_wikitext(model, tokens[index : index + 1], device, 1) for index in range(tokens.shape[0])]
    token_count = sum(int(item["token_count"]) for item in sequences)
    total_nll = math.fsum(float(item["total_negative_log_likelihood"]) for item in sequences)
    mean_nll = total_nll / token_count
    return {
        "total_negative_log_likelihood": total_nll,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(mean_nll),
        "token_count": token_count,
        "window_count": sum(int(item["window_count"]) for item in sequences),
        "sample_count": len(sequences),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_device_bytes": max(int(item["peak_device_bytes"]) for item in sequences),
        "peak_host_bytes": max(int(item["peak_host_bytes"]) for item in sequences),
        "sequences": [
            {
                "mean_negative_log_likelihood": item["mean_negative_log_likelihood"],
                "perplexity": item["perplexity"],
                "token_count": item["token_count"],
            }
            for item in sequences
        ],
    }


def _paired_nll_payload(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    seed: int = 0,
    resamples: int = 10_000,
) -> dict[str, object]:
    before = baseline.get("sequences")
    after = candidate.get("sequences")
    if not isinstance(before, list) or not isinstance(after, list) or len(before) != len(after) or not before:
        raise ValueError("paired NLL requires aligned sequence results")
    deltas = []
    generator = random.Random(seed)
    for _sample in range(resamples):
        indices = [generator.randrange(len(before)) for _ in before]
        tokens = math.fsum(float(before[index]["token_count"]) for index in indices)
        before_nll = (
            math.fsum(
                float(before[index]["mean_negative_log_likelihood"]) * float(before[index]["token_count"])
                for index in indices
            )
            / tokens
        )
        after_nll = (
            math.fsum(
                float(after[index]["mean_negative_log_likelihood"]) * float(after[index]["token_count"])
                for index in indices
            )
            / tokens
        )
        deltas.append(after_nll - before_nll)
    deltas.sort()
    point = float(candidate["mean_negative_log_likelihood"]) - float(baseline["mean_negative_log_likelihood"])
    lower = deltas[int(0.025 * resamples)]
    upper = deltas[int(0.975 * resamples) - 1]
    return {
        "point_delta": point,
        "relative_delta": point / float(baseline["mean_negative_log_likelihood"]),
        "lower_delta": lower,
        "upper_delta": upper,
        "confidence": 0.95,
        "resamples": resamples,
        "improved_with_confidence": point < 0 and upper < 0,
    }


def _load_overlays(
    overlays: tuple[Path, ...],
) -> tuple[dict[str, torch.Tensor], tuple[dict[str, Any], ...]]:
    tensors: dict[str, torch.Tensor] = {}
    manifests = []
    for overlay in overlays:
        values, manifest = _load_overlay(overlay)
        overlap = set(tensors) & set(values)
        if overlap:
            raise ValueError(f"MLP policy overlays contain duplicate layers: {sorted(overlap)}")
        tensors.update(values)
        manifests.append({"directory": str(overlay), **manifest})
    return tensors, tuple(manifests)


def _module_parent(block: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    current = block
    for part in parts[:-1]:
        child = current[part] if isinstance(current, nn.ModuleDict) else getattr(current, part, None)
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
    original = parent[name] if isinstance(parent, nn.ModuleDict) else getattr(parent, name, None)
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
    if args.samples <= 0 or args.sequence_length <= 1 or args.wikitext_offset < 0:
        raise ValueError("WikiText transfer protocol is invalid")
    overlay_paths = (args.overlay, *args.additional_overlay)
    tensors, manifests = _load_overlays(overlay_paths)
    if args.wikitext_offset == 0:
        tokens, dataset_fingerprint, bos_token_id = _protocol_tokens(
            args.snapshot,
            args.samples,
            args.sequence_length,
        )
    else:
        all_tokens, dataset_fingerprint, bos_token_id = _wikitext_tokens(
            args.snapshot,
            samples=args.wikitext_offset + args.samples,
            sequence_length=args.sequence_length,
            local_files_only=False,
        )
        tokens = all_tokens[args.wikitext_offset : args.wikitext_offset + args.samples]
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
        baseline = _evaluate_per_sequence(
            loaded.model,
            tokens,
            args.device,
        )
        base = getattr(loaded.model, "model", None)
        decoder = getattr(base, "layers", None)
        if not isinstance(decoder, nn.ModuleList):
            raise TypeError("frozen model does not expose decoder blocks")
        component_candidate = None
        component_overlay = None
        component_application = None
        if args.component_overlay is not None:
            loaded_component_overlay = load_factorized_component_overlay(
                args.component_overlay,
                frozen_identity={
                    "model_hash": loaded.identity.model_hash,
                    "config_hash": loaded.identity.config_hash,
                    "plan_hash": loaded.identity.plan_hash,
                },
                global_tuning=loaded.global_tuning,
            )
            applied = apply_factorized_component_overlay(
                loaded.model,
                loaded_component_overlay,
            )
            component_candidate = _evaluate_per_sequence(
                loaded.model,
                tokens,
                args.device,
            )
            component_overlay = {
                "directory": str(args.component_overlay),
                **loaded_component_overlay.manifest,
            }
            component_application = {
                "tensor_count": applied.tensor_count,
                "layer_count": applied.layer_count,
                "replaced_bytes": applied.replaced_bytes,
                "replacement_bytes": applied.replacement_bytes,
            }
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
        candidate = _evaluate_per_sequence(
            loaded.model,
            tokens,
            args.device,
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
            "schema_version": 4,
            "status": "completed",
            "role": "analysis-only MLP policy frozen-run transfer",
            "model_revision": args.model_revision,
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "use_global_tuning": args.use_global_tuning,
            "overlays": manifests,
            "component_overlay": component_overlay,
            "component_application": component_application,
            "protocol": {
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "wikitext_offset": args.wikitext_offset,
                "dataset_fingerprint": dataset_fingerprint,
                "bos_token_id": bos_token_id,
                "token_hash": _token_hash(tokens),
            },
            "baseline": baseline,
            "candidate": candidate,
            "mean_negative_log_likelihood_delta": (candidate_nll - baseline_nll),
            "perplexity_delta": candidate_ppl - baseline_ppl,
            "perplexity_relative_delta": (candidate_ppl / baseline_ppl - 1),
            "paired_candidate_minus_baseline_nll": _paired_nll_payload(
                baseline,
                candidate,
            ),
            "component_candidate": component_candidate,
            "paired_component_minus_baseline_nll": (
                None
                if component_candidate is None
                else _paired_nll_payload(baseline, component_candidate)
            ),
            "paired_component_minus_dense_nll": (
                None
                if component_candidate is None
                else _paired_nll_payload(candidate, component_candidate)
            ),
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
