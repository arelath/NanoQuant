"""Compare dense MLP overlays on a pinned held-out C4 token window."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_composed_context_coordinate_sweep import _overlay_replacements
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import _dtype, _paired_payload, _replace_weights
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION
from probe_non_wikitext_kd_quality import C4_REVISION, _load_c4_tokens

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.kl_splice import DenseKlSpliceEvaluator, collect_splice_reconstructions
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.safetensors_io import SAFETENSORS
from nanoquant.kl_budget_workflow import _token_hash


def _parse_overlay(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("overlay must use name=path")
    return name, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--c4-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=_parse_overlay, action="append", required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--c4-documents", type=int, default=1100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _load_partial_overlay(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    manifest_path = path / "manifest.json"
    tensor_path = path / "weights.safetensors"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise ValueError("partial MLP overlay is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest.get("tensors")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("tensor_sha256") != hash_file(tensor_path)
        or not isinstance(inventory, dict)
        or manifest.get("layer_count") != len(inventory)
    ):
        raise ValueError("partial MLP overlay identity is invalid")
    tensors = SAFETENSORS.load(tensor_path)
    if set(tensors) != set(inventory) or not tensors:
        raise ValueError("partial MLP overlay inventory is invalid")
    for name, value in tensors.items():
        expected = inventory[name]
        if (
            not name.startswith("model.layers.")
            or not name.endswith(".weight")
            or value.ndim != 2
            or not torch.isfinite(value).all()
            or list(value.shape) != expected.get("shape")
            or str(value.dtype).removeprefix("torch.") != expected.get("dtype")
        ):
            raise ValueError("partial MLP overlay tensor is invalid")
    return tensors, manifest


def run(args: argparse.Namespace) -> int:
    if min(args.samples, args.sequence_length - 1) <= 0 or args.offset < 0:
        raise ValueError("C4 overlay protocol is invalid")
    model_config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(model_config)
    tokens, fingerprint, bos_id = _load_c4_tokens(
        args.snapshot,
        revision=C4_REVISION,
        data_file=str(args.c4_file),
        documents=args.c4_documents,
        offset=args.offset,
        samples=args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    loaded_overlays = {}
    manifests = {}
    for name, path in args.overlay:
        tensors, manifest = _load_partial_overlay(path)
        loaded_overlays[name] = _overlay_replacements(tensors)
        manifests[name] = {"directory": str(path.resolve()), **manifest}
    with acquire_device_lease(args.device):
        loaded = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name=MODEL_SOURCE,
            revision=PINNED_MODEL_REVISION,
            device=args.device,
            verify_hashes=False,
            backend="factorized",
            use_global_tuning=True,
        )
        baseline = collect_splice_reconstructions(loaded)
        candidates = {
            name: _replace_weights(baseline, replacements)
            for name, replacements in loaded_overlays.items()
        }
        identity = to_dict(loaded.identity)
        global_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(model_config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        results = {}
        teacher_nll = 0.0
        teacher_cache: tuple[torch.Tensor, ...] = ()
        for name, reconstructions in {"baseline": baseline, **candidates}.items():
            evaluator = DenseKlSpliceEvaluator(
                teacher,
                reconstructions,
                tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            if not teacher_cache:
                teacher_nll, teacher_cache = evaluator.teacher_cache_state()
            else:
                evaluator.install_teacher_cache(teacher_nll, teacher_cache)
            results[name] = evaluator("full")
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()
        del teacher
    baseline_result = results["baseline"]
    output = {
        "schema_version": 1,
        "status": "complete",
        "role": "analysis-only dense MLP overlay C4 comparison",
        "run_output": str(args.run_output.resolve()),
        "identity": identity,
        "global_tuning": global_tuning,
        "overlays": manifests,
        "protocol": {
            "dataset": "allenai/c4",
            "revision": C4_REVISION,
            "offset": args.offset,
            "samples": args.samples,
            "sequence_length": args.sequence_length,
            "token_hash": _token_hash(tokens),
            "dataset_fingerprint": fingerprint,
            "bos_token_id": bos_id,
        },
        "results": {name: to_dict(result) for name, result in results.items()},
        "paired_candidate_minus_baseline": {
            name: {
                "kl": _paired_payload(baseline_result, result, seed=0),
                "nll": _paired_metric_payload(
                    baseline_result, result, "negative_log_likelihood"
                ),
            }
            for name, result in results.items()
            if name != "baseline"
        },
    }
    names = [name for name, _path in args.overlay]
    first = names[0]
    output["paired_overlay_minus_first"] = {
        name: {
            "kl": _paired_payload(results[first], results[name], seed=0),
            "nll": _paired_metric_payload(
                results[first], results[name], "negative_log_likelihood"
            ),
        }
        for name in names[1:]
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
