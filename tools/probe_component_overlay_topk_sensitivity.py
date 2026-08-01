"""Measure whether retained top-k KD targets can see a component-overlay correction."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_global_foldable_mlp_multipliers import (
    _evaluate_topk_kl,
    _load_calibration,
    _load_training_cache,
)
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.io_utils import atomic_write_json


def _parse_arm(value: str) -> tuple[str, Path | None]:
    name, separator, path = value.partition("=")
    if not name.strip() or (separator and not path.strip()):
        raise argparse.ArgumentTypeError("arm must use name or name=component-overlay")
    return name.strip(), None if not separator else Path(path.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", type=_parse_arm, action="append", required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-chunk-size", type=int, default=128)
    return parser


def run(args: argparse.Namespace) -> int:
    if (
        len(args.arm) < 2
        or len({name for name, _path in args.arm}) != len(args.arm)
        or args.token_chunk_size <= 0
    ):
        raise ValueError("component-overlay top-k sensitivity protocol is invalid")
    tokens = _load_calibration(args.run_output)
    cache = _load_training_cache(args.run_output, epochs=1)
    results = {}
    manifests = {}
    identity = None
    global_tuning = None
    with acquire_device_lease(args.device):
        for name, overlay in args.arm:
            loaded = load_frozen_run(
                args.run_output,
                args.snapshot,
                source_name=MODEL_SOURCE,
                revision=args.model_revision,
                device=args.device,
                verify_hashes=False,
                backend="factorized",
                use_global_tuning=True,
                component_overlay=overlay,
            )
            observed_identity = {
                "model_hash": loaded.identity.model_hash,
                "config_hash": loaded.identity.config_hash,
                "plan_hash": loaded.identity.plan_hash,
            }
            observed_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
            if identity is not None and identity != observed_identity:
                raise ValueError("top-k sensitivity arms have different frozen identities")
            if global_tuning is not None and global_tuning != observed_tuning:
                raise ValueError("top-k sensitivity arms have different global tuning identities")
            identity = observed_identity
            global_tuning = observed_tuning
            manifests[name] = (
                None
                if overlay is None
                else {
                    "directory": str(overlay),
                    **json.loads((overlay / "manifest.json").read_text(encoding="utf-8")),
                }
            )
            results[name] = _evaluate_topk_kl(
                loaded.model,
                tokens,
                cache,
                device=args.device,
                token_chunk_size=args.token_chunk_size,
            )
            del loaded
            gc.collect()
            torch.cuda.empty_cache()
    baseline_name = args.arm[0][0]
    baseline = results[baseline_name]
    comparisons = {
        f"{name}_minus_{baseline_name}": {
            key: float(result[key]) - float(baseline[key])
            for key in ("cross_entropy", "teacher_entropy", "topk_kl")
        }
        for name, result in results.items()
        if name != baseline_name
    }
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only retained top-k objective sensitivity to component overlays",
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "teacher_cache": {
                "epoch_count": len(cache.epochs),
                "batch_count": len(cache.epochs[0]),
                "bytes": cache.bytes,
            },
            "component_overlays": manifests,
            "results": results,
            "comparisons": comparisons,
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
