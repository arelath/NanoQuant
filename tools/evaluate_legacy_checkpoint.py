"""Evaluate a retained packed legacy checkpoint through rewrite reference modules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch
from evaluate_wikitext import _checkpoint_dtype, _evaluate, _protocol_tokens
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.legacy_checkpoint import apply_legacy_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-arrow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation window batch size; values above 1 are faster but approximate for parity.",
    )
    parser.add_argument("--backend", choices=("factorized", "dense"), default="factorized")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--wait-for-device-seconds",
        type=float,
        default=0.0,
        help="wait this long for the shared device lease instead of failing immediately",
    )
    args = parser.parse_args()

    tokens, fingerprint, bos_id = _protocol_tokens(
        args.snapshot,
        args.samples,
        args.sequence_length,
        args.dataset_arrow,
    )
    with wait_for_device_lease(args.device, args.wait_for_device_seconds):
        model = cast(
            nn.Module,
            AutoModelForCausalLM.from_pretrained(
                args.snapshot,
                local_files_only=True,
                torch_dtype=_checkpoint_dtype(args.snapshot),
            ),
        )
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError("legacy checkpoint must contain a tensor state dictionary")
        layers = apply_legacy_checkpoint(model, cast(dict[str, Any], state), backend=args.backend)
        model.to(args.device).eval()
        cast(Any, model).config.use_cache = False
        result = _evaluate(model, tokens, args.device, args.batch_size)
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "snapshot": str(args.snapshot.resolve()),
        "dataset_arrow": str(args.dataset_arrow.resolve()),
        "dataset_fingerprint": fingerprint,
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "backend": args.backend,
        "bos_token_id": bos_id,
        "layer_count": len(layers),
        "token_hash": "sha256:"
        + hashlib.sha256(tokens.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
