"""Evaluate a packed NanoQuant artifact on the exact retained WikiText protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import _paths  # noqa: F401
from evaluate_wikitext import _protocol_tokens
from probe_mlp_policy_frozen_transfer import _evaluate_per_sequence

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.packed_model_loader import load_packed_model
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--wikitext-offset", type=int, default=0)
    parser.add_argument("--expected-blocks", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("dense", "factorized"), default="factorized")
    parser.add_argument("--no-global-tuning", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.samples <= 0 or args.sequence_length <= 1 or args.wikitext_offset < 0:
        raise ValueError("packed WikiText protocol is invalid")
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
        loaded = load_packed_model(
            args.packed_artifact,
            args.run_output,
            args.snapshot,
            source_name=args.source,
            revision=args.revision,
            expected_blocks=args.expected_blocks,
            device=args.device,
            backend=args.backend,
            use_global_tuning=not args.no_global_tuning,
        )
        result = _evaluate_per_sequence(loaded.model, tokens, args.device)
        identity = to_dict(loaded.identity)
        global_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
        packed_descriptor_sha256 = loaded.packed_descriptor_sha256
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "packed NanoQuant exact WikiText probe",
            "packed_artifact": str(args.packed_artifact),
            "packed_descriptor_sha256": packed_descriptor_sha256,
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "backend": args.backend,
            "protocol": {
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "wikitext_offset": args.wikitext_offset,
                "dataset_fingerprint": dataset_fingerprint,
                "bos_token_id": bos_token_id,
                "token_hash": _token_hash(tokens),
            },
            "wikitext": result,
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
