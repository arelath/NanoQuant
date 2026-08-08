"""Evaluate a compact product-codebook overlay on paired held-out teacher KL."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import _paired_payload, _select_token_window
from probe_factorized_component_overlays_kl import _evaluate_arm
from probe_mlp_overlays_kl import _split_tokens
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION

from nanoquant.application.kl_budget import KlSequenceResult
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.packed_model_loader import load_packed_model
from nanoquant.kl_budget_workflow import _token_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--product-overlay", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wikitext-split", choices=("test", "validation"), default="validation")
    parser.add_argument("--wikitext-offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    return (
        {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(value, torch.float32)
        if isinstance(value, str)
        else torch.float32
    )


def run(args: argparse.Namespace) -> int:
    if (
        args.samples <= 0
        or args.sequence_length <= 1
        or args.wikitext_offset < 0
        or args.token_chunk_size <= 0
    ):
        raise ValueError("packed product-codebook KL protocol is invalid")
    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    all_tokens, fingerprint, bos_token_id = _split_tokens(
        args.snapshot,
        split=args.wikitext_split,
        samples=args.wikitext_offset + args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    tokens = _select_token_window(all_tokens, offset=args.wikitext_offset, samples=args.samples)
    arms = ("packed_base", "product_codebook_packed")
    protocol = {
        "wikitext_split": args.wikitext_split,
        "wikitext_offset": args.wikitext_offset,
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "token_chunk_size": args.token_chunk_size,
        "dataset_fingerprint": fingerprint,
        "bos_token_id": bos_token_id,
        "token_hash": _token_hash(tokens),
        "packed_descriptor_sha256": hash_file(
            args.packed_artifact / "nanoquant-packed-model.json"
        ),
        "product_descriptor_sha256": hash_file(
            args.product_overlay / "nanoquant-product-codebook-overlay.json"
        ),
        "arms": list(arms),
    }
    if args.output.is_file():
        completed = json.loads(args.output.read_text(encoding="utf-8"))
        if completed.get("status") != "completed" or completed.get("protocol") != protocol:
            raise ValueError("existing packed product-codebook KL output differs")
        return 0
    checkpoint_path = args.output.with_name(args.output.stem + ".checkpoint.json")
    checkpoint: dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "protocol": protocol,
        "sequences": {},
    }
    if checkpoint_path.is_file():
        loaded_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            loaded_checkpoint.get("schema_version") != 1
            or loaded_checkpoint.get("protocol") != protocol
            or not isinstance(loaded_checkpoint.get("sequences"), dict)
        ):
            raise ValueError("packed product-codebook KL checkpoint differs")
        checkpoint = loaded_checkpoint

    def restored(name: str) -> tuple[KlSequenceResult, ...]:
        values = cast(dict[str, object], checkpoint["sequences"]).get(name, [])
        if not isinstance(values, list):
            raise ValueError("packed product-codebook KL arm checkpoint is invalid")
        return tuple(
            KlSequenceResult(
                float(value["negative_log_likelihood"]),
                float(value["kl_nats_per_token"]),
                int(value["token_count"]),
            )
            for value in values
        )

    def save_progress(name: str, sequences: tuple[KlSequenceResult, ...]) -> None:
        cast(dict[str, object], checkpoint["sequences"])[name] = [
            to_dict(value) for value in sequences
        ]
        atomic_write_json(checkpoint_path, checkpoint)

    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        cast(Any, teacher).config.use_cache = False
        results = {}
        identity = None
        global_tuning = None
        observed_product_hash = None
        for name, overlay in (
            ("packed_base", None),
            ("product_codebook_packed", args.product_overlay),
        ):
            loaded = load_packed_model(
                args.packed_artifact,
                args.run_output,
                args.snapshot,
                source_name=MODEL_SOURCE,
                revision=args.model_revision,
                expected_blocks=26,
                device=args.device,
                backend="factorized",
                use_global_tuning=True,
                product_codebook_overlay=overlay,
            )
            current_identity = to_dict(loaded.identity)
            current_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
            if identity is not None and current_identity != identity:
                raise ValueError("packed KL arms have different frozen identities")
            if global_tuning is not None and current_tuning != global_tuning:
                raise ValueError("packed KL arms have different global tuning identities")
            identity = current_identity
            global_tuning = current_tuning
            observed_product_hash = (
                observed_product_hash or loaded.product_codebook_descriptor_sha256
            )

            def progress(
                sequences: tuple[KlSequenceResult, ...],
                arm: str = name,
            ) -> None:
                save_progress(arm, sequences)

            results[name] = _evaluate_arm(
                name,
                teacher,
                loaded.model,
                tokens,
                device=args.device,
                token_chunk_size=args.token_chunk_size,
                completed=restored(name),
                progress=progress,
            )
            del loaded
            gc.collect()
            torch.cuda.empty_cache()
        del teacher
    before = results["packed_base"]
    after = results["product_codebook_packed"]
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "packed product-codebook unchanged-bit component replay KL gate",
            "model_revision": args.model_revision,
            "run_output": str(args.run_output.resolve()),
            "packed_artifact": str(args.packed_artifact.resolve()),
            "product_overlay": str(args.product_overlay.resolve()),
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "product_descriptor_sha256": observed_product_hash,
            "protocol": protocol,
            "results": {name: to_dict(result) for name, result in results.items()},
            "product_minus_base": {
                "nll": _paired_metric_payload(before, after, "negative_log_likelihood"),
                "kl": _paired_payload(before, after, seed=0),
            },
        },
    )
    checkpoint["status"] = "completed"
    atomic_write_json(checkpoint_path, checkpoint)
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
