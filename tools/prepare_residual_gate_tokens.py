"""Prepare and permanently retire a fresh C4 residual-stage selection slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
from open_experiment048_c4_slice import _exclusive_registry_lock
from probe_non_wikitext_kd_quality import _load_c4_tokens, _token_hash
from probe_residual_stage_functional import file_hash
from safetensors.torch import save_file
from validate_evaluation_slice_registry import validate_registry

from nanoquant.infrastructure.io_utils import atomic_write_json


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    tokens, fingerprint, bos = _load_c4_tokens(
        args.snapshot, revision="f998d2cd8b92435980789e3ecb2f89b4c68bfe1e", data_file=str(args.c4_file),
        documents=1100, offset=args.offset, samples=args.samples, sequence_length=512, local_files_only=True)
    path = args.output / "tokens.safetensors"
    save_file({"input_ids": tokens}, str(path))
    entry = dict(id=args.slice_id, dataset="allenai/c4", split="validation", offset=args.offset,
                 samples=args.samples, sequence_length=512, token_start=args.offset * 512,
                 token_end=(args.offset + args.samples) * 512, token_hash=_token_hash(tokens),
                 status="retired", consumer="residual-stage-" + args.role,
                 evidence=str(args.output), retirement=dict(action="permanently-retired-before-evaluation",
                                                           role=args.role))
    with _exclusive_registry_lock(args.registry):
        before = json.loads(args.registry.read_text())
        validate_registry(before)
        after = json.loads(json.dumps(before))
        after["slices"].append(entry)
        validate_registry(after)
        atomic_write_json(args.output / "registry-before.json", before)
        atomic_write_json(args.output / "registry-after.json", after)
        atomic_write_json(args.registry, after)
    receipt = dict(entry=entry, file_sha256=file_hash(path), dataset_fingerprint=fingerprint, bos_token_id=bos,
                   c4_file=str(args.c4_file), c4_file_sha256=file_hash(args.c4_file),
                   registry=str(args.registry.resolve()), registry_after_sha256=file_hash(args.registry),
                   retired_before_evaluation=True)
    atomic_write_json(path.with_suffix(".json"), receipt)
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--c4-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("Docs/evaluation-slice-registry.json"))
    parser.add_argument("--slice-id", default="residual-stage-c4-validation472-48x512")
    parser.add_argument("--offset", type=int, default=472)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--role", choices=("fit", "selection", "confirmation"), default="selection")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
