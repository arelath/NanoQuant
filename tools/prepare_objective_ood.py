"""Pin a PTB transfer slice, preserving source bytes and permanent retirement."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import _paths  # noqa: F401
from open_experiment048_c4_slice import _exclusive_registry_lock
from probe_non_wikitext_kd_quality import _token_hash
from probe_residual_stage_functional import file_hash
from safetensors.torch import save_file
from transformers import AutoTokenizer
from validate_evaluation_slice_registry import validate_registry

from nanoquant.infrastructure.io_utils import atomic_write_json

PTB_COMMIT = "76870253cfca069477f06b7056af87f98490b6eb"
PTB_URL = f"https://raw.githubusercontent.com/wojzaremba/lstm/{PTB_COMMIT}/data/ptb.test.txt"


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "ptb.test.txt"
    with urllib.request.urlopen(PTB_URL, timeout=60) as response:
        raw.write_bytes(response.read())
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    # Preserve the processed corpus's text and newlines. These are Gemma BPE
    # scores, not the original benchmark's word-vocabulary perplexities.
    stream = tokenizer(raw.read_text(encoding="utf-8"), add_special_tokens=True, return_tensors="pt").input_ids
    count, length = 96, 512
    if stream.shape[1] < count * length or tokenizer.bos_token_id != 2 or stream[0, 0] != 2:
        raise ValueError("PTB token coverage or pinned tokenizer BOS mismatch")
    tokens = stream[0, :count * length].reshape(count, length).contiguous()
    path = args.output / "tokens.safetensors"
    save_file({"input_ids": tokens}, str(path))
    entry = dict(id="objective-continuation-ptb-test000-96x512", dataset="ptb:wojzaremba-word-level",
                 split="test", offset=0, samples=count, sequence_length=512, token_start=0, token_end=count * length,
                 token_hash=_token_hash(tokens), status="retired", consumer="objective-continuation-transfer",
                 evidence=str(args.output), retirement=dict(action="permanently-retired-before-evaluation",
                                                           role="transfer"))
    with _exclusive_registry_lock(args.registry):
        before = json.loads(args.registry.read_text())
        validate_registry(before)
        after = json.loads(json.dumps(before))
        after["slices"].append(entry)
        validate_registry(after)
        atomic_write_json(args.output / "registry-before.json", before)
        atomic_write_json(args.output / "registry-after.json", after)
        atomic_write_json(args.registry, after)
    receipt = dict(entry=entry, file_sha256=file_hash(path), source_url=PTB_URL, source_commit=PTB_COMMIT,
                   source_sha256=file_hash(raw), source_bytes=raw.stat().st_size,
                   source_tokens=stream.shape[1], context_policy="contiguous 512-token windows; first token is context",
                   registry=str(args.registry.resolve()), retired_before_evaluation=True)
    atomic_write_json(path.with_suffix(".json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("Docs/evaluation-slice-registry.json"))
    run(parser.parse_args())
