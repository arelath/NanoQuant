"""Build an offline, auditable Gemma 3 text calibration candidate (no model inference)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import _paths  # noqa: F401

from nanoquant.infrastructure.io_utils import hash_file

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "mixed": ("eaddario/imatrix-calibration", "e87ed55dcba9d9c3a3e41539f3e728e981b1daa4",
              "combined_all_huge.parquet", "content"),
    "instructions": ("mlabonne/harmless_alpaca", "02c6a92cfcf11bb0c387334f8146d149d65b587f",
                     "data/train-00000-of-00001.parquet", "text"),
    "prose": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
              "wikitext-2-raw-v1/train-00000-of-00001.parquet", "text"),
}
MODEL = "google/gemma-3-1b-it"
REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
WEIGHTS = {"prose": 0.30, "instructions": 0.20, "code": 0.25, "math": 0.15, "multiscript": 0.10}
CODE = re.compile(r"\b(Python|JavaScript|SQL|function|algorithm|code|C\+\+|TypeScript|Rust|Java)\b")
MATH = re.compile(r"\b(equation|integer|triangle|polynomial|probability|calculate|compute)\b", re.I)
EVAL_CACHES = ("Salesforce___wikitext", "allenai___c4", "cais___mmlu", "openai___gsm8k",
               "allenai___ai2_arc", "Rowan___hellaswag", "allenai___winogrande", "ybisk___piqa",
               "baber___piqa", "aps___super_glue", "ChilleD___svamp", "MU-NLPC___calc-asdiv_a",
               "MU-NLPC___calc-svamp", "mlabonne___harmless_alpaca")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text: str, width: int = 13) -> set[bytes]:
    """Case/punctuation-insensitive word overlaps; not a semantic decontamination guarantee."""
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    return {hashlib.blake2b(" ".join(words[i:i + width]).encode(), digest_size=16).digest()
            for i in range(len(words) - width + 1)}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def evaluation_index(cache: Path):
    import pyarrow as pa

    overlap = set()
    exact = set()
    receipts = []
    for dataset in EVAL_CACHES:
        for path in sorted((cache / dataset).rglob("*.arrow")):
            if not re.search(r"(?:test|validation|valid|dev)(?:[-.]|$)", path.name):
                continue
            with pa.memory_map(str(path), "r") as source:
                for batch in pa.ipc.open_stream(source):
                    for row in batch.to_pylist():
                        for text in strings(row):
                            if len(text.strip()) >= 30:
                                exact.add(digest(normalized(text)))
                                overlap.update(shingles(text))
            receipts.append({"path": str(path), "sha256": hash_file(path)})
    if not receipts:
        raise ValueError("no cached evaluation splits found; provide the correct --datasets-cache")
    return exact, overlap, receipts


def classify(text: str) -> str | None:
    if "```" in text or CODE.search(text):
        return "code"
    if "$" in text or "\\[" in text or MATH.search(text):
        return "math"
    letters = [c for c in text if c.isalpha()]
    if sum(ord(c) > 127 for c in letters) > max(4, 0.15 * len(letters)):
        return "multiscript"
    return None


def source_records(key: str, path: Path):
    import pyarrow.parquet as pq

    column = SOURCES[key][3]
    article = "preamble"
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=[column]):
        for text in batch.column(0).to_pylist():
            if not isinstance(text, str):
                continue
            parts = text.splitlines() if key == "mixed" else [text]
            for part in parts:
                part = part.strip()
                if key == "prose" and re.fullmatch(r"= [^=]+ =", part):
                    article = normalized(part)
                category = classify(part) if key == "mixed" else key
                if category:
                    yield {"source": key, "category": category, "text": part,
                           "group": digest(article if key == "prose" else normalized(part))}


def render(record: dict, tokenizer) -> str:
    text = record["text"]
    if record["category"] in {"instructions", "code", "math"}:
        return tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                             tokenize=False, add_generation_prompt=False)
    return tokenizer.bos_token + text + tokenizer.eos_token


def choose(records: list[dict], budget: int, seed: int, excluded_shingles: set[bytes]):
    """Whole records, no replacement; fail if a quota cannot be filled."""
    pools = defaultdict(list)
    for record in records:
        pools[record["category"]].append(record)
    chosen = []
    for index, (category, weight) in enumerate(WEIGHTS.items()):
        pool = sorted(pools[category], key=lambda r: r["id"])
        random.Random(seed + index).shuffle(pool)
        total = 0
        target = math.ceil(budget * weight)
        for record in pool:
            if excluded_shingles and shingles(record["text"]) & excluded_shingles:
                continue
            chosen.append(record)
            total += record["tokens"]
            if total >= target:
                break
        if total < target:
            raise ValueError(f"insufficient {category} tokens: {total}/{target}; lower budget or add sources")
    # Interleave categories throughout the file, including short prefix runs.
    queues = defaultdict(list)
    for record in chosen:
        queues[record["category"]].append(record)
    emitted = Counter()
    ordered = []
    while any(queues.values()):
        category = min((k for k in WEIGHTS if queues[k]), key=lambda k: emitted[k] / WEIGHTS[k])
        record = queues[category].pop()
        ordered.append(record)
        emitted[category] += record["tokens"]
    return ordered


def write_split(root: Path, name: str, records: list[dict], tokenizer):
    rendered = [r["rendered"] for r in records]
    # Each record starts with BOS and ends with EOS or end_of_turn; no synthetic padding.
    text = "".join(rendered)
    (root / f"{name}.txt").write_bytes(text.encode("utf-8"))
    with (root / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    ids = tokenizer.encode(text, add_special_tokens=False)
    per_category = Counter()
    for record in records:
        per_category[record["category"]] += record["tokens"]
    return {"records": len(records), "tokens": len(ids), "record_tokens": sum(per_category.values()),
            "category_tokens": dict(per_category), "category_records": dict(Counter(r["category"] for r in records)),
            "full_contexts": {str(c): len(ids) // c for c in (512, 2048, 4096)},
            "trailing_tokens": {str(c): len(ids) % c for c in (512, 2048, 4096)},
            "unk_tokens": ids.count(tokenizer.unk_token_id), "pad_tokens": ids.count(tokenizer.pad_token_id)}


def build(args):
    import transformers
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    if args.tokens < 1000 or args.heldout_tokens < 1000:
        raise ValueError("token budgets must be at least 1000")
    if args.output.exists():
        raise ValueError("output already exists; choose a new directory (never overwrite a corpus)")
    snapshot = Path(snapshot_download(
        MODEL, revision=REVISION, local_files_only=True,
        allow_patterns=["tokenizer*", "special_tokens_map.json", "*.jinja", "config.json"]))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer.model_max_length = 10**9
    exact_eval, eval_shingles, eval_files = evaluation_index(args.datasets_cache)
    print(f"Evaluation exclusion index: {len(eval_files)} files, {len(eval_shingles)} shingles", flush=True)
    source_info = {}
    candidates = []
    seen = set()
    rejected = Counter()
    for key, (repo, revision, filename, _) in SOURCES.items():
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_files_only=True))
        source_info[key] = {"repository": repo, "revision": revision, "filename": filename,
                            "sha256": hash_file(path), "path": str(path)}
        records = []
        for ordinal, record in enumerate(source_records(key, path)):
            text = record["text"]
            identity = digest(normalized(text))
            if len(text) < 60 or len(text) > 20000 or "\ufffd" in text or "\x00" in text:
                rejected["length_or_invalid_text"] += 1
                continue
            if any(token in text for token in tokenizer.all_special_tokens):
                rejected["embedded_special_token"] += 1
                continue
            if identity in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(identity)
            if identity in exact_eval or shingles(text) & eval_shingles:
                rejected["evaluation_overlap"] += 1
                continue
            record.update({"id": identity, "source_ordinal": ordinal, "rendered": render(record, tokenizer)})
            records.append(record)
        encoded = tokenizer([r["rendered"] for r in records], add_special_tokens=False)["input_ids"]
        for record, ids in zip(records, encoded, strict=True):
            if len(ids) < 16 or len(ids) > 2048 or tokenizer.unk_token_id in ids:
                rejected["token_length_or_unknown"] += 1
                continue
            record["tokens"] = len(ids)
            # Keep all paragraphs of a WikiText article in one split.
            is_heldout = int(digest(f"{args.seed}:{record['group']}")[:8], 16) % 10 == 0
            record["split"] = "heldout" if is_heldout else "calibration"
            candidates.append(record)
        print(f"Read {key}; eligible records so far: {len(candidates)}", flush=True)
    heldout = choose([r for r in candidates if r["split"] == "heldout"], args.heldout_tokens, args.seed + 100, set())
    holdout_shingles = set()
    for record in heldout:
        holdout_shingles.update(shingles(record["text"]))
    calibration = choose([r for r in candidates if r["split"] == "calibration"], args.tokens,
                         args.seed, holdout_shingles)
    if {r["id"] for r in calibration} & {r["id"] for r in heldout}:
        raise AssertionError("split identity overlap")
    args.output.mkdir(parents=True)
    stats = {name: write_split(args.output, name, records, tokenizer)
             for name, records in (("calibration", calibration), ("heldout", heldout))}
    compatibility = {}
    if args.verify_tokenizer:
        other = AutoTokenizer.from_pretrained(str(args.verify_tokenizer), local_files_only=True)
        other.model_max_length = 10**9
        for name, records in (("calibration", calibration), ("heldout", heldout)):
            for record in records:
                if render(record, other) != record["rendered"]:
                    raise ValueError("second tokenizer renders a different template")
            text = (args.output / f"{name}.txt").read_text(encoding="utf-8")
            if tokenizer.encode(text, add_special_tokens=False) != other.encode(text, add_special_tokens=False):
                raise ValueError("second tokenizer produces different tokens")
        compatibility = {"snapshot": str(args.verify_tokenizer), "exact_text_and_token_match": True,
                         "files": {p.name: hash_file(p) for p in sorted(args.verify_tokenizer.glob("tokenizer*"))}}
    manifest = {
        "schema": "nanoquant-gemma-calibration-corpus-v1", "status": "text-corpus-complete",
        "quality_status": "candidate; no imatrix collection or quantization comparison performed",
        "seed": args.seed, "target_tokens": args.tokens, "target_heldout_tokens": args.heldout_tokens,
        "weights": WEIGHTS, "sources": source_info, "rejections": dict(rejected), "splits": stats,
        "tokenizer": {"model": MODEL, "revision": REVISION, "transformers": transformers.__version__,
                      "files": {p.name: hash_file(p) for p in sorted(snapshot.glob("tokenizer*"))}},
        "second_tokenizer_validation": compatibility,
        "evaluation_exclusions": {"files": eval_files, "ngram_words": 13,
                                  "exact_records": len(exact_eval), "shingles": len(eval_shingles)},
        "split_policy": "hash source groups before selection; exclude heldout 13-word overlaps from calibration",
        "format": "Gemma template user turns for prompts; BOS/text/EOS for prose; parse special tokens",
        "limits": ["Prompt-only instruction data: no assistant answers or complete multi-turn conversations.",
                   "Text only: no vision calibration, and no long-context acceptance.",
                   "Heuristic code/math/multiscript labels; multiscript is not verified language identification.",
                   "Source literal backslash escapes are preserved; no ambiguous blanket unescaping.",
                   "Cached evaluation overlap filtering is lexical, partial, and not a semantic leakage guarantee.",
                   "Heldout is disjoint from this candidate, not necessarily from earlier calibration runs."],
        "builder_sha256": hash_file(Path(__file__)),
        "files": {p.name: {"sha256": hash_file(p), "bytes": p.stat().st_size}
                  for p in sorted(args.output.iterdir()) if p.is_file()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Datasets/gemma-3-imatrix-calibration-v1")
    parser.add_argument("--tokens", type=int, default=1000000)
    parser.add_argument("--heldout-tokens", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--datasets-cache", type=Path, default=Path.home() / ".cache/huggingface/datasets")
    parser.add_argument("--verify-tokenizer", type=Path, help="second local Gemma tokenizer snapshot")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
