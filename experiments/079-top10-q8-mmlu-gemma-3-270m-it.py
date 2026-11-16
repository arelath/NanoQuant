"""Experiment 079: matched five-shot MMLU with ten Q8_0 projection replacements."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import gguf
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from materialize_nanoquant_gguf import dense_nanoquant_projections  # noqa: E402
from measure_gguf_kl import PROJECTIONS, check_norms, map_tensors, read_gguf  # noqa: E402

OUT = ROOT / "outputs/079"
RESULT = ROOT / "Results/079"
ORIGINAL = ROOT / "Results/074/gemma-3-270m-it-nanoquant-pre-kd.gguf"
COPY = RESULT / ORIGINAL.name
HYBRID = RESULT / "gemma-3-270m-it-nanoquant-pre-kd-top10-q8_0-materialized.gguf"
SOURCE = "unsloth/gemma-3-270m-it"
REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
LETTERS = "ABCD"


def source_snapshot():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=SOURCE, revision=REVISION, local_files_only=True))


def prepare(snapshot):
    from safetensors import safe_open

    ranking_path = ROOT / "outputs/074/074-imatrix-weighted-kl.json"
    ranking = json.loads(ranking_path.read_bytes())
    source_hash = hash_file(ORIGINAL)
    (arm,) = (r for r in ranking["results"] if r["sha256"] == source_hash)
    assert len(arm["projections"]) == 126
    selected = sorted(arm["projections"].items(), key=lambda p: (-p[1]["imatrix_gaussian_kl"], p[0]))[:10]
    reverse = {v: k for k, v in PROJECTIONS.items()}
    replacements = {}
    for key, row in selected:
        block, part = key.split(":")
        name = f"blk.{block}.{reverse[part]}.weight"
        replacements[name] = {"key": key, "name": name, "imatrix_kl": row["imatrix_gaussian_kl"]}

    identity = {
        "baseline_sha256": source_hash,
        "source_snapshot": str(snapshot),
        "source_safetensors_sha256": hash_file(snapshot / "model.safetensors"),
        "ranking_sha256": hash_file(ranking_path),
        "replacements": list(replacements.values()),
    }
    receipt = OUT / "models.json"
    if receipt.exists():
        cached = json.loads(receipt.read_bytes())
        if (
            cached["identity"] == identity
            and hash_file(COPY) == source_hash
            and hash_file(HYBRID) == cached["hybrid_sha256"]
        ):
            return cached
        raise RuntimeError("Existing 079 artifact identity differs")
    if COPY.exists():
        assert hash_file(COPY) == source_hash
    else:
        shutil.copyfile(ORIGINAL, COPY)
        assert hash_file(COPY) == source_hash

    reader = gguf.GGUFReader(COPY)
    stored = {t.name: t for t in reader.tensors}
    dense = dense_nanoquant_projections(reader.fields, stored, gguf)
    assert len(dense) == 126 and replacements.keys() <= dense.keys()
    writer = gguf.GGUFWriter(None, reader.fields["general.architecture"].contents(), endianess=reader.endianess)
    for name, field in reader.fields.items():
        if name.startswith(("GGUF.", "quantization.nanoquant.")) or name in {
            "general.architecture",
            "general.file_type",
        }:
            continue
        if name == "general.alignment":
            writer.add_custom_alignment(int(field.contents()))
        else:
            value_type = field.types[0]
            subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(name, field.contents(), value_type, sub_type=subtype)

    q8_type = gguf.GGMLQuantizationType.Q8_0
    replaced_payloads = {}
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as ref:
        for name, projection in dense.items():
            if name in replacements:
                block, path = replacements[name]["key"].split(":")
                original = ref.get_tensor(f"model.layers.{block}.{path}.weight").float().numpy()
                assert original.shape == projection.weight.shape and original.shape[-1] % 32 == 0
                payload = gguf.quantize(original, q8_type)
                replaced_payloads[name] = payload
                writer.add_tensor(name, payload, raw_dtype=q8_type)
            else:
                writer.add_tensor(name, projection.weight)
    sidecars = {name for projection in dense.values() for name in projection.sources}
    untouched = {name: tensor for name, tensor in stored.items() if name not in sidecars}
    for name, tensor in untouched.items():
        assert ".nq_" not in name
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
    temp = HYBRID.with_suffix(".pending.gguf")
    writer.write_header_to_file(temp)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    checked = {t.name: t for t in gguf.GGUFReader(temp).tensors}
    assert checked.keys() == dense.keys() | untouched.keys()
    for name, tensor in checked.items():
        if name in replacements:
            assert tensor.tensor_type == q8_type and np.array_equal(tensor.data, replaced_payloads[name])
        elif name in dense:
            assert tensor.tensor_type == gguf.GGMLQuantizationType.F32
            assert np.array_equal(tensor.data, dense[name].weight)
        else:
            assert tensor.tensor_type == untouched[name].tensor_type
            assert np.array_equal(tensor.data, untouched[name].data)
    del tensor, checked
    temp.replace(HYBRID)
    info = {
        "identity": identity,
        "hybrid_sha256": hash_file(HYBRID),
        "baseline_bytes": COPY.stat().st_size,
        "hybrid_bytes": HYBRID.stat().st_size,
    }
    atomic_write_json(receipt, info)
    return info


def prepare_questions(tokenizer):
    meta = json.loads((OUT / "datasets.json").read_bytes())["mmlu"]
    rows = pq.read_table(meta["files"]["test"]).to_pylist()
    examples = pq.read_table(meta["files"]["dev"]).to_pylist()
    by_subject = defaultdict(list)
    for example in examples:
        by_subject[example["subject"]].append(example)

    def format_question(row):
        options = row["choices"]
        gold = row["answer"]
        assert len(options) == len(LETTERS) and 0 <= gold < len(options)
        lines = [row["question"].strip()]
        lines.extend(f"{letter}. {value.strip()}" for letter, value in zip(LETTERS, options, strict=True))
        return "\n".join(lines) + "\nAnswer:", len(options), gold

    prefixes = {}
    for subject, group in by_subject.items():
        assert len(group) == 5
        examples = [format_question(row) for row in group[:5]]
        header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        prefixes[subject] = header + "".join(f"{prompt} {LETTERS[gold]}\n\n" for prompt, _, gold in examples)
    records = []
    prompts = []
    for index, row in enumerate(rows):
        prompt, count, gold = format_question(row)
        prompts.append(prefixes[row["subject"]] + prompt)
        records.append((index, row["subject"], count, gold))
    assert len(records) == 14042 and len(prefixes) == 57
    tokens = tokenizer(prompts, add_special_tokens=True, verbose=False).input_ids
    label_ids = []
    for label in LETTERS:
        prefix = tokenizer.encode("Answer:", add_special_tokens=False)
        joined = tokenizer.encode(f"Answer: {label}", add_special_tokens=False)
        assert joined[:-1] == prefix
        label_ids.append(joined[-1])
    assert len(set(label_ids)) == len(label_ids)
    identity = {
        "dataset": meta["repo"],
        "revision": meta["revision"],
        "split_sha256": {k: hash_file(Path(v)) for k, v in meta["files"].items()},
        "method": "five same-subject examples, direct next-token answer letter, restricted to available choices",
        "prompt_token_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
        "answer_label_ids": label_ids,
        "max_prompt_tokens": max(map(len, tokens)),
    }
    return identity, records, tokens


def load_model(snapshot, gguf_path):
    model = load_causal_language_model(
        snapshot,
        torch_dtype=torch.bfloat16,
        attention_implementation="sdpa",
        local_files_only=True,
    ).eval()
    if gguf_path is not None:
        from nanoquant.infrastructure.model_adapters import adapter_for_config

        adapter = adapter_for_config(json.loads((snapshot / "config.json").read_bytes()))
        layers = adapter.get_decoder_layers(model)
        targets = [f"{block}:{part}" for block in range(18) for part in PROJECTIONS.values()]
        projections, embedding, norms = map_tensors(read_gguf(gguf_path), targets)
        assert embedding is not None and model.config.tie_word_embeddings
        assert check_norms(model, adapter, norms) <= 0.004
        with torch.no_grad():
            for key, proj in projections.items():
                block, path = key.split(":")
                weight = layers[int(block)].get_submodule(path).weight
                assert weight.shape == proj["values"].shape
                weight.copy_(proj["values"])
            model.get_input_embeddings().weight.copy_(embedding["values"])
    return model.to("cuda:0")


@torch.inference_mode()
def evaluate(label, snapshot, gguf_path, data, pad_id, max_examples=None):
    protocol, rows, tokens = data
    prototype = max_examples is not None
    indices = sorted(range(min(len(rows), max_examples or len(rows))), key=lambda i: len(tokens[i]))
    output = OUT / (label + ".json")
    model_identity = hash_file(gguf_path) if gguf_path else hash_file(snapshot / "model.safetensors")
    identity = {
        "protocol": protocol,
        "weights_sha256": model_identity,
        "source": SOURCE,
        "revision": REVISION,
        "arithmetic": "Transformers BF16 SDPA",
        "torch": str(torch.__version__),
        "script_sha256": hash_file(Path(__file__)),
    }
    if not prototype and output.exists():
        result = json.loads(output.read_bytes())
        if result["identity"] == identity and len(result["examples"]) == len(rows):
            return result
        raise RuntimeError(f"Existing result identity differs for {label}")
    model = load_model(snapshot, gguf_path)
    results = {}
    start = time.perf_counter()
    labels = torch.tensor(protocol["answer_label_ids"], device="cuda:0")
    for start_index in range(0, len(indices), 32):
        batch = indices[start_index : start_index + 32]
        batch_tokens = [tokens[i] for i in batch]
        width = max(map(len, batch_tokens))
        ids = torch.tensor([[pad_id] * (width - len(row)) + row for row in batch_tokens], device="cuda:0")
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in batch_tokens], device="cuda:0")
        pos = mask.cumsum(dim=1).sub(1).clamp_min(0)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1).logits
        answers = logits[:, -1].index_select(1, labels).float().cpu()
        for i, values in zip(batch, answers, strict=True):
            qid, subject, num_choices, gold = rows[i]
            pred = int(values[:num_choices].argmax())
            results[i] = {"id": qid, "subject": subject, "gold": gold, "predicted": pred}
        if len(results) % 1024 == 0 or len(results) == len(indices):
            print(f"{label}: {len(results)}/{len(indices)} in {time.perf_counter() - start:.1f}s", flush=True)
    counts = Counter(r["subject"] for r in results.values())
    correct = Counter(r["subject"] for r in results.values() if r["gold"] == r["predicted"])
    result = {
        "label": label,
        "identity": identity,
        "count": len(indices),
        "correct": sum(correct.values()),
        "seconds": time.perf_counter() - start,
        "subjects": {s: {"count": counts[s], "correct": correct[s]} for s in sorted(counts)},
        "examples": [results[i] for i in sorted(results)],
    }
    if not prototype:
        atomic_write_json(output, result)
    del model
    torch.cuda.empty_cache()
    return result


def render(results, models):
    lines = [
        "# Experiment 079 | Gemma 3 270M IT and top-ten Q8_0 replacements",
        "",
        "The NanoQuant control is a byte-for-byte copy of experiment 074 before global KL distillation.",
        "The hybrid replaces the ten projections with the highest pre-KD imatrix-weighted KL estimate "
        "with Q8_0 weights quantized from the original BF16 checkpoint.",
        "The other 116 projections in the hybrid GGUF store the same reconstructed FP32 weights as the staged "
        "control. The embedding and norms retain their original stored bytes.",
        "The standalone hybrid GGUF materializes the remaining projections, which increases file size. "
        "All three models are evaluated with the same dense BF16 Transformers backend.",
        "",
        "Standard MMLU uses the full 14,042-question test set across 57 subjects, with five dev examples per subject.",
        "The model selects the answer letter A through D with the largest next-token logit.",
        "",
        "| Model | Benchmark | Correct / total | Accuracy |",
        "| --- | --- | ---: | ---: |",
    ]
    for label, r in results:
        lines.append(f"| {label} | MMLU | {r['correct']:,} / {r['count']:,} | {r['correct'] / r['count']:.2%} |")
    if len(results) == 3:
        base, bf16, hybrid = (r for _, r in results)
        assert base["identity"]["protocol"] == bf16["identity"]["protocol"] == hybrid["identity"]["protocol"]
        gained = lost = 0
        for a, b, ref in zip(base["examples"], hybrid["examples"], bf16["examples"], strict=True):
            assert (
                (a["id"], a["subject"], a["gold"])
                == (b["id"], b["subject"], b["gold"])
                == (ref["id"], ref["subject"], ref["gold"])
            )
            gained += int(a["predicted"] != a["gold"] and b["predicted"] == b["gold"])
            lost += int(a["predicted"] == a["gold"] and b["predicted"] != b["gold"])
        assert gained - lost == hybrid["correct"] - base["correct"]
        delta = 100 * (gained - lost) / base["count"]
        lines += [
            "",
            f"The matched MMLU change was **{delta:+.2f} percentage points**. "
            f"The hybrid fixed {gained:,} answers and missed {lost:,} the control got right.",
            "",
            "| MMLU subject | Questions | BF16 | Control | Q8_0 hybrid |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for subject, a in base["subjects"].items():
            b = hybrid["subjects"][subject]
            ref = bf16["subjects"][subject]
            assert a["count"] == b["count"] == ref["count"]
            lines.append(
                f"| {subject} | {a['count']:,} | {ref['correct'] / ref['count']:.2%} | "
                f"{a['correct'] / a['count']:.2%} | {b['correct'] / b['count']:.2%} |"
            )
    lines += [
        "",
        "## Replacements",
        "",
        "| Rank | Block | Projection | Estimated KL (nats/token) |",
        "| ---: | ---: | --- | ---: |",
    ]
    for i, row in enumerate(models["identity"]["replacements"], 1):
        block, name = row["key"].split(":")
        lines.append(f"| {i} | {block} | `{name}` | {row['imatrix_kl']:.6f} |")
    lines += [
        "",
        "Block indices are zero-based.",
        "",
        f"- Control GGUF: {models['baseline_bytes']:,} bytes, SHA-256 `{models['identity']['baseline_sha256']}`.",
        f"- Hybrid GGUF: {models['hybrid_bytes']:,} bytes, SHA-256 `{models['hybrid_sha256']}`.",
        f"- BF16 snapshot: `{SOURCE}@{REVISION}`.",
        "- [MMLU dataset](https://huggingface.co/datasets/cais/mmlu).",
        "- Inputs, model hashes, per-subject counts and per-question predictions are under `outputs/079`.",
        "",
    ]
    atomic_write_text(RESULT / "079-mmlu-quality.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--smoke", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True, parents=True)
    RESULT.mkdir(exist_ok=True, parents=True)
    snapshot = source_snapshot()
    models = prepare(snapshot)
    if args.setup_only:
        render([], models)
        return
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    mmlu = prepare_questions(tokenizer)
    results = []
    torch.backends.cuda.matmul.allow_tf32 = False
    with acquire_device_lease("cuda:0"):
        for title, label, file in (
            ("074 pre-KD copy", "pre-kd-mmlu", COPY),
            ("BF16", "bf16-standard-mmlu", None),
            ("079 top-ten Q8_0", "q8-top10-mmlu", HYBRID),
        ):
            result = evaluate(label, snapshot, file, mmlu, tokenizer.pad_token_id, args.smoke)
            results.append((title, result))
            if args.smoke is None:
                render(results, models)


if __name__ == "__main__":
    main()
