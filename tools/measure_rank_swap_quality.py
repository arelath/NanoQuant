"""Score exact factor-byte exchanges on teacher text and matched benchmark subsets."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import gguf
import torch
from materialize_nanoquant_gguf import iter_dense_nanoquant_projections
from measure_gguf_kl import PROJECTIONS, swapped, sweep_session
from measure_joint_kl_targets import projection_weights
from transformers import AutoTokenizer

from nanoquant.application.evaluation import model_logits
from nanoquant.application.task_evaluation import (
    MultipleChoiceEvaluationRequest,
    evaluate_multiple_choice,
    pinned_legacy_multiple_choice_tasks,
)
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.hf_mmlu import evaluate_mmlu, prepare_mmlu
from nanoquant.infrastructure.hf_task_evaluation import (
    hash_hf_tokenizer_snapshot,
    prepare_pinned_hf_multiple_choice_inputs,
)
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file

BENCHMARKS = ("arc_easy", "boolq", "mmlu")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_export(path: Path) -> tuple[dict, dict, dict]:
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    weights, groups, costs = {}, defaultdict(list), {}
    for name, projection in iter_dense_nanoquant_projections(reader.fields, tensors, gguf):
        _, block, kind, _ = name.split(".")
        group_kind = projection.group.split(".")[-1]
        owner = f"{block}:{PROJECTIONS.get(group_kind, 'self_attn.attn_qkv')}"
        target = f"{block}:{PROJECTIONS[kind]}"
        if target in weights:
            raise ValueError(f"duplicate projection {target}")
        weights[target] = torch.from_numpy(projection.weight)
        groups[owner].append(target)
        costs[owner] = sum(int(tensors[tensor].n_bytes) for tensor in projection.sources)
    return weights, dict(groups), costs


def prepare_tasks(plan: dict, attribution: dict) -> tuple[dict, object, int, dict]:
    number = next(iter(plan["sources"]))
    directory = Path(plan["sources"][number]).parent
    (quality_path,) = directory.glob(f"{number}-*-compress-and-benchmark-*-quality.json")
    quality = read_json(quality_path)
    reference = {item["result"]["task_name"]: item for item in quality["results"]["frozen"]["tasks"]}
    snapshot = Path(attribution["checkpoint"])
    teacher = attribution["protocol"]["teacher"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer_hash = hash_hf_tokenizer_snapshot(snapshot)
    if tokenizer_hash != quality["protocol"]["tokenizer_hash"]:
        raise ValueError("task tokenizer differs from the benchmark")
    rng = random.Random(plan["sample_seed"])
    prepared_tasks = {}
    indices = {}
    for task in pinned_legacy_multiple_choice_tasks():
        if task.task_name not in BENCHMARKS:
            continue
        prepared = prepare_pinned_hf_multiple_choice_inputs(
            task, tokenizer, tokenizer_name=teacher["source"], tokenizer_revision=teacher["revision"],
            tokenizer_content_hash=tokenizer_hash, maximum_samples=quality["protocol"]["task_limit"],
            local_files_only=True,
        )
        if to_dict(prepared.cache_identity) != reference[task.task_name]["task_input_identity"]:
            raise ValueError(f"{task.task_name} inputs differ from the original benchmark")
        selected = sorted(rng.sample(range(len(prepared.examples)), plan["task_samples"]))
        indices[task.task_name] = selected
        prepared_tasks[task.task_name] = (task, tuple(prepared.examples[i] for i in selected))
    mmlu = prepare_mmlu(tokenizer, local_files_only=True)
    if mmlu.identity != quality["protocol"]["mmlu"]:
        raise ValueError("MMLU prompts differ from the original benchmark")
    groups = defaultdict(list)
    for index, subject in enumerate(mmlu.subjects):
        groups[subject].append(index)
    selected = sorted(i for subject in sorted(groups) for i in rng.sample(groups[subject], plan["mmlu_per_subject"]))
    indices["mmlu"] = selected
    subset = replace(mmlu, tokens=tuple(mmlu.tokens[i] for i in selected),
                     subjects=tuple(mmlu.subjects[i] for i in selected), gold=tuple(mmlu.gold[i] for i in selected))
    return prepared_tasks, subset, quality["protocol"]["pad_token_id"], indices


def score_tasks(session, tasks: dict, mmlu, pad: int) -> dict:
    scores = {}
    device = session.protocol["device"]
    logits = model_logits(session.model)
    for name, (task, examples) in tasks.items():
        result = evaluate_multiple_choice(
            MultipleChoiceEvaluationRequest(task, examples, batch_size=4, pad_token_id=pad, device=device), logits,
        )
        field = "normalized_correct" if result.primary_metric == "acc_norm" else "raw_correct"
        scores[name] = {"accuracy": result.primary_value,
                        "correct": [getattr(row, field) for row in result.examples]}
    result = evaluate_mmlu(session.model, mmlu, device=device, pad_token_id=pad, after_batch=lambda: None)
    scores["mmlu"] = {"accuracy": result["primary_value"],
                     "correct": [row["predicted"] == row["gold"] for row in result["examples"]]}
    return scores


def render(report: dict) -> str:
    lines = [
        "# Conditional rank swaps between Experiments 080 and 083", "",
        "Each swap replaces complete exported projections within an otherwise quantized model.",
        "Shared QKV moves as one unit. Every exchange preserves factor and scale bytes in its base model.",
        "The dense FP16 replay uses clean embeddings and normalization weights.",
        "The task columns show net changes in correct answers on matched subsets.",
        "", "| Base | Donor | Swap | Isolated ΔKL | Joint ΔKL | ARC Easy | BoolQ | MMLU |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in report["results"].items():
        reference = report["results"][item["base"]]
        joint = item["text"]["kl_nats_per_token"] - reference["text"]["kl_nats_per_token"]
        task_deltas = [
            sum(item["benchmark"][task]["correct"]) - sum(reference["benchmark"][task]["correct"])
            for task in BENCHMARKS
        ]
        cells = " | ".join(f"{delta:+d}" for delta in task_deltas)
        lines.append(f"| {item['base']} | {item['donor'] or '—'} | {name} | {item['isolated_delta']:+.6f} "
                     f"| {joint:+.6f} | {cells} |")
    lines += ["", "The JSON includes exact units, bytes, per-window scores, and paired task outcomes.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    attribution = read_json(plan["attribution"])
    prior_joint = read_json(plan["comparison"])
    prior_isolated = read_json(plan["isolated"])["runs"]
    sweep = Path(attribution["measurements"])
    sweep_hash = hash_file(sweep)
    if sweep_hash != attribution["measurement_sha256"] or sweep_hash != prior_joint["sweep_sha256"]:
        raise ValueError("the reference comparisons require the same sensitivity teacher")
    paths = {label: Path(path) for label, path in plan["sources"].items()}
    hashes = {label: hash_file(path) for label, path in paths.items()}
    if any(value != prior_joint["experiments"][label]["gguf"]["sha256"] for label, value in hashes.items()):
        raise ValueError("an export differs from the joint comparison")
    data = args.output.with_suffix(".json")
    dependencies = [Path(__file__), *(Path(__file__).with_name(name) for name in (
        "materialize_nanoquant_gguf.py", "measure_gguf_kl.py", "kl_sensitivity_dataset.py",
    ))]
    identity = {
        "plan": hash_file(args.plan), "exports": hashes, "sweep": sweep_hash,
        "joint": hash_file(Path(plan["comparison"])), "isolated": hash_file(Path(plan["isolated"])),
        "code": {path.name: hash_file(path) for path in dependencies},
    }
    previous = read_json(data) if data.exists() else {}
    if previous and previous["identity"] != identity:
        raise ValueError("the existing report has different inputs")
    tasks, mmlu, pad, indices = prepare_tasks(plan, attribution)
    print("Prepared matched benchmark inputs.", flush=True)
    report = {"identity": identity, "benchmark_indices": indices, "results": previous.get("results", {})}
    with sweep_session(sweep, local_files_only=True) as session:
        weights = projection_weights(session.model, session.adapter, session.targets)
        exports = {label: read_export(path) for label, path in paths.items()}
        first, second = exports
        groups = exports[first][1]
        if groups != exports[second][1] or set(weights) != set(exports[first][0]):
            raise ValueError("export projection inventories differ")
        old, new = exports[first][2], exports[second][2]
        for key, group in plan["swaps"].items():
            if len(group) != len(set(group)) or sum(new[unit] - old[unit] for unit in group):
                raise ValueError(f"{key} does not preserve exact factor bytes")
        embedding = session.model.get_input_embeddings().weight
        original_embedding = embedding.detach().cpu().clone()
        for base, donor in ((first, second), (second, first)):
            base_weights = exports[base][0]
            cases = [(base, []), *[(f"{base}/{name}", units) for name, units in plan["swaps"].items()]]
            with swapped({key: (weights[key], value) for key, value in base_weights.items()}):
                for name, units in cases:
                    if name in report["results"]:
                        continue
                    started = time.perf_counter()
                    targets = [target for unit in units for target in groups[unit]]
                    replacement = {target: (weights[target], exports[donor][0][target]) for target in targets}
                    print(f"Scoring {name}", flush=True)
                    with swapped(replacement):
                        text_score = session.score()
                        if not units:
                            measured = text_score["kl_nats_per_token"] - session.clean["kl_nats_per_token"]
                            expected = prior_joint["experiments"][base]["gguf"]["variants"]["projections"]
                            if not math.isclose(measured, expected["kl_above_control"], abs_tol=2e-5):
                                raise ValueError(f"{base}: joint KL control {measured:.8f} differs from "
                                                 f"{expected['kl_above_control']:.8f}")
                        benchmark = score_tasks(session, tasks, mmlu, pad)
                    isolated_delta = math.fsum(
                        prior_isolated[donor]["values"][target] - prior_isolated[base]["values"][target]
                        for target in targets
                    )
                    report["results"][name] = {
                        "base": base, "donor": donor if units else None, "units": units,
                        "bytes": {unit: {label: exports[label][2][unit] for label in paths} for unit in units},
                        "text": text_score, "benchmark": benchmark, "isolated_delta": isolated_delta,
                        "seconds": time.perf_counter() - started,
                    }
                    atomic_write_json(data, report)
                    atomic_write_text(args.output, render(report))
        if not torch.equal(embedding.detach().cpu(), original_embedding):
            raise RuntimeError("embedding changed during the comparison")
    report["complete"] = True
    atomic_write_json(data, report)
    print(f"Report written to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
