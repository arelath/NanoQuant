"""Frozen objective-arm PTB transfer and protocol-matched standard task scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import torch
from objective_ablation_common import install_dense, load_scales, normalized_hidden_loss
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_distillation_objectives import ARMS, load_models
from probe_non_wikitext_kd_quality import _token_hash
from probe_residual_stage_functional import BASELINE_SHA, REVISION, aggregate, file_hash
from safetensors.torch import load_file
from validate_evaluation_slice_registry import validate_registry

from nanoquant.application.kl_budget import causal_kl_nll_per_sequence_from_logits
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.quality_evaluation import QualityEvaluationRequest, _evaluate_model, prepare_quality_inputs


def paired_task_accuracy(before: dict, after: dict) -> dict:
    baseline = {task["result"]["task_name"]: task for task in before["tasks"]}
    candidate = {task["result"]["task_name"]: task for task in after["tasks"]}
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("task coverage differs")
    generator = np.random.default_rng(0)
    bootstraps, deltas = [], []
    for name, task in baseline.items():
        other = candidate[name]
        left, right = task["result"], other["result"]
        if (task["task_input_identity"] != other["task_input_identity"]
                or left["primary_metric"] != right["primary_metric"]):
            raise ValueError("task input or metric differs")
        key = {"acc": "raw_correct", "acc_norm": "normalized_correct"}[left["primary_metric"]]
        previous = {item["sample_id"]: item for item in left["examples"]}
        current = {item["sample_id"]: item for item in right["examples"]}
        if set(previous) != set(current) or not previous:
            raise ValueError("paired task sample identities differ")
        values = np.array([int(current[sample][key]) - int(item[key]) for sample, item in previous.items()])
        deltas.append(float(values.mean()))
        bootstraps.append(values[generator.integers(0, len(values), size=(10000, len(values)))].mean(1))
    distribution = np.mean(bootstraps, axis=0)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return dict(point_delta_pp=100 * float(np.mean(deltas)), lower_delta_pp=100 * float(low),
                upper_delta_pp=100 * float(high), confidence=0.95, resamples=10000,
                task_count=len(deltas), method="paired correctness bootstrap within each task; equal task weighting")


def run(args):
    torch.set_num_threads(8)
    if file_hash(args.output / "common060.gguf") != BASELINE_SHA:
        raise ValueError("common student identity changed")
    token_receipt = json.loads(args.tokens.with_suffix(".json").read_text())
    entry = token_receipt["entry"]
    registry = json.loads(Path(token_receipt["registry"]).read_text())
    validate_registry(registry)
    tokens = load_file(str(args.tokens))["input_ids"].long()
    if (file_hash(args.tokens) != token_receipt["file_sha256"] or _token_hash(tokens) != entry["token_hash"]
            or entry["retirement"]["role"] != "transfer" or not token_receipt["retired_before_evaluation"]
            or next(item for item in registry["slices"] if item["id"] == entry["id"]) != entry):
        raise ValueError("PTB provenance/retirement mismatch")
    states = {"untreated060": load_file(str(args.output / "common-scales.safetensors"))}
    hashes = {}
    for arm in ARMS:
        path = args.output / arm / "scales.safetensors"
        receipt = json.loads(path.with_name("receipt.json").read_text())
        if file_hash(path) != receipt["sha256"] or receipt["steps"] != 256:
            raise ValueError("transfer requires frozen 256-step objective endpoints")
        states[arm] = load_file(str(path))
        hashes[arm] = file_hash(path)
    folder = args.output / "transfer"
    folder.mkdir(exist_ok=True)
    request = QualityEvaluationRequest(
        snapshot=args.snapshot, source="google/gemma-3-1b-it", revision=REVISION, run_output=folder,
        backend="dense", use_global_tuning=False, device=args.device,
        wikitext_samples=64, wikitext_sequence_length=128, wikitext_batch_size=1,
        task_names=("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq"),
        task_limit=200, task_batch_size=1, local_files_only=True)
    identity = dict(training_protocol_sha256=file_hash(args.output / "protocol.json"),
                    tokens=token_receipt, frozen_scales=hashes, tool_sha256=file_hash(Path(__file__)),
                    quality_request=to_dict(request),
                    gate="PTB paired NLL 95% upper <= +0.01; task comparison is standard 200-example replay")
    path = folder / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("transfer resume identity mismatch")
    atomic_write_json(path, identity)
    progress = folder / "ptb-progress.json"
    rows = json.loads(progress.read_text())["rows"] if progress.exists() else []
    inputs = prepare_quality_inputs(request, progress=lambda event, fields:
                                    print(json.dumps(dict(preparation=event, **fields)), flush=True))
    atomic_write_json(folder / "quality-inputs.json", dict(
        wikitext_fingerprint=inputs.wikitext_fingerprint, wikitext_token_hash=_token_hash(inputs.wikitext_tokens),
        tokenizer_hash=inputs.tokenizer_hash,
        tasks=[dict(identity=to_dict(task.cache_identity), examples=len(task.examples)) for task in inputs.tasks]))
    with acquire_device_lease(args.device):
        teacher, student, units, _ = load_models(args, trainable=False)
        with torch.inference_mode():
            # Measure the canonical reconstruction's numerical change from the
            # unchanged CPU GGUF recovery before attributing objective effects.
            audit_batch = tokens[:1].to(args.device)
            teacher_logits = teacher(input_ids=audit_batch, use_cache=False).logits
            restored = student(input_ids=audit_batch, use_cache=False).logits
            before = causal_kl_nll_per_sequence_from_logits(
                teacher_logits, restored, audit_batch, token_chunk_size=32)[0]
            load_scales(units, states["untreated060"])
            install_dense(student, units)
            canonical = student(input_ids=audit_batch, use_cache=False).logits
            after = causal_kl_nll_per_sequence_from_logits(
                teacher_logits, canonical, audit_batch, token_chunk_size=32)[0]
            atomic_write_json(folder / "canonicalization-audit.json", dict(before=to_dict(before), after=to_dict(after),
                              kl_delta=after.kl_nats_per_token - before.kl_nats_per_token,
                              nll_delta=after.negative_log_likelihood - before.negative_log_likelihood))
            del teacher_logits, restored, canonical
            for arm, state in states.items():
                load_scales(units, state)
                install_dense(student, units)
                for index, sequence in enumerate(tokens):
                    if index < len(rows) and arm in rows[index]:
                        continue
                    batch = sequence[None].to(args.device)
                    teacher_output = teacher(input_ids=batch, use_cache=False, output_hidden_states=True)
                    student_output = student(input_ids=batch, use_cache=False, output_hidden_states=True)
                    teacher_logits, logits = teacher_output.logits, student_output.logits
                    metric = causal_kl_nll_per_sequence_from_logits(
                        teacher_logits, logits, batch, token_chunk_size=32)[0]
                    if index == len(rows):
                        rows.append({})
                    rows[index][arm] = to_dict(metric)
                    rows[index][arm + "_hidden_nmse"] = float(normalized_hidden_loss(
                        student_output.hidden_states, teacher_output.hidden_states))
                    del teacher_logits, logits, teacher_output, student_output
                    atomic_write_json(progress, dict(status="running", rows=rows))
                    if (index + 1) % 24 == 0:
                        print(json.dumps(dict(ptb_arm=arm, sequences=index + 1, total=len(tokens))), flush=True)
        results = {arm: aggregate(arm, rows) for arm in states}
        pairs = [("untreated060", arm) for arm in ARMS] + [("top_k", "full_kl"), ("top_k", "full_kl_hidden")]
        comparisons = {after + "_minus_" + before:
                       _paired_metric_payload(results[before], results[after], "negative_log_likelihood")
                       for before, after in pairs}
        atomic_write_json(folder / "ptb-summary.json", dict(status="completed",
                          results={arm: dict(**to_dict(result), perplexity=math.exp(result.negative_log_likelihood))
                                   for arm, result in results.items()}, paired_nll=comparisons,
                          mean_hidden_nmse={arm: sum(row[arm + "_hidden_nmse"] for row in rows) / len(rows)
                                            for arm in states},
                          interpretation="Gemma BPE perplexity on processed PTB; not original word-level PTB PPL"))
        atomic_write_json(progress, dict(status="completed", rows=rows))
        teacher.to("cpu")
        torch.cuda.empty_cache()
        for arm, state in states.items():
            path = folder / (arm + "-quality.json")
            if path.exists():
                continue
            with torch.inference_mode():
                load_scales(units, state)
                install_dense(student, units)
                result = _evaluate_model(arm, student, request, inputs, progress=lambda event, fields:
                                         print(json.dumps(dict(quality=event, **fields)), flush=True))
            atomic_write_json(path, result)
        scores = {arm: json.loads((folder / (arm + "-quality.json")).read_text()) for arm in states}
        task_pairs = {after + "_minus_" + before: paired_task_accuracy(scores[before], scores[after])
                      for before, after in pairs}
        atomic_write_json(folder / "task-paired.json", dict(paired=task_pairs))
        atomic_write_json(folder / "summary.json", dict(status="completed", quality_files=
                          {arm: dict(path=str(folder / (arm + "-quality.json")),
                                     sha256=file_hash(folder / (arm + "-quality.json"))) for arm in states},
                          ptb_summary_sha256=file_hash(folder / "ptb-summary.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="existing objective training root")
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, default=Path("D:/dev/research/llama.cpp"))
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
