"""Compare a Transformers checkpoint with GGUF models on the full pinned quality suite."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import torch
import transformers
from benchmark_hf_model import ROOT, _gguf_parts, _progress, _source
from materialize_nanoquant_gguf import materialize_gguf
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file
from nanoquant.llamacpp_quality import (
    LlamaCppQualityRequest,
    _quality_sequences,
    _read_scores,
    _resolve_runner,
    _run,
    _runtime_files,
    _Score,
    _Sequence,
    _write_input,
    execute_llamacpp_quality_evaluation,
)
from nanoquant.quality_evaluation import (
    DEFAULT_QUALITY_TASKS,
    PreparedQualityInputs,
    QualityEvaluationRequest,
    describe_quality_protocol,
    execute_quality_evaluation,
    prepare_quality_inputs,
)


def _file_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": hash_file(path)}


def _checkpoint_files(snapshot: Path) -> list[dict[str, Any]]:
    weights = sorted(snapshot.glob("model*.safetensors")) or sorted(snapshot.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError(f"No Transformers checkpoint weights found in {snapshot}")
    return [_file_record(path) for path in weights]


def _validate_gguf_vocab(paths: list[Path], snapshot: Path, root: Path, inputs: PreparedQualityInputs) -> None:
    sys.path.insert(0, str(root / "gguf-py"))
    try:
        reader = importlib.import_module("gguf").GGUFReader
    finally:
        sys.path.pop(0)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokens = inputs.wikitext_tokens
    if inputs.wikitext_attention_mask is not None:
        tokens = tokens[inputs.wikitext_attention_mask]
    used = set(tokens.flatten().tolist())
    for task in inputs.tasks:
        for example in task.examples:
            for sequence in (*example.contexts, *example.continuations):
                used.update(sequence)
    ids = sorted(used)
    labels = tokenizer.convert_ids_to_tokens(ids)
    for path in paths:
        vocab = reader(path, "r").fields["tokenizer.ggml.tokens"].contents()
        for index, label in zip(ids, labels, strict=True):
            if index >= len(vocab):
                raise ValueError(f"{path.name} has no token ID {index} from the shared tokenizer")
            token = vocab[index]
            # SentencePiece GGUF exporters can store whitespace-only pieces as literal spaces.
            space_equivalent = isinstance(label, str) and set(label) == {"▁"} and token == " " * len(label)
            if token != label and not space_equivalent:
                raise ValueError(f"{path.name} has a different token at shared tokenizer ID {index}")
        print(f"Verified {len(ids):,} scored-input token IDs against {path.name}.", flush=True)


def _validate_result(result: dict[str, Any], inputs: PreparedQualityInputs) -> None:
    windows = inputs.wikitext_tokens
    expected_tokens = (
        windows.numel() if inputs.wikitext_attention_mask is None else int(inputs.wikitext_attention_mask.sum())
    ) - len(windows)
    wt = result["wikitext"]
    if (
        wt["token_count"] != expected_tokens
        or wt["window_count"] != len(windows)
        or not math.isfinite(wt["perplexity"])
    ):
        raise ValueError("Incomplete WikiText-2 results")
    for scored, task in zip(result["tasks"], inputs.tasks, strict=True):
        observed = scored["result"]
        if (
            observed["sample_count"] != len(task.examples)
            or scored["task_input_identity"] != to_dict(task.cache_identity)
            or observed["primary_metric"] != task.task.primary_metric
            or not math.isfinite(observed["primary_value"])
            or [example["sample_id"] for example in observed["examples"]] != [item.sample_id for item in task.examples]
        ):
            raise ValueError(f"Incomplete or mismatched results for {task.task.task_name}")


def _render_comparison(summary: dict[str, Any]) -> str:
    models = summary["models"]
    reference = models[0]["scores"]
    lines = [
        "# Full model quality comparison", "",
        "Each model scored the same prepared inputs from the complete pinned task splits and WikiText-2 test.",
        "", "## Quality", "",
        "| Benchmark | Metric | Examples or tokens | " + " | ".join(model["label"] for model in models) + " |",
        "| --- | --- | ---: | " + " | ".join("---:" for _ in models) + " |",
        f"| WikiText-2 | perplexity ↓ | {reference['wikitext']['token_count']:,} | "
        + " | ".join(f"{model['scores']['wikitext']['perplexity']:.6f}" for model in models) + " |",
    ]
    for index, task in enumerate(reference["tasks"]):
        lines.append(
            f"| {task['task_name']} | {task['primary_metric']} ↑ | {task['sample_count']:,} | "
            + " | ".join(f"{model['scores']['tasks'][index]['primary_value']:.2%}" for model in models) + " |"
        )
    lines.append("| Six-task mean | accuracy ↑ | — | " + " | ".join(f"{m['task_mean']:.2%}" for m in models) + " |")
    lines.extend((
        "", "## Storage and execution", "",
        "| Model | Backend | Model bytes | Elapsed seconds |",
        "| --- | --- | ---: | ---: |",
    ))
    for model in models:
        lines.append(
            f"| {model['label']} | {model['backend']} | {model['bytes']:,} | {model['elapsed_seconds']:.2f} |"
        )
    for model in models:
        if "materialization" in model:
            validation = model["materialization"]["validation"]
            lines.extend((
                "",
                f"{model['label']} was scored using FP32 projections reconstructed from the packed GGUF. "
                "Its reported size is the original packed file. "
                f"The paired check against the packed runtime used {validation['sequences']} sequences "
                f"and found a maximum mean NLL difference of {validation['max_abs_mean_nll']:.6f} per target token.",
            ))
    lines.extend(("", "## Provenance", ""))
    for model in models:
        lines.append(
            f"- **{model['label']}** `{model['source']}` at `{model['revision']}`. "
            f"[Raw scores]({model['report']})"
        )
    return "\n".join((*lines, ""))


def _model_summary(
    label: str, source: str, revision: str, backend: str, size: int, report: Path, scores: dict[str, Any],
) -> dict[str, Any]:
    tasks = [
        {key: task["result"][key] for key in ("task_name", "sample_count", "primary_metric", "primary_value")}
        for task in scores["tasks"]
    ]
    return {
        "label": label, "source": source, "revision": revision, "backend": backend, "bytes": size,
        "report": report.name, "elapsed_seconds": scores["elapsed_seconds"],
        "task_mean": fmean(task["primary_value"] for task in tasks),
        "scores": {"wikitext": scores["wikitext"], "tasks": tasks},
    }


def _validate_materialized_scores(
    request: LlamaCppQualityRequest,
    original: Path,
    quality: QualityEvaluationRequest,
    inputs: PreparedQualityInputs,
    materialization: dict[str, Any],
) -> dict[str, Any]:
    runner = _resolve_runner(request)
    sequences, tasks, _ = _quality_sequences(quality, inputs)
    selected: list[_Sequence] = []
    offset = 0
    for length, count in [(len(inputs.wikitext_tokens), 8), *((len(task), 4) for task in tasks)]:
        count = min(count, length)
        selected.extend(sequences[offset + index * length // count] for index in range(count))
        offset += length
    sample = tuple(selected)
    receipt_path = request.gguf.with_suffix(".paired-validation.json")
    with tempfile.TemporaryDirectory(prefix=".validate-materialized-", dir=request.output.parent) as temp:
        input_path = Path(temp) / "input.bin"
        input_hash = _write_input(input_path, sample)
        identity = {
            "input_sha256": input_hash, "original_sha256": materialization["source_sha256"],
            "materialized_sha256": materialization["materialized_sha256"],
            "runtime_files": [_file_record(path) for path in _runtime_files(request.llama_cpp_root, runner)],
            "device": request.device, "parallel": request.parallel, "max_allowed_abs_mean_nll": 0.02,
        }
        if receipt_path.is_file():
            existing = cast(dict[str, Any], json.loads(receipt_path.read_text(encoding="utf-8")))
            if existing["identity"] == identity and existing["passed"]:
                return existing
        print(f"Checking {len(sample)} sequences against the packed {original.name}.", flush=True)
        scores: list[tuple[_Score, ...]] = []
        with acquire_device_lease(request.device):
            for model in (original, request.gguf):
                score_path = Path(temp) / f"scores-{len(scores)}.bin"
                _run(replace(request, gguf=model), runner, input_path, score_path)
                scores.append(_read_scores(score_path, len(sample)))
        errors = []
        for packed, dense in zip(*scores, strict=True):
            if packed.token_count != dense.token_count:
                raise ValueError("Packed and reconstructed evaluations scored different targets")
            errors.append(abs(packed.negative_log_likelihood - dense.negative_log_likelihood) / packed.token_count)
        result = {
            "identity": identity, "sequences": len(sample), "max_abs_mean_nll": max(errors),
            "mean_abs_mean_nll": fmean(errors), "passed": max(errors) <= identity["max_allowed_abs_mean_nll"],
        }
        atomic_write_json(receipt_path, result)
        if not result["passed"]:
            raise ValueError(f"FP32 reconstruction differs from packed scores. See {receipt_path}")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Transformers model ID or local checkpoint directory")
    parser.add_argument("--baseline-revision", help="Hugging Face baseline commit, tag, or branch")
    parser.add_argument("--gguf", nargs=2, action="append", required=True, metavar=("LABEL", "MODEL"),
                        help="Repeat for each GGUF. Pass a local path or a Hugging Face file URL.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for reports and resume receipts")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--llama-cpp-root", type=Path,
                        default=Path(os.environ.get("NANOQUANT_LLAMA_CPP_ROOT", ROOT.parent / "llama.cpp")))
    parser.add_argument("--llama-cpp-runner", type=Path)
    parser.add_argument("--parallel", type=int, default=4, help="Concurrent GGUF sequences")
    parser.add_argument("--materialize-nanoquant", action="append", default=[], metavar="LABEL",
                        help="Score a staged NanoQuant GGUF via verified FP32 reconstruction")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached benchmark datasets")
    args = parser.parse_args()
    labels = [label for label, _ in args.gguf]
    if len(set(labels)) != len(labels) or any(
        label.lower() in {"baseline", "comparison"} or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label) is None
        for label in labels
    ):
        parser.error("Use distinct lowercase GGUF labels made of letters, digits, hyphens, and underscores.")
    if args.parallel <= 0:
        parser.error("--parallel must be positive.")
    if not set(args.materialize_nanoquant) <= set(labels):
        parser.error("--materialize-nanoquant must name a label supplied with --gguf.")
    snapshot, source, revision, _ = _source(args.baseline, args.baseline_revision)
    if not snapshot.is_dir():
        parser.error("--baseline must identify a Transformers checkpoint.")
    candidates = [(label, *_source(value, None)[:3]) for label, value in args.gguf]
    if any(not path.is_file() or path.suffix.lower() != ".gguf" for _, path, _, _ in candidates):
        parser.error("Each --gguf must identify a GGUF file.")

    output = args.output.resolve()
    request = QualityEvaluationRequest(
        snapshot=snapshot, source=source, revision=revision, run_output=output, device=args.device,
        wikitext_samples=None, wikitext_sequence_length=128, task_names=DEFAULT_QUALITY_TASKS, task_limit=None,
        local_files_only=args.local_files_only,
    )
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).get_text_config()
    context = getattr(config, "max_position_embeddings", None)
    if isinstance(context, int) and context < request.wikitext_sequence_length:
        raise ValueError("The baseline context is too short for the WikiText-2 protocol")
    inputs = prepare_quality_inputs(request, progress=_progress)
    if isinstance(context, int):
        inputs = replace(inputs, tasks=tuple(
            replace(task, task=replace(task.task, maximum_length=min(context, task.task.maximum_length)))
            for task in inputs.tasks
        ))
    protocol = describe_quality_protocol(request, inputs)
    _validate_gguf_vocab([path for _, path, _, _ in candidates], snapshot, args.llama_cpp_root, inputs)
    weights = _checkpoint_files(snapshot)
    plan = to_dict({
        "protocol": protocol,
        "task_inputs": [to_dict(task.cache_identity) for task in inputs.tasks],
        "baseline": {
            "source": source, "revision": revision, "weights": weights, "config": hash_file(snapshot / "config.json"),
            "device": args.device, "torch": str(torch.__version__), "transformers": transformers.__version__,
        },
    })
    plan_path = output / "baseline-plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("The output directory contains a different baseline or input protocol")
    baseline_path = output / "baseline.json"
    if plan_path.is_file() and baseline_path.is_file():
        report = json.loads(baseline_path.read_text(encoding="utf-8"))
        if not report["passed"] or any(report["protocol"].get(key) != value for key, value in plan["protocol"].items()):
            raise ValueError("The saved baseline report does not match the full benchmark protocol")
        baseline = report["results"]["base"]
        _validate_result(baseline, inputs)
        print(f"Reusing verified baseline from {baseline_path}.", flush=True)
    else:
        atomic_write_json(plan_path, plan)
        report = execute_quality_evaluation(request, prepared=inputs, progress=_progress, evaluate_candidate=False)
        baseline = report["results"]["base"]
        _validate_result(baseline, inputs)
        atomic_write_json(baseline_path, report)

    models = [_model_summary(
        "Baseline", source, revision, "Transformers", sum(file["bytes"] for file in weights), baseline_path, baseline,
    )]
    comparisons = {}
    for label, path, candidate_source, candidate_revision in candidates:
        report_path = output / f"{label}.json"
        scoring_request = LlamaCppQualityRequest(
            gguf=path, output=report_path, llama_cpp_root=args.llama_cpp_root, runner=args.llama_cpp_runner,
            device=args.device, gpu_layers=0 if args.device == "cpu" else -1, parallel=args.parallel,
            gguf_shards=tuple(path.parent / name for name in _gguf_parts(path.name)[1:]),
        )
        materialization = None
        if label in args.materialize_nanoquant:
            if scoring_request.gguf_shards:
                raise ValueError("Materialization requires a single-file NanoQuant GGUF")
            dense_path = output / f"{label}-evaluation-fp32.gguf"
            materialization = materialize_gguf(path, dense_path, args.llama_cpp_root)
            scoring_request = replace(scoring_request, gguf=dense_path)
            materialization["validation"] = _validate_materialized_scores(
                scoring_request, path, request, inputs, materialization,
            )
        print(f"Scoring {label} from {scoring_request.gguf}.", flush=True)
        report = execute_llamacpp_quality_evaluation(
            scoring_request, request, inputs, baseline, protocol,
        )
        scored = report["results"]["gguf"]
        _validate_result(scored, inputs)
        size = report["gguf"]["bytes"] + sum(shard["bytes"] for shard in report["identity"].get("gguf_shards", ()))
        models.append(_model_summary(
            label, candidate_source, candidate_revision, "llama.cpp", size, report_path, scored,
        ))
        if materialization is not None:
            report["materialization"] = materialization
            atomic_write_json(report_path, report)
            models[-1].update({
                "backend": "llama.cpp FP32 reconstruction",
                "bytes": materialization["source_bytes"],
                "materialization": materialization,
            })
        comparisons[label] = report["comparison"]
    summary = {"protocol": protocol, "models": models, "comparisons_vs_baseline": comparisons}
    atomic_write_json(output / "comparison.json", summary)
    atomic_write_text(output / "comparison.md", _render_comparison(summary))
    print(f"Report: {output / 'comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
