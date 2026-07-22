"""Run the Experiment 016 quality protocol and compare an error-budget candidate."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.runtime_export import (
    export_frozen_run_logical,
    validate_frozen_run_logical,
)
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation
from nanoquant.runtime import (
    RuntimeModelMetadata,
    convert_logical_to_packed,
    open_logical_artifact,
    open_packed_artifact,
    validate_packed_conversion,
    validate_packed_reference_parity,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_QUALITY = (
    ROOT / "Results/016/016-compress-and-benchmark-gemma-3-270m-it-quality.json"
)
DEFAULT_BASELINE_SUMMARY = (
    ROOT / "Results/016/016-compress-and-benchmark-gemma-3-270m-it-summary.json"
)
PROTOCOL_FIELDS = (
    "wikitext_dataset",
    "wikitext_fingerprint",
    "wikitext_samples",
    "wikitext_sequence_length",
    "wikitext_batch_size",
    "wikitext_token_hash",
    "task_names",
    "task_limit",
    "task_batch_size",
    "tokenizer_hash",
    "base_execution",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, default=DEFAULT_BASELINE_QUALITY)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-global-tuning", action="store_true")
    parser.add_argument("--require-improvement", action="store_true")
    return parser


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite")
    return result


def _model_metrics(
    payload: Mapping[str, Any],
    label: str,
) -> tuple[float, float, dict[str, dict[str, object]]]:
    results = cast(Mapping[str, Any], payload["results"])
    model = cast(Mapping[str, Any], results[label])
    wikitext = cast(Mapping[str, object], model["wikitext"])
    tasks: dict[str, dict[str, object]] = {}
    for item in cast(list[Mapping[str, Any]], model["tasks"]):
        result = cast(dict[str, object], item["result"])
        tasks[str(result["task_name"])] = result
    return (
        _number(wikitext["mean_negative_log_likelihood"], "mean NLL"),
        _number(wikitext["perplexity"], "perplexity"),
        tasks,
    )


def _task_comparison(
    baseline: Mapping[str, dict[str, object]],
    candidate: Mapping[str, dict[str, object]],
) -> tuple[list[dict[str, object]], float]:
    if baseline.keys() != candidate.keys():
        raise ValueError("candidate task set does not match the retained baseline")
    rows = []
    for name in baseline:
        base = _number(baseline[name]["primary_value"], f"{name} baseline metric")
        value = _number(candidate[name]["primary_value"], f"{name} candidate metric")
        base_metric = str(baseline[name]["primary_metric"])
        candidate_metric = str(candidate[name]["primary_metric"])
        if base_metric != candidate_metric:
            raise ValueError(f"candidate metric for {name} changed")
        rows.append(
            {
                "task_name": name,
                "metric": base_metric,
                "baseline": base,
                "candidate": value,
                "delta": value - base,
            }
        )
    return rows, sum(_number(row["delta"], "task delta") for row in rows) / len(rows)


def _quality_claim_passed(
    *,
    protocol_identity_matched: bool,
    base_results_reproduced: bool,
    packed_evaluation_identity_matched: bool,
    global_tuning_identity_matched: bool,
    same_or_lower_budget: bool,
    nll_improved: bool,
    exact_packed_conversion: bool,
    exact_packed_reference_parity: bool,
) -> bool:
    return all(
        (
            protocol_identity_matched,
            base_results_reproduced,
            packed_evaluation_identity_matched,
            global_tuning_identity_matched,
            same_or_lower_budget,
            nll_improved,
            exact_packed_conversion,
            exact_packed_reference_parity,
        )
    )


def _render(payload: Mapping[str, Any]) -> str:
    budget = cast(Mapping[str, object], payload["budget"])
    quality = cast(Mapping[str, object], payload["quality"])
    packed = cast(Mapping[str, object], payload["packed_runtime"])
    tasks = cast(list[Mapping[str, object]], quality["tasks"])
    lines = [
        "# Gemma-3-270M error-budget quality comparison",
        "",
        f"- Protocol identity matched: `{payload['protocol_identity_matched']}`",
        f"- BF16 base results reproduced: `{payload['base_results_reproduced']}`",
        f"- Exact packed artifact evaluated: `{payload['packed_evaluation_identity_matched']}`",
        f"- Exact global tuning evaluated: `{payload['global_tuning_identity_matched']}`",
        f"- Same-or-lower effective BPW: `{budget['same_or_lower']}`",
        f"- Packed/reference parity passed: `{packed['exact_reference_parity']}`",
        f"- WikiText NLL improved: `{quality['nll_improved']}`",
        f"- Overall claim passed: `{payload['quality_improved_at_same_budget']}`",
        "",
        "## Budget",
        "",
        "| Run | Effective BPW |",
        "| --- | ---: |",
        f"| Experiment 016 | {_number(budget['baseline_effective_bpw'], 'baseline BPW'):.9f} |",
        f"| Candidate | {_number(budget['candidate_effective_bpw'], 'candidate BPW'):.9f} |",
        "",
        "## WikiText-2",
        "",
        "| Metric | Experiment 016 | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Mean NLL ↓ | {_number(quality['baseline_nll'], 'baseline NLL'):.9f} | "
            f"{_number(quality['candidate_nll'], 'candidate NLL'):.9f} | "
            f"{_number(quality['nll_delta'], 'NLL delta'):+.9f} |"
        ),
        (
            f"| Perplexity ↓ | "
            f"{_number(quality['baseline_perplexity'], 'baseline perplexity'):.6f} | "
            f"{_number(quality['candidate_perplexity'], 'candidate perplexity'):.6f} | "
            f"{_number(quality['perplexity_delta'], 'perplexity delta'):+.6f} |"
        ),
        "",
        "## Pinned tasks",
        "",
        "| Task | Metric | Experiment 016 | Candidate | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in tasks:
        lines.append(
            f"| {row['task_name']} | {row['metric']} ↑ | "
            f"{_number(row['baseline'], 'task baseline'):.4f} | "
            f"{_number(row['candidate'], 'task candidate'):.4f} | "
            f"{_number(row['delta'], 'task delta'):+.4f} |"
        )
    lines.extend(
        (
            "",
            f"Task macro delta: {_number(quality['task_macro_delta'], 'task macro delta'):+.6f}.",
            "",
            "The quality claim is based on the exact retained token/task identities and requires lower "
            "WikiText NLL at no greater effective BPW. Task results are reported independently because "
            "the fixed 200-example subsets are noisier than token NLL.",
            "",
        )
    )
    return "\n".join(lines)


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    run = args.candidate_run.resolve()
    summary_path = run / "candidate-summary.json"
    candidate_summary = _read_object(summary_path)
    if candidate_summary.get("status") != "completed":
        raise ValueError(f"candidate is not complete: {summary_path}")
    expected_global_tuning: Mapping[str, object] | None = None
    if args.use_global_tuning:
        distillation_summary = _read_object(run / "distillation-summary.json")
        if distillation_summary.get("status") != "completed":
            raise ValueError("candidate global distillation is not complete")
        expected_global_tuning = cast(Mapping[str, object], distillation_summary["artifact"])
    manifest = _read_object(run / "manifest.json")
    resolved = cast(Mapping[str, Any], manifest["resolved_config"])
    baseline_quality = _read_object(args.baseline_quality.resolve())
    baseline_summary = _read_object(args.baseline_summary.resolve())
    baseline_model = cast(Mapping[str, object], baseline_quality["model"])
    source = str(resolved["source"])
    revision = str(resolved["revision"])
    if source != str(baseline_model["source"]) or revision != str(baseline_model["revision"]):
        raise ValueError("candidate model identity does not match Experiment 016")
    protocol = cast(Mapping[str, Any], baseline_quality["protocol"])
    snapshot = Path(str(resolved["snapshot"])).resolve()
    model_source = SafetensorsModelSource(
        snapshot,
        source=source,
        revision=revision,
        verify_hashes=False,
    )
    checkpoint = model_source.inventory()
    model_inventory = adapter_for_config(checkpoint.config).model_inventory(model_source).model
    runtime_metadata = RuntimeModelMetadata(
        source,
        revision,
        "gemma3",
        model_inventory.config_hash,
        checkpoint.tokenizer_hash,
    )
    runtime_root = run / "runtime" / ("tuned" if args.use_global_tuning else "static")
    logical_output = runtime_root / "logical"
    packed_output = runtime_root / "packed"
    block_count = int(candidate_summary["blocks"])
    if logical_output.exists():
        open_logical_artifact(logical_output, verify_hashes=True)
    else:
        export_frozen_run_logical(
            run,
            logical_output,
            runtime_metadata,
            block_count,
            use_global_tuning=args.use_global_tuning,
            fresh_validation=True,
        )
    logical_validation = validate_frozen_run_logical(
        run,
        logical_output,
        block_count,
        use_global_tuning=args.use_global_tuning,
        fresh_validation=True,
    )
    if packed_output.exists():
        open_packed_artifact(packed_output, verify_hashes=True)
    else:
        convert_logical_to_packed(logical_output, packed_output)
    packed_conversion = validate_packed_conversion(logical_output, packed_output)
    packed_parity = validate_packed_reference_parity(logical_output, packed_output)
    request = QualityEvaluationRequest(
        snapshot=snapshot,
        source=source,
        revision=revision,
        run_output=run,
        device=args.device,
        backend="factorized",
        use_global_tuning=args.use_global_tuning,
        wikitext_samples=int(protocol["wikitext_samples"]),
        wikitext_sequence_length=int(protocol["wikitext_sequence_length"]),
        wikitext_batch_size=int(protocol["wikitext_batch_size"]),
        task_names=tuple(str(item) for item in protocol["task_names"]),
        task_limit=int(protocol["task_limit"]),
        task_batch_size=int(protocol["task_batch_size"]),
        local_files_only=args.local_files_only,
        maximum_wddm_shared_bytes=int(resolved["maximum_wddm_shared_bytes"]),
        packed_artifact=packed_output,
    )
    quality_variant = "tuned" if args.use_global_tuning else "static"
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else run / f"quality-comparison-{quality_variant}.json"
    )
    output_markdown = (
        args.output_markdown.resolve()
        if args.output_markdown is not None
        else run / f"quality-comparison-{quality_variant}.md"
    )
    progress_path = run / f"quality-progress-{quality_variant}.jsonl"

    def progress(event: str, fields: Mapping[str, object]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        print(line, flush=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    evaluation = execute_quality_evaluation(request, progress=progress)
    evaluation_candidate = cast(Mapping[str, object], evaluation["candidate"])
    packed_descriptor_hash = hash_file(packed_output / "nanoquant-packed-model.json")
    packed_evaluation_identity_matched = (
        evaluation_candidate.get("packed_artifact") == str(packed_output.resolve())
        and evaluation_candidate.get("packed_descriptor_sha256") == packed_descriptor_hash
    )
    global_tuning_identity_matched = (
        evaluation_candidate.get("global_tuning") == expected_global_tuning
        if args.use_global_tuning
        else evaluation_candidate.get("global_tuning") is None
    )
    candidate_protocol = cast(Mapping[str, object], evaluation["protocol"])
    protocol_match = all(
        json.dumps(candidate_protocol[field], sort_keys=True)
        == json.dumps(protocol[field], sort_keys=True)
        for field in PROTOCOL_FIELDS
    )
    baseline_base_nll, baseline_base_ppl, baseline_base_tasks = _model_metrics(
        baseline_quality,
        "base",
    )
    repeated_base_nll, repeated_base_ppl, repeated_base_tasks = _model_metrics(
        evaluation,
        "base",
    )
    base_task_rows, base_task_macro_delta = _task_comparison(
        baseline_base_tasks,
        repeated_base_tasks,
    )
    base_results_reproduced = (
        abs(repeated_base_nll - baseline_base_nll) <= 1e-12
        and abs(repeated_base_ppl - baseline_base_ppl) <= 1e-9
        and all(abs(_number(row["delta"], "base task delta")) <= 1e-12 for row in base_task_rows)
    )
    baseline_nll, baseline_ppl, baseline_tasks = _model_metrics(baseline_quality, "frozen")
    candidate_nll, candidate_ppl, candidate_tasks = _model_metrics(evaluation, "frozen")
    task_rows, task_macro_delta = _task_comparison(baseline_tasks, candidate_tasks)
    baseline_compression = cast(
        Mapping[str, object],
        cast(Mapping[str, Any], baseline_summary["compression"]),
    )
    baseline_bpw = _number(baseline_compression["effective_bpw"], "baseline effective BPW")
    candidate_bpw = _number(candidate_summary["effective_bpw"], "candidate effective BPW")
    same_or_lower = candidate_bpw <= baseline_bpw + 1e-12
    nll_improved = candidate_nll < baseline_nll
    quality_improved = _quality_claim_passed(
        protocol_identity_matched=protocol_match,
        base_results_reproduced=base_results_reproduced,
        packed_evaluation_identity_matched=packed_evaluation_identity_matched,
        global_tuning_identity_matched=global_tuning_identity_matched,
        same_or_lower_budget=same_or_lower,
        nll_improved=nll_improved,
        exact_packed_conversion=packed_conversion.exact,
        exact_packed_reference_parity=packed_parity.maximum_absolute_error == 0.0,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "candidate_run": str(run),
        "use_global_tuning": args.use_global_tuning,
        "baseline_quality": str(args.baseline_quality.resolve()),
        "baseline_summary": str(args.baseline_summary.resolve()),
        "protocol_identity_matched": protocol_match,
        "base_results_reproduced": base_results_reproduced,
        "packed_evaluation_identity_matched": packed_evaluation_identity_matched,
        "global_tuning_identity_matched": global_tuning_identity_matched,
        "base_reproduction": {
            "baseline_nll": baseline_base_nll,
            "repeated_nll": repeated_base_nll,
            "nll_delta": repeated_base_nll - baseline_base_nll,
            "baseline_perplexity": baseline_base_ppl,
            "repeated_perplexity": repeated_base_ppl,
            "perplexity_delta": repeated_base_ppl - baseline_base_ppl,
            "tasks": base_task_rows,
            "task_macro_delta": base_task_macro_delta,
        },
        "budget": {
            "baseline_effective_bpw": baseline_bpw,
            "candidate_effective_bpw": candidate_bpw,
            "delta": candidate_bpw - baseline_bpw,
            "same_or_lower": same_or_lower,
        },
        "packed_runtime": {
            "logical_output": str(logical_output),
            "packed_output": str(packed_output),
            "logical_validation": {
                "blocks": logical_validation.block_count,
                "layers": logical_validation.layer_count,
                "tensor_count": logical_validation.tensor_count,
                "tensor_bytes": logical_validation.tensor_bytes,
                "exact": logical_validation.exact,
            },
            "conversion_layers": packed_conversion.layer_count,
            "logical_tensor_count": packed_conversion.logical_tensor_count,
            "packed_tensor_count": packed_conversion.packed_tensor_count,
            "exact_conversion": packed_conversion.exact,
            "parity_layers": packed_parity.layer_count,
            "parity_output_elements": packed_parity.output_elements,
            "maximum_absolute_error": packed_parity.maximum_absolute_error,
            "maximum_error_layer": packed_parity.maximum_error_layer,
            "exact_reference_parity": packed_parity.maximum_absolute_error == 0.0,
        },
        "quality": {
            "baseline_nll": baseline_nll,
            "candidate_nll": candidate_nll,
            "nll_delta": candidate_nll - baseline_nll,
            "nll_improved": nll_improved,
            "baseline_perplexity": baseline_ppl,
            "candidate_perplexity": candidate_ppl,
            "perplexity_delta": candidate_ppl - baseline_ppl,
            "tasks": task_rows,
            "task_macro_delta": task_macro_delta,
        },
        "quality_improved_at_same_budget": quality_improved,
        "candidate_evaluation": evaluation,
    }
    atomic_write_json(output_json, payload)
    atomic_write_text(output_markdown, _render(payload))
    print(json.dumps({key: value for key, value in payload.items() if key != "candidate_evaluation"}, indent=2))
    if args.require_improvement and not quality_improved:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
