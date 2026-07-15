"""Build a self-contained replay-to-full evaluation campaign from retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, cast

from nanoquant.application.evaluation import (
    EvaluatorRegistry,
    EvaluatorSpec,
    GatePolicy,
    GateRule,
)
from nanoquant.application.evaluation_campaign import (
    EvaluationCampaignResult,
    EvaluationTierPlan,
    LayerReplayEvidence,
    run_evaluation_campaign,
)
from nanoquant.application.report import render_run_report
from nanoquant.config.codec import canonical_json, to_dict
from nanoquant.config.schema import ObservabilityConfig
from nanoquant.domain.runs import RunStatus
from nanoquant.infrastructure.environment import capture_environment
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.run_session import open_run_session
from nanoquant.infrastructure.runs import (
    RunDirectory,
    initial_manifest_from_resolved,
    launcher_provenance,
    transition,
)


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"campaign input must contain one JSON object: {path}")
    return cast(dict[str, Any], payload)


def _copy_inputs(output: Path, sources: dict[str, Path]) -> dict[str, Path]:
    destination = output / "inputs"
    destination.mkdir(parents=True, exist_ok=False)
    copied: dict[str, Path] = {}
    for name, source in sorted(sources.items()):
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / f"{name}{source.suffix}"
        shutil.copyfile(source, target)
        copied[name] = target
    return copied


def _quick_result(inputs: dict[str, Path]) -> dict[str, object]:
    summary = _object(inputs["full-parity-summary"])
    validation = _object(inputs["validation"])
    trajectory = cast(dict[str, Any], summary["trajectory"])
    quantization = cast(dict[str, Any], summary["quantization"])
    artifact_validation = cast(dict[str, Any], summary["artifact_validation"])
    metrics = {
        "artifact_complete": float(bool(artifact_validation["complete"])),
        "rank_mismatch_count": float(quantization["rank_mismatch_count"]),
        "effective_bpw": float(quantization["effective_bpw"]),
        "maximum_block_loss_percent_delta": float(trajectory["maximum_absolute_percent_delta"]),
    }
    if int(validation["committed_layer_count"]) != 182:
        raise ValueError("quick evidence does not contain the complete Gemma layer inventory")
    return {
        "tier": "quick",
        "metrics": metrics,
        "committed_layers": validation["committed_layer_count"],
        "validated_artifacts": artifact_validation["artifacts_validated"],
        "source_sha256": {
            name: hash_file(inputs[name])
            for name in ("full-parity-summary", "validation")
        },
    }


def _standard_result(inputs: dict[str, Path]) -> dict[str, object]:
    summary = _object(inputs["full-parity-summary"])
    evaluation = _object(inputs["wikitext2"])
    parity = cast(dict[str, Any], summary["wikitext2_exact"])
    frozen = cast(dict[str, Any], cast(dict[str, Any], evaluation["results"])["frozen"])
    if frozen["perplexity"] != parity["tuned_perplexity"]:
        raise ValueError("standard evaluation perplexity differs from the compact parity summary")
    return {
        "tier": "standard",
        "metrics": {
            "wikitext2_percent_delta": float(parity["tuned_percent_delta"]),
            "wikitext2_scored_target_count": float(parity["scored_tokens"]),
        },
        "perplexity": frozen["perplexity"],
        "legacy_perplexity": parity["contemporary_legacy_tuned_perplexity"],
        "token_hash": parity["token_hash"],
        "source_sha256": hash_file(inputs["wikitext2"]),
    }


def _case_by_name(payload: dict[str, Any], name: str) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("runtime benchmark has no cases")
    for case in cases:
        if isinstance(case, dict) and case.get("name") == name:
            return cast(dict[str, Any], case)
    raise ValueError(f"runtime benchmark is missing case {name}")


def _full_result(inputs: dict[str, Path]) -> dict[str, object]:
    long_context = _object(inputs["long-context"])
    rewrite = _object(inputs["rewrite-runtime"])
    reference = _object(inputs["llamacpp-runtime"])
    evaluation = cast(dict[str, Any], long_context["evaluation"])
    long_case = cast(dict[str, Any], cast(list[Any], evaluation["cases"])[0])
    complete = _case_by_name(rewrite, "complete-generation")
    timing = cast(dict[str, Any], complete["timing"])
    throughput = cast(dict[str, Any], timing["throughput_per_second"])
    reference_throughput = cast(dict[str, Any], reference["decode_throughput_tokens_per_second"])
    candidate_rate = float(throughput["p50"])
    baseline_rate = float(reference_throughput["p50"])
    dispatch = cast(dict[str, Any], rewrite["dispatch"])
    fallbacks = sum(
        int(dispatch[name])
        for name in (
            "prefill_fallback_count",
            "decode_fallback_count",
            "cuda_graph_fallback_count",
        )
    ) + int(evaluation["total_unexpected_fallbacks"])
    metrics = {
        "long_context_exact": float(bool(evaluation["passed"] and long_case["exact_tokens"])),
        "unexpected_fallback_count": float(fallbacks),
        "long_context_peak_device_bytes": float(evaluation["peak_device_bytes"]),
        "runtime_reference_ratio": candidate_rate / baseline_rate,
    }
    return {
        "tier": "full",
        "metrics": metrics,
        "long_context_total_tokens": evaluation["maximum_total_tokens"],
        "candidate_tokens_per_second": candidate_rate,
        "reference_tokens_per_second": baseline_rate,
        "source_sha256": {
            name: hash_file(inputs[name])
            for name in ("long-context", "rewrite-runtime", "llamacpp-runtime")
        },
    }


def _layer_replay(inputs: dict[str, Path]) -> LayerReplayEvidence:
    comparison = _object(inputs["legacy-comparison"])
    blocks = comparison.get("blocks")
    if not isinstance(blocks, list) or len(blocks) < 4:
        raise ValueError("layer replay comparison does not contain four blocks")
    deltas = []
    for block in blocks[:4]:
        if not isinstance(block, dict):
            raise ValueError("layer replay block must be an object")
        baseline = cast(dict[str, Any], cast(dict[str, Any], block["baselines"])["contemporary-legacy"])
        deltas.append(abs(float(baseline["percent_delta"])))
    mean_delta = math.fsum(deltas) / len(deltas)
    return LayerReplayEvidence(
        "gemma-v28-first-four-blocks",
        "1",
        "sha256:" + hash_file(inputs["legacy-comparison"]),
        (("mean_absolute_block_loss_percent_delta", mean_delta),),
        mean_delta <= 1.0,
    )


def _metric_extractor(_specification: EvaluatorSpec, result: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(result, dict) or not isinstance(result.get("metrics"), dict):
        raise ValueError("campaign evaluator result has no metric object")
    metrics = cast(dict[str, object], result["metrics"])
    return tuple((name, float(value)) for name, value in sorted(metrics.items()))


def _campaign(inputs: dict[str, Path]) -> EvaluationCampaignResult:
    registry = EvaluatorRegistry()
    quick = EvaluatorSpec("gemma-artifact-and-trajectory", "1", "quick")
    standard = EvaluatorSpec("gemma-wikitext2-exact", "1", "standard")
    full = EvaluatorSpec("gemma-deployment-full", "1", "full")
    registry.register(quick, lambda _request: _quick_result(inputs))
    registry.register(standard, lambda _request: _standard_result(inputs))
    registry.register(full, lambda _request: _full_result(inputs))
    plans = (
        EvaluationTierPlan(
            "quick",
            ((quick.name, quick.version, None),),
            GatePolicy(
                "gemma-quick-promotion",
                "1",
                (
                    GateRule("artifact_complete", minimum=1),
                    GateRule("rank_mismatch_count", maximum=0),
                    GateRule("effective_bpw", maximum=1.01),
                    GateRule("maximum_block_loss_percent_delta", maximum=5),
                ),
            ),
        ),
        EvaluationTierPlan(
            "standard",
            ((standard.name, standard.version, None),),
            GatePolicy(
                "gemma-standard-promotion",
                "1",
                (
                    GateRule("wikitext2_percent_delta", maximum=2.27),
                    GateRule("wikitext2_scored_target_count", minimum=8128),
                ),
            ),
        ),
        EvaluationTierPlan(
            "full",
            ((full.name, full.version, None),),
            GatePolicy(
                "gemma-full-promotion",
                "1",
                (
                    GateRule("long_context_exact", minimum=1),
                    GateRule("unexpected_fallback_count", maximum=0),
                    GateRule("long_context_peak_device_bytes", maximum=12_878_086_144),
                    GateRule("runtime_reference_ratio", minimum=0.7),
                ),
            ),
        ),
    )
    return run_evaluation_campaign(
        candidate="gemma-pageable-v28",
        baseline="contemporary-legacy-018-and-compatible-llama.cpp",
        layer_replay=_layer_replay(inputs),
        registry=registry,
        plans=plans,
        extract_metrics=_metric_extractor,
    )


def _comparison_markdown(result: EvaluationCampaignResult, inputs: dict[str, Path]) -> str:
    lines = [
        "# Gemma v28 evaluation campaign",
        "",
        f"- Candidate: `{result.candidate}`",
        f"- Baseline: `{result.baseline}`",
        f"- Outcome: **{result.outcome}**",
        f"- Campaign identity: `{result.semantic_key}`",
        f"- Recommended next action: {result.recommended_next_action}",
        "",
        "## Promotion path",
        "",
        "| Stage | Decision | Metrics | Policy |",
        "| --- | --- | --- | --- |",
        (
            f"| layer-replay | {'promotion' if result.layer_replay.passed else 'rejection'} | "
            f"`{json.dumps(dict(result.layer_replay.metrics), sort_keys=True)}` | captured replay bound |"
        ),
    ]
    for tier in result.tiers:
        lines.append(
            f"| {tier.tier} | {tier.decision.outcome} | "
            f"`{json.dumps(dict(tier.metrics), sort_keys=True)}` | `{tier.decision.policy_key}` |"
        )
    lines.extend(("", "## Retained inputs", "", "| Input | Bytes | SHA-256 |", "| --- | ---: | --- |"))
    for _name, path in sorted(inputs.items()):
        lines.append(f"| `inputs/{path.name}` | {path.stat().st_size} | `{hash_file(path)}` |")
    lines.extend(
        (
            "",
            "The directory contains copied immutable inputs, canonical evaluator outputs, structured lifecycle and",
            "promotion events, the resolved campaign intent/environment, cost observations, and this comparison.",
        )
    )
    return "\n".join(lines) + "\n"


def _run(args: argparse.Namespace) -> EvaluationCampaignResult:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"campaign output directory is not empty: {output}")
    directory = RunDirectory(output.parent, output.name)
    inputs = _copy_inputs(
        directory.root,
        {
            "full-parity-summary": args.full_parity_summary,
            "validation": args.validation,
            "legacy-comparison": args.legacy_comparison,
            "wikitext2": args.wikitext2,
            "long-context": args.long_context,
            "rewrite-runtime": args.rewrite_runtime,
            "llamacpp-runtime": args.llamacpp_runtime,
        },
    )
    resolved = {
        "component": "evaluation-campaign",
        "intent": {
            "name": "Gemma v28 replay-to-full gate",
            "purpose": "Demonstrate ordered replay, quick, standard, and full evaluation promotion.",
            "hypothesis": "The parity candidate passes frozen quality, size, runtime, and memory gates.",
            "baseline_run": "contemporary-legacy-018-and-compatible-llama.cpp",
        },
        "candidate": "gemma-pageable-v28",
        "input_hashes": {name: hash_file(path) for name, path in sorted(inputs.items())},
    }
    config_hash = "sha256:" + hashlib.sha256(canonical_json(resolved).encode("utf-8")).hexdigest()
    manifest = initial_manifest_from_resolved(
        config_hash,
        resolved,
        launcher_provenance(Path(__file__), None),
        capture_environment(),
        run_id=output.name,
    )
    started = time.perf_counter()
    with open_run_session(
        directory.root,
        manifest=manifest,
        observability=ObservabilityConfig(record_resource_interval_seconds=60),
        console=False,
    ) as session:
        manifest = transition(session.manifest, RunStatus.RUNNING)
        directory.write_manifest(manifest)
        session.events.emit("run", "info", "run.started", config_hash=config_hash)
        result = _campaign(inputs)
        session.events.emit(
            "layer-replay",
            "info",
            "layer_replay.completed",
            passed=result.layer_replay.passed,
            semantic_key=result.layer_replay.semantic_key,
            **dict(result.layer_replay.metrics),
        )
        for tier in result.tiers:
            event_metrics = {
                name: int(value) if name.endswith("_bytes") else value
                for name, value in tier.metrics
            }
            session.events.emit(
                "evaluation",
                "info",
                f"evaluation.{tier.tier}.completed",
                decision=tier.decision.outcome,
                policy_key=tier.decision.policy_key,
                evaluator_result_keys=[item.result_key for item in tier.evaluators],
                **event_metrics,
            )
        result_path = directory.root / "results" / "campaign.json"
        result_path.parent.mkdir(exist_ok=True)
        atomic_write_json(result_path, to_dict(result))
        comparison = _comparison_markdown(result, inputs)
        reports = directory.root / "reports"
        (reports / "comparison.md").write_text(comparison, encoding="utf-8")
        session.events.emit(
            "comparison",
            "info",
            "comparison.completed",
            candidate=result.candidate,
            baseline=result.baseline,
            outcome=result.outcome,
        )
        elapsed = time.perf_counter() - started
        full_metrics = dict(result.tiers[-1].metrics) if result.tiers else {}
        conclusion = (
            "Candidate progressed from retained layer replay through quick, standard, and full evaluation."
            if result.passed
            else f"Evaluation campaign stopped at {result.completed_tier}: {result.outcome}."
        )
        session.events.emit(
            "run",
            "info" if result.passed else "warning",
            "run.completed",
            wall_seconds=elapsed,
            peak_device_bytes=int(full_metrics.get("long_context_peak_device_bytes", 0)),
            conclusion=conclusion,
            recommended_next_action=result.recommended_next_action,
        )
        artifact_ids = tuple(
            f"sha256:{hash_file(path)}" for path in (*inputs.values(), result_path)
        )
        manifest = transition(manifest, RunStatus.COMPLETED, artifacts=artifact_ids)
        directory.write_manifest(manifest)
        (reports / "summary.md").write_text(render_run_report(directory.root), encoding="utf-8")
    if not result.passed:
        raise RuntimeError(f"evaluation campaign did not reach full promotion: {result.outcome}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-parity-summary", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--legacy-comparison", type=Path, required=True)
    parser.add_argument("--wikitext2", type=Path, required=True)
    parser.add_argument("--long-context", type=Path, required=True)
    parser.add_argument("--rewrite-runtime", type=Path, required=True)
    parser.add_argument("--llamacpp-runtime", type=Path, required=True)
    args = parser.parse_args()
    result = _run(args)
    print(json.dumps(to_dict(result), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
