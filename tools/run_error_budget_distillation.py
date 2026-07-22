"""Run or resume the scheduled scale/bias distillation phase for a completed candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import _paths  # noqa: F401
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.application.distillation import DistillationMetrics
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.domain.models import ArtifactRef, GlobalTuningResult
from nanoquant.global_distillation import run_global_topk_distillation
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.hf_calibration_dataset import materialize_pinned_calibration
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.runtime_export import load_frozen_run_auxiliary
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    distillation_request_from_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interrupt-after-epoch-commits", type=int)
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def _completed_payload(
    config: RunConfig,
    reference: ArtifactRef,
    result: GlobalTuningResult,
    metrics: DistillationMetrics,
    *,
    recovered_existing: bool,
) -> dict[str, object]:
    return {
        "status": "completed",
        "recovered_existing": recovered_existing,
        "config": to_dict(config),
        "artifact": to_dict(reference),
        "metrics": to_dict(metrics),
        "epoch_losses": result.epoch_losses,
        "selected_parameter_count": result.selected_parameter_count,
        "teacher_cache_bytes": result.teacher_cache_bytes,
    }


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    run = args.candidate_run.resolve()
    candidate_summary = json.loads((run / "candidate-summary.json").read_text(encoding="utf-8"))
    if not isinstance(candidate_summary, dict) or candidate_summary.get("status") != "completed":
        raise ValueError("global distillation requires a completed error-budget candidate")
    config_payload = candidate_summary.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("candidate summary has no canonical config")
    config = from_dict(RunConfig, config_payload, path="candidate.config")
    config = replace(
        config,
        distillation=replace(config.distillation, enabled=True),
        runtime=replace(config.runtime, compute_device=args.device),
    )
    summary_path = run / "distillation-summary.json"
    existing = active_global_tuning(run)
    if existing is not None and not args.replace_existing:
        validated = load_frozen_run_auxiliary(
            run,
            int(candidate_summary["blocks"]),
            use_global_tuning=True,
            fresh_validation=True,
        )
        if validated.global_tuning != existing:
            raise ValueError("validated global tuning differs from the active pointer")
        recovered = load_global_tuning(existing, LocalArtifactStore(run / "artifacts")).result
        metrics = DistillationMetrics(
            recovered.epoch_losses,
            recovered.steps_completed,
            recovered.selected_parameter_count,
            recovered.teacher_cache_bytes,
        )
        payload = _completed_payload(
            config,
            existing,
            recovered,
            metrics,
            recovered_existing=True,
        )
        atomic_write_json(summary_path, payload)
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0
    calibration = materialize_pinned_calibration(
        args.calibration_source,
        run,
        sample_count=config.calibration.sample_count,
        sequence_length=config.model.sequence_length,
        seed=config.reproducibility.seed,
        preparation_id=None,
        tokenizer_identity=f"{config.model.source}@{config.model.revision}",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    inputs = ResolvedResidentInputs(
        snapshot=args.snapshot.resolve(),
        output=run,
        registry_root=run.parent,
        token_ids=calibration.input_ids,
        quality_token_ids=calibration.input_ids[:1, :8],
        launcher_path=Path(__file__),
        pad_token_id=tokenizer.pad_token_id,
    )
    options = ResidentExecutionOptions(
        interrupt_after_distillation_epoch_commits=args.interrupt_after_epoch_commits,
        replace_existing_global_tuning=args.replace_existing,
        maximum_wddm_shared_bytes=int(0.75 * 2**30),
    )
    try:
        result = run_global_topk_distillation(
            distillation_request_from_config(config, inputs, options)
        )
    except InterruptedError as error:
        atomic_write_json(
            summary_path,
            {"status": "interrupted", "reason": str(error), "config": to_dict(config)},
        )
        return 2
    payload = _completed_payload(
        config,
        result.reference,
        result.result,
        result.metrics,
        recovered_existing=False,
    )
    atomic_write_json(summary_path, payload)
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
