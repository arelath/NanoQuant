"""Prepare a tuning batch schedule continuation without recomputing completed blocks.

Dry-run by default. --apply imports only commit metadata under an explicit new
identity, preserves original evidence, and does not start compression. Blocks
before --start-block keep the batch sizes they were committed with; blocks at
and after it use the scheduled logical batch (measured once when the resumed
run starts unless --batch-size fixes it).
"""

import argparse
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import _paths  # noqa: F401

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.config.schema import RunConfig, TuningBatchScheduleConfig
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    _prepare_optimizer_continuation,
    _resident_config_hash,
    _resident_manifest,
)
from nanoquant.resident_workflow import ResidentExecutionOptions, ResolvedResidentInputs, resident_request_from_config


def request_for_batch_schedule_continuation(
    output: Path,
    start_block: int,
    maximum_batch_size: int,
    batch_size: int | None,
    memory_safety_factor: float,
) -> ResidentQuantizationRequest:
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    resolved = saved["resolved_config"]
    source_config = from_dict(RunConfig, resolved["canonical_run_config"])
    if source_config.block_tuning.batch_schedule != TuningBatchScheduleConfig():
        raise ValueError("the saved run already uses a tuning batch schedule")
    schedule = TuningBatchScheduleConfig(
        enabled=True,
        start_block=start_block,
        batch_size=batch_size,
        maximum_batch_size=maximum_batch_size,
        memory_safety_factor=memory_safety_factor,
        continuation_journal_sha256=hashlib.sha256((output / "state/journal.jsonl").read_bytes()).hexdigest(),
    )
    config = replace(source_config, block_tuning=replace(source_config.block_tuning, batch_schedule=schedule))
    raise_for_issues(validate(config, ValidationPhase.RESOLVED))
    if resolved["quality_token_ids"] is not None or any(resolved.get(name) is not None for name in (
        "precomputed_calibration", "precomputed_objectives", "precomputed_plan",
    )):
        raise ValueError("this preparation tool requires run-local calibration and no inline quality tokens")
    receipt = json.loads((output / "calibration-input.json").read_text(encoding="utf-8"))
    tokens = load_pinned_calibration(output, ArtifactRef("calibration-dataset-manifest", receipt["artifact_id"], 1))
    options = ResidentExecutionOptions(**{
        field.name: resolved[field.name] for field in fields(ResidentExecutionOptions) if field.name in resolved
    })
    inputs = ResolvedResidentInputs(
        Path(resolved["snapshot"]), output.resolve(), Path(resolved["registry_root"]), tokens.input_ids, None,
        pad_token_id=resolved.get("pad_token_id"),
    )
    source_request = resident_request_from_config(source_config, inputs, options)
    if _resident_manifest(source_request, "resident-quantization").config_hash != saved["config_hash"]:
        raise ValueError("saved request cannot be reproduced exactly; refusing continuation")
    records = [json.loads(line) for line in (output / "state/journal.jsonl").read_text().splitlines()]
    if not records or any(record["identity"]["config_hash"] != _resident_config_hash(source_request)
                          for record in records):
        raise ValueError("saved checkpoint algorithm identity differs from its reproduced request")
    blocks = sorted({record["block"] for record in records})
    if blocks and blocks[-1] >= start_block:
        raise ValueError(
            f"journal already contains block {blocks[-1]}; choose a start block after every committed record"
        )
    return resident_request_from_config(config, inputs, options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--start-block", type=int, required=True)
    parser.add_argument("--maximum-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, help="fix the scheduled batch instead of measuring it")
    parser.add_argument("--memory-safety-factor", type=float, default=1.5)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    request = request_for_batch_schedule_continuation(
        args.run_output, args.start_block, args.maximum_batch_size, args.batch_size, args.memory_safety_factor,
    )
    proposed = _resident_manifest(request, "resident-quantization")
    summary = {
        "applied": args.apply,
        "batch_schedule": to_dict(request.tuning_batch_schedule),
        "resident_config_hash": _resident_config_hash(request),
        "manifest_config_hash": proposed.config_hash,
    }
    if args.apply:
        with acquire_device_lease(request.device):
            _prepare_optimizer_continuation(request, proposed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
