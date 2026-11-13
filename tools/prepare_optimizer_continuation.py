"""Prepare a fused-optimizer continuation without recomputing completed weights.

Dry-run by default. --apply imports only commit metadata under an explicit new
identity, preserves original evidence, and does not start compression.
"""

import argparse
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import _paths  # noqa: F401

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.config.schema import FactorizedOptimizerConfig, RunConfig
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.resident_quantization import (
    _prepare_optimizer_continuation,
    _resident_config_hash,
    _resident_manifest,
)
from nanoquant.resident_workflow import ResidentExecutionOptions, ResolvedResidentInputs, resident_request_from_config


def request_for_continuation(output, start_block, layer_patterns):
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    resolved = saved["resolved_config"]
    config = from_dict(RunConfig, resolved["canonical_run_config"])
    source_config = config
    original_policy = config.block_tuning.factorized.optimizer
    if original_policy == FactorizedOptimizerConfig():
        policy = FactorizedOptimizerConfig(
            "fused_adamw_scales_fp32", start_block, tuple(layer_patterns),
            hashlib.sha256((output / "state/journal.jsonl").read_bytes()).hexdigest(),
        )
    else:
        policy = original_policy
        if policy.start_block != start_block or policy.layer_patterns != tuple(layer_patterns):
            raise ValueError("existing continuation policy differs")
    config = replace(config, block_tuning=replace(
        config.block_tuning, factorized=replace(config.block_tuning.factorized, optimizer=policy),
    ))
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
    return resident_request_from_config(config, inputs, options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--start-block", type=int, required=True)
    parser.add_argument("--layers", nargs="+", required=True, help="Projection paths that passed quality validation")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    request = request_for_continuation(args.run_output, args.start_block, args.layers)
    proposed = _resident_manifest(request, "resident-quantization")
    summary = {"applied": args.apply, "optimizer": to_dict(request.factorized_tuning_optimizer),
               "resident_config_hash": _resident_config_hash(request), "manifest_config_hash": proposed.config_hash}
    if args.apply:
        with acquire_device_lease(request.device):
            _prepare_optimizer_continuation(request, proposed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
