"""Extract measured allocation responses into a reusable dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from nanoquant.application.kl_budget import load_kl_budget_profile, measured_unit_kl_anchors, validate_kl_budget_profile
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.config.schema import RankResponseSource, RunConfig
from nanoquant.domain.models import ArtifactRef, LayerInventory, QuantizationPlan, SharedInputGroupPlan
from nanoquant.infrastructure.allocation_profile_dataset import (
    AllocationProfileDataset,
    load_allocation_profile_dataset,
    publish_allocation_profile_dataset,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.runs import RunLease
from nanoquant.resident_quantization import _rank_probe_units, _RankProbeEvidence


def _read_artifact(store: LocalArtifactStore, reference: ArtifactRef, filename: str) -> dict[str, Any]:
    descriptor = store.validate(reference.artifact_id)
    if (
        descriptor.artifact_type != str(reference.artifact_type)
        or descriptor.schema_version != reference.schema_version
    ):
        raise ValueError("allocation source artifact type or schema differs")
    return dict(json.loads((store.path_for(reference.artifact_id) / filename).read_text(encoding="utf-8")))


def build_allocation_profile_dataset(
    run_output: Path,
    output: Path,
    *,
    kl_profile_path: Path | None = None,
) -> AllocationProfileDataset:
    """Copy scalar evidence after validating the source profile and all unit probes."""
    store = LocalArtifactStore(run_output / "artifacts")
    state = json.loads((run_output / "state/preprocessing.json").read_text(encoding="utf-8"))
    plan_ref = from_dict(ArtifactRef, state["plan"])
    plan = from_dict(QuantizationPlan, _read_artifact(store, plan_ref, "plan.json"))
    if plan.reconstruction_profile is None:
        raise ValueError("source run has no completed reconstruction allocation profile")
    profile_payload = _read_artifact(store, plan.reconstruction_profile, "reconstruction-rank-profile.json")
    if profile_payload.get("allocation_dataset") is not None:
        raise ValueError("source run already uses an approximate dataset; reuse that original dataset")
    probe_plan_ref = from_dict(ArtifactRef, profile_payload["probe_plan"])
    probe_plan = _read_artifact(store, probe_plan_ref, "rank-probe-plan.json")
    baseline_plan = from_dict(QuantizationPlan, probe_plan["baseline_plan"])
    units = dict(_rank_probe_units(baseline_plan))
    if baseline_plan.model != plan.model:
        raise ValueError("source allocation and probe plans have different models")
    measured = {}
    for value in profile_payload["unit_results"]:
        reference = from_dict(ArtifactRef, value)
        evidence = from_dict(_RankProbeEvidence, _read_artifact(store, reference, "rank-probe-result.json"))
        if evidence.unit_id in measured or evidence.probe_plan != probe_plan_ref:
            raise ValueError("source allocation probes are duplicated or belong to another plan")
        unit = units[evidence.unit_id]
        members = unit.members if isinstance(unit, SharedInputGroupPlan) else (unit,)
        weights = tuple(
            member.weight if isinstance(member, LayerInventory) else member.source_weight for member in members
        )
        if (
            evidence.baseline_rank != unit.rank
            or evidence.members != tuple(member.layer for member in members)
            or evidence.source_weight_hashes != tuple(weight.content_hash for weight in weights)
        ):
            raise ValueError("source allocation probe does not match its physical unit")
        encoded = to_dict(evidence)
        encoded.pop("probe_plan")
        measured[evidence.unit_id] = {
            "evidence": encoded,
            "member_shapes": [list(weight.spec.shape) for weight in weights],
            "source_probe_id": reference.artifact_id,
        }
    if set(measured) != set(units):
        raise ValueError("source allocation profile does not cover every unit")
    manifest_path = run_output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    recipe = manifest.get("resolved_config", {}).get("canonical_run_config", {})
    if kl_profile_path is None:
        configured = recipe.get("allocation", {}).get("kl_profile_artifact")
        if configured:
            kl_profile_path = Path(configured)
    kl_profile = None
    if kl_profile_path is not None:
        if kl_profile_path.is_dir():
            kl_profile_path /= "kl-budget-profile.json"
        kl_profile = load_kl_budget_profile(kl_profile_path)
        validate_kl_budget_profile(kl_profile, model_source=plan.model.source, model_revision=plan.model.revision)
        measured_unit_kl_anchors(kl_profile, tuple(units))
    payload = {
        "schema_version": 1,
        "model": to_dict(plan.model),
        "source_run": str(run_output.resolve()),
        "source_plan_id": plan_ref.artifact_id,
        "source_profile_id": plan.reconstruction_profile.artifact_id,
        "source_probe_plan_id": probe_plan_ref.artifact_id,
        "probe_protocol": {key: value for key, value in probe_plan.items() if key != "baseline_plan"},
        "source_recipe": recipe,
        "units": [measured[unit_id] for unit_id in units],
        "kl_profile_key": None if kl_profile is None else kl_profile.profile_key,
    }
    output.mkdir(parents=True, exist_ok=True)
    with RunLease(output / ".active-lease.json"):
        return publish_allocation_profile_dataset(output, payload, kl_profile)


def config_with_allocation_dataset(config: RunConfig, root: Path) -> RunConfig:
    """Select donor estimates explicitly while preserving the current fitting recipe."""
    dataset = load_allocation_profile_dataset(root)
    if dataset.payload["model"]["source"] != config.model.source or dataset.payload["model"]["revision"] != str(
        config.model.revision
    ):
        raise ValueError("allocation profile dataset belongs to a different model revision")
    if dataset.kl_profile is None:
        raise ValueError("D2 allocation dataset requires KL sensitivities; build with --kl-profile")
    reconstruction = replace(
        config.allocation.reconstruction,
        enabled=True,
        response_source=RankResponseSource.DATASET,
        response_curves=(),
        allocation_dataset=str(root.resolve()),
        allocation_dataset_key=dataset.reference.artifact_id,
        response_profile_provenance=f"approximate allocation dataset {dataset.reference.artifact_id}",
    )
    return replace(
        config,
        allocation=replace(
            config.allocation,
            reconstruction=reconstruction,
            kl_profile_artifact=str(dataset.directory.resolve()),
            kl_profile_key=dataset.kl_profile.profile_key,
        ),
    )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True, help="run with a completed rank-probe profile")
    parser.add_argument("--output", type=Path, help="defaults to Datasets/<source-run-name>-allocation")
    parser.add_argument("--kl-profile", type=Path, help="optional override for the source recipe's KL profile")
    args = parser.parse_args(arguments)
    output = args.output or Path(__file__).resolve().parents[2] / "Datasets" / f"{args.run_output.name}-allocation"
    dataset = build_allocation_profile_dataset(args.run_output, output, kl_profile_path=args.kl_profile)
    print(f"Allocation dataset saved to {output.resolve()}")
    print(f"Units {len(dataset.payload['units'])}, KL sensitivities {dataset.kl_profile is not None}")
    print(f"Dataset key {dataset.reference.artifact_id}")
    return 0
