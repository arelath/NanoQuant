"""Validate and freeze the pre-selection identity of a fresh Experiment 048 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import _paths  # noqa: F401
from probe_distillation_checkpoint_tail_mass import CheckpointCandidate, discover_checkpoints
from select_c4_capability_correction_checkpoint import RULE
from validate_evaluation_slice_registry import validate_registry

from nanoquant.config.codec import config_hash, from_dict, semantic_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.domain.models import ArtifactRef
from nanoquant.domain.runs import RunManifest, RunStatus
from nanoquant.global_distillation import distillation_protocol_hash
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.resident_workflow import primary_distillation_config_from_run_config

EXPERIMENT_NUMBER = 48
MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
PRIMARY_PROTOCOL_HASH = (
    "sha256:0ed7993a02eb980403ebeb97ff2d2cbf738242e64e6a7d07ad9f2900ef611936"
)
PRIMARY_STEPS = 256
CORRECTION_NAMESPACE = "global-distillation-mass-floor"
CORRECTION_STEPS = {1: 32, 2: 64, 3: 96, 4: 128}
SELECTOR_TOLERANCE = 0.01
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--strict-validation", type=Path, required=True)
    parser.add_argument("--slice-registry", type=Path, required=True)
    parser.add_argument("--selection-slice-id", required=True)
    parser.add_argument("--confirmation-slice-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    return "sha256:" + hash_file(path)


def _validate_experiment048_config(config: RunConfig) -> str:
    distillation = config.distillation
    correction = distillation.mass_floor_correction
    primary_hash = distillation_protocol_hash(
        primary_distillation_config_from_run_config(config)
    )
    if (
        config.intent.experiment_number != EXPERIMENT_NUMBER
        or str(config.model.revision) != MODEL_REVISION
        or not distillation.enabled
        or distillation.loss.value != "top_k_tail"
        or distillation.epochs != 8
        or distillation.maximum_batches_per_epoch != 32
        or distillation.tail_mass_weight != 0.5
        or primary_hash != PRIMARY_PROTOCOL_HASH
        or not correction.enabled
        or correction.expected_initializer_protocol_hash != primary_hash
        or correction.expected_initializer_steps != PRIMARY_STEPS
        or correction.epochs != 4
        or correction.learning_rate != 1e-5
        or correction.maximum_batches_per_epoch != 32
        or correction.scheduler_total_steps != 128
        or correction.minimum_teacher_mass_ratio != 0.8
        or correction.mass_loss_weight != 2.0
        or distillation.final_norm_calibration.enabled
        or distillation.foldable_mlp_multipliers.enabled
    ):
        raise ValueError("Experiment 048 run config differs from the frozen campaign")
    return primary_hash


def _reserved_c4_slices(
    payload: object,
    *,
    selection_id: str,
    confirmation_id: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    audit = validate_registry(payload)
    if not isinstance(payload, dict) or selection_id == confirmation_id:
        raise ValueError("Experiment 048 C4 slice identities are invalid")
    entries = payload.get("slices")
    if not isinstance(entries, list):
        raise ValueError("Experiment 048 C4 slice registry is invalid")

    def selected(identity: str, consumer: str) -> dict[str, object]:
        matches = [entry for entry in entries if entry.get("id") == identity]
        if len(matches) != 1 or not isinstance(matches[0], dict):
            raise ValueError(f"Experiment 048 slice {identity} is missing or ambiguous")
        entry = cast(dict[str, object], matches[0])
        token_hash = entry.get("token_hash")
        if (
            entry.get("dataset") != "allenai/c4"
            or entry.get("split") != "validation"
            or entry.get("status") != "reserved"
            or entry.get("consumer") != consumer
            or entry.get("samples") != 48
            or entry.get("sequence_length") != 512
            or not isinstance(token_hash, str)
            or not token_hash.startswith("sha256:")
            or len(token_hash) != 71
        ):
            raise ValueError(f"Experiment 048 slice {identity} differs from its frozen role")
        return entry

    selection = selected(selection_id, "experiment-048-correction-selection")
    confirmation = selected(confirmation_id, "experiment-048-final-confirmation")
    return selection, confirmation, audit


def _checkpoint_receipts(
    candidates: tuple[CheckpointCandidate, ...],
    *,
    primary: ArtifactRef,
    source_blocks: tuple[ArtifactRef, ...],
    correction_protocol_hash: str,
) -> tuple[dict[str, object], ...]:
    by_epoch = {candidate.epoch: candidate for candidate in candidates}
    if set(by_epoch) != set(CORRECTION_STEPS) or len(by_epoch) != len(candidates):
        raise ValueError("Experiment 048 correction checkpoint inventory differs")
    receipts = []
    for epoch, expected_steps in CORRECTION_STEPS.items():
        candidate = by_epoch[epoch]
        if (
            candidate.steps != expected_steps
            or candidate.identity.protocol_hash != correction_protocol_hash
            or candidate.identity.initializer_global_tuning != primary
            or candidate.identity.source_blocks != source_blocks
        ):
            raise ValueError(
                f"Experiment 048 correction checkpoint epoch {epoch} differs"
            )
        receipts.append(
            {
                "epoch": epoch,
                "steps": candidate.steps,
                "reference": to_dict(candidate.reference),
                "identity": to_dict(candidate.identity),
            }
        )
    return tuple(receipts)


def run(args: argparse.Namespace) -> int:
    run_output = args.run_output.resolve()
    launcher = args.launcher.resolve()
    validation_path = args.strict_validation.resolve()
    registry_path = args.slice_registry.resolve()
    manifest_path = run_output / "manifest.json"
    manifest = from_dict(
        RunManifest,
        json.loads(manifest_path.read_text(encoding="utf-8")),
        path="experiment048.manifest",
    )
    canonical = manifest.resolved_config.get("canonical_run_config")
    if manifest.status is not RunStatus.COMPLETED or not isinstance(canonical, dict):
        raise ValueError("Experiment 048 resident workflow is not complete")
    config = from_dict(RunConfig, canonical, path="experiment048.config")
    primary_hash = _validate_experiment048_config(config)
    if (
        manifest.launcher.content_hash != _sha256(launcher)
        or manifest.launcher.experiment_number != EXPERIMENT_NUMBER
    ):
        raise ValueError("Experiment 048 launcher or config identity differs")

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("complete") is not True
        or Path(str(validation.get("run_output"))).resolve() != run_output
        or validation.get("block_records") != 26
        or validation.get("committed_layer_count") != 130
        or not isinstance(validation.get("identity"), dict)
    ):
        raise ValueError("Experiment 048 strict resident validation is incomplete")

    artifacts = LocalArtifactStore(
        run_output / "artifacts",
        use_persistent_validation_cache=False,
    )
    primary_pointer = run_output / "global-distillation-result.json"
    correction_pointer = run_output / f"{CORRECTION_NAMESPACE}-result.json"
    primary = from_dict(
        ArtifactRef,
        json.loads(primary_pointer.read_text(encoding="utf-8")),
        path="experiment048.primary",
    )
    correction = from_dict(
        ArtifactRef,
        json.loads(correction_pointer.read_text(encoding="utf-8")),
        path="experiment048.correction",
    )
    active = active_global_tuning(run_output)
    primary_result = load_global_tuning(primary, artifacts).result
    correction_result = load_global_tuning(correction, artifacts).result
    if (
        active != correction
        or primary_result.protocol_hash != primary_hash
        or primary_result.steps_completed != PRIMARY_STEPS
        or correction_result.steps_completed != CORRECTION_STEPS[4]
        or correction_result.source_blocks != primary_result.source_blocks
        or correction_result.selected_parameter_count
        != primary_result.selected_parameter_count
    ):
        raise ValueError("Experiment 048 primary or correction endpoint differs")
    candidates = discover_checkpoints(
        run_output,
        set(CORRECTION_STEPS),
        state_namespace=CORRECTION_NAMESPACE,
    )
    checkpoints = _checkpoint_receipts(
        candidates,
        primary=primary,
        source_blocks=primary_result.source_blocks,
        correction_protocol_hash=correction_result.protocol_hash,
    )

    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    selection, confirmation, registry_audit = _reserved_c4_slices(
        registry_payload,
        selection_id=args.selection_slice_id,
        confirmation_id=args.confirmation_slice_id,
    )
    repository_root = Path(__file__).resolve().parent.parent
    bound_files = {
        name: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
        for name, path in {
            "launcher": launcher,
            "manifest": manifest_path,
            "strict_validation": validation_path,
            "slice_registry": registry_path,
            "selector": repository_root
            / "tools"
            / "select_c4_capability_correction_checkpoint.py",
            "c4_evaluator": repository_root / "tools" / "probe_non_wikitext_kd_quality.py",
            "temperature_fitter": repository_root / "tools" / "fit_non_wikitext_temperature.py",
            "checkpoint_materializer": repository_root
            / "tools"
            / "materialize_topk_tail_checkpoint.py",
            "temperature_protocol": repository_root
            / "Docs"
            / "82-temperature-calibration-reporting-protocol.md",
        }.items()
    }
    protocol: dict[str, object] = {
        "experiment": EXPERIMENT_NUMBER,
        "run_output": str(run_output),
        "run_id": manifest.run_id,
        "config_hash": config_hash(config),
        "resident_identity": validation["identity"],
        "primary": {
            "reference": to_dict(primary),
            "protocol_hash": primary_result.protocol_hash,
            "steps": primary_result.steps_completed,
        },
        "correction": {
            "active_reference": to_dict(correction),
            "protocol_hash": correction_result.protocol_hash,
            "state_namespace": CORRECTION_NAMESPACE,
            "checkpoints": checkpoints,
        },
        "selector": {
            "rule": RULE,
            "baseline": {"name": "uncorrected", "steps": PRIMARY_STEPS},
            "arms": [
                {"name": f"correction{epoch}", "steps": steps}
                for epoch, steps in CORRECTION_STEPS.items()
            ],
            "tolerance": SELECTOR_TOLERANCE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "slices": {
            "selection": selection,
            "confirmation": confirmation,
            "registry_audit": registry_audit,
        },
        "bound_files": bound_files,
    }
    receipt = {
        "schema_version": 1,
        "status": "ready_for_selection_evaluation",
        "protocol_hash": semantic_hash(protocol),
        "protocol": protocol,
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != receipt:
            raise FileExistsError(
                f"refusing to replace a different Experiment 048 receipt: {args.output}"
            )
        return 0
    atomic_write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
