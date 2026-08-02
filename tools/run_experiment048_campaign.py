"""Run the fail-closed adaptive Experiment 048 campaign as a resumable stage graph."""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
from recipes import ExperimentDefinition

from nanoquant.compression_quality_workflow import (
    CompressionQualityExperiment,
    ResolvedCompressionQualityExperiment,
    execute_compression_quality_experiment,
    resolve_compression_quality_experiment,
)
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    execute_resident_workflow,
    resolve_resident_experiment_inputs,
)

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "experiments/048-adaptive-capability-correction-d2-gemma-3-1b-it.py"
REGISTRY = ROOT / "Docs/evaluation-slice-registry.json"
CORRECTION_NAMESPACE = "global-distillation-mass-floor"
DEFAULT_REFERENCES = (
    (
        "accepted040",
        ROOT / "evidence/040/040-low-pressure-weight2-epoch1-fold1p015-gemma-3-1b-it",
        32,
    ),
    (
        "tailaware044",
        ROOT / "evidence/044/044-tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it",
        256,
    ),
)


@dataclass(frozen=True, slots=True)
class CampaignPaths:
    root: Path
    resident: Path
    strict_validation: Path
    receipt: Path
    selection_quality: Path
    selection_checkpoint: Path
    decision: Path
    derived: Path
    derived_validation: Path
    fallback: Path
    fallback_validation: Path
    baseline_temperature: Path
    selected_temperature: Path
    confirmation_quality: Path
    summary: Path


def _campaign_paths(root: Path) -> CampaignPaths:
    root = root.resolve()
    resident = ROOT / "evidence/048/048-adaptive-capability-correction-d2-gemma-3-1b-it"
    selection_quality = root / "selection-c4.json"
    return CampaignPaths(
        root,
        resident,
        root / "resident-strict-validation.json",
        root / "campaign-receipt.json",
        selection_quality,
        selection_quality.with_name(selection_quality.stem + ".checkpoint.json"),
        root / "selection-decision.json",
        ROOT / "evidence/048/048-adaptive-capability-correction-selected",
        root / "selected-strict-validation.json",
        ROOT / "evidence/048/048-adaptive-capability-correction-fallback",
        root / "fallback-strict-validation.json",
        root / "temperature-uncorrected.json",
        root / "temperature-selected.json",
        root / "confirmation-c4.json",
        root / "campaign-summary.json",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "evidence/048/campaign")
    parser.add_argument("--slice-registry", type=Path, default=REGISTRY)
    parser.add_argument("--selection-slice-id")
    parser.add_argument("--confirmation-slice-id")
    parser.add_argument("--selection-offset", type=int, default=344)
    parser.add_argument("--confirmation-offset", type=int, default=392)
    parser.add_argument("--c4-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("resident", "complete"), help=argparse.SUPPRESS)
    parser.add_argument("--derived-run", type=Path, help=argparse.SUPPRESS)
    return parser


def _load_definition() -> ExperimentDefinition[CompressionQualityExperiment]:
    namespace = runpy.run_path(str(LAUNCHER), run_name="nanoquant_experiment048_campaign")
    definition = namespace.get("EXPERIMENT")
    if not isinstance(definition, ExperimentDefinition) or not isinstance(
        definition.workflow, CompressionQualityExperiment
    ):
        raise TypeError("Experiment 048 launcher does not expose a compression-quality definition")
    return cast(ExperimentDefinition[CompressionQualityExperiment], definition)


def _worker(args: argparse.Namespace) -> int:
    if args.snapshot is None:
        raise ValueError("campaign worker requires --snapshot")
    definition = _load_definition()
    if args.worker == "resident":
        inputs = resolve_resident_experiment_inputs(definition.config, launcher_path=LAUNCHER)
        if inputs.snapshot != args.snapshot.resolve():
            raise ValueError("resolved Experiment 048 snapshot differs from --snapshot")
        execute_resident_workflow(
            definition.config,
            inputs,
            ResidentExecutionOptions(maximum_wddm_shared_bytes=int(0.75 * 2**30)),
        )
        return 0
    if args.worker == "complete":
        if args.derived_run is None:
            raise ValueError("complete worker requires --derived-run")
        inputs = resolve_resident_experiment_inputs(
            definition.config,
            launcher_path=LAUNCHER,
            output_override=args.derived_run,
        )
        if inputs.snapshot != args.snapshot.resolve():
            raise ValueError("resolved Experiment 048 snapshot differs from --snapshot")
        resolved_base = resolve_compression_quality_experiment(
            definition.config,
            definition.workflow,
            launcher_path=LAUNCHER,
        )
        resolved = replace(resolved_base, inputs=inputs)
        execute_compression_quality_experiment(
            definition.config,
            replace(definition.workflow, allow_relocated_completed_run=True),
            cast(ResolvedCompressionQualityExperiment, resolved),
        )
        return 0
    raise ValueError("unknown Experiment 048 campaign worker")


def _python(tool: str, *arguments: object) -> list[str]:
    return [sys.executable, str(ROOT / "tools" / tool), *(str(item) for item in arguments)]


def _arm(name: str, mode: str, run: Path, *extra: object) -> str:
    return f"{name}={';'.join((mode, str(run.resolve()), *(str(item) for item in extra)))}"


def _static_plan(paths: CampaignPaths) -> tuple[str, ...]:
    return (
        "resident: fresh factorization + primary256 + correction32/64/96/128",
        "strict-resident-validation",
        "immutable-campaign-receipt",
        "open-and-retire-selection-slice",
        "selection-c4: uncorrected + four same-run correction checkpoints",
        "immutable-checkpoint-decision",
        "materialize-and-strictly-validate-selected-checkpoint-or-primary-fallback",
        "if corrected: fit both temperatures on retired selection data",
        "if corrected: open final slice and run raw/fitted absolute C4 confirmation",
        "execute-complete-compression on selected derived run",
        "logical + packed + GGUF + WikiText + 1000-example six-task quality",
    )


def _completed_json(path: Path, *, status: str = "completed") -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == status


def _slice_retired(registry: Path, identity: str, role: str) -> bool:
    if not registry.is_file():
        return False
    payload = json.loads(registry.read_text(encoding="utf-8"))
    matches = [entry for entry in payload.get("slices", []) if entry.get("id") == identity]
    return (
        len(matches) == 1
        and matches[0].get("status") == "retired"
        and matches[0].get("retirement", {}).get("role") == role
    )


def _materialized(paths: CampaignPaths) -> bool:
    return _materialized_run(paths.derived)


def _materialized_run(run_output: Path) -> bool:
    receipt = run_output / "topk-tail-materialization.json"
    if not receipt.is_file():
        return False
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    return (
        payload.get("schema_version") == 2
        and payload.get("exact_reload_audit", {}).get("passed") is True
        and _completed_json(run_output / "manifest.json")
    )


def _complete_summary(path: Path, run_output: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Path(str(payload.get("compression", {}).get("run_output"))).resolve() == run_output.resolve() and isinstance(
        payload.get("passed"), bool
    )


def _run_stage(name: str, command: list[str], complete: Any, root: Path) -> None:
    if complete():
        return
    command = [str(item) for item in command]
    root.mkdir(parents=True, exist_ok=True)
    with (
        (root / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout,
        (root / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Experiment 048 stage {name} failed with exit code {result.returncode}")
    if not complete():
        raise RuntimeError(f"Experiment 048 stage {name} did not produce valid completion evidence")


def _reference_arguments() -> list[str]:
    arguments: list[str] = []
    for name, path, steps in DEFAULT_REFERENCES:
        arguments.extend(("--retained-reference", f"{name}={path.resolve()};{steps}"))
    return arguments


def _selection_evaluation_command(args: argparse.Namespace, paths: CampaignPaths) -> list[str]:
    command = _python(
        "probe_non_wikitext_kd_quality.py",
        "--snapshot",
        args.snapshot.resolve(),
        "--output",
        paths.selection_quality,
    )
    command.extend(
        ("--arm", _arm("uncorrected", "tuning", paths.resident, paths.resident / "global-distillation-result.json"))
    )
    for epoch in range(1, 5):
        command.extend(
            (
                "--arm",
                _arm(
                    "correction" + str(epoch), "checkpoint", paths.resident, paths.resident, epoch, CORRECTION_NAMESPACE
                ),
            )
        )
    command.extend(("--primary-baseline", "uncorrected", "--primary-candidate", "correction4"))
    for name, steps in (("uncorrected", 256), *((f"correction{epoch}", epoch * 32) for epoch in range(1, 5))):
        command.extend(("--expected-steps", f"{name}={steps}"))
    command.extend(
        (
            "--slice-registry",
            args.slice_registry.resolve(),
            "--slice-id",
            args.selection_slice_id,
            "--offset",
            args.selection_offset,
            "--samples",
            48,
            "--sequence-length",
            512,
            "--c4-file",
            args.c4_file,
            "--device",
            args.device,
        )
    )
    if args.local_files_only:
        command.append("--local-files-only")
    return [str(item) for item in command]


def _selected(decision: dict[str, Any]) -> tuple[str, int, int, str, bool]:
    name = str(decision.get("selected_arm"))
    steps = int(decision.get("selected_steps", -1))
    if name == "uncorrected" and steps == 256 and decision.get("correction_applied") is False:
        return name, steps, 8, "global-distillation", False
    if name.startswith("correction") and name[-1:] in {"1", "2", "3", "4"}:
        epoch = int(name[-1])
        if steps == epoch * 32 and decision.get("correction_applied") is True:
            return name, steps, epoch, CORRECTION_NAMESPACE, True
    raise ValueError("Experiment 048 selector decision has an invalid selected arm")


def _fit_command(
    args: argparse.Namespace,
    paths: CampaignPaths,
    *,
    name: str,
    arm: str,
    role: str,
    steps: int,
    output: Path,
) -> list[str]:
    command = _python(
        "fit_non_wikitext_temperature.py",
        "--snapshot",
        args.snapshot.resolve(),
        "--output",
        output,
        "--arm",
        arm,
        "--role",
        role,
        "--expected-steps",
        steps,
        "--selection-decision",
        paths.decision,
        "--slice-registry",
        args.slice_registry.resolve(),
        "--slice-id",
        args.selection_slice_id,
        "--offset",
        args.selection_offset,
        "--samples",
        48,
        "--sequence-length",
        512,
        "--c4-file",
        args.c4_file,
        "--device",
        args.device,
    )
    if args.local_files_only:
        command.append("--local-files-only")
    if not arm.startswith(name + "="):
        raise ValueError("temperature-fit arm name differs")
    return [str(item) for item in command]


def _confirmation_command(
    args: argparse.Namespace,
    paths: CampaignPaths,
    *,
    selected_name: str,
    selected_steps: int,
) -> list[str]:
    command = _python(
        "probe_non_wikitext_kd_quality.py",
        "--snapshot",
        args.snapshot.resolve(),
        "--output",
        paths.confirmation_quality,
    )
    arm_inventory = (
        ("prekd", _arm("prekd", "prekd", paths.resident), 0, True),
        (
            "uncorrected",
            _arm(
                "uncorrected",
                "tuning",
                paths.resident,
                paths.resident / "global-distillation-result.json",
            ),
            256,
            False,
        ),
        (
            selected_name,
            _arm(
                selected_name,
                "checkpoint",
                paths.resident,
                paths.resident,
                selected_steps // 32,
                CORRECTION_NAMESPACE,
            ),
            selected_steps,
            False,
        ),
        *((name, _arm(name, "postkd", reference), steps, True) for name, reference, steps in DEFAULT_REFERENCES),
    )
    for name, arm, steps, reference in arm_inventory:
        command.extend(("--arm", arm, "--expected-steps", f"{name}={steps}"))
        if reference:
            command.extend(("--reference-arm", name))
    command.extend(
        (
            "--primary-baseline",
            "uncorrected",
            "--primary-candidate",
            selected_name,
            "--temperature-fit-receipt",
            f"uncorrected={paths.baseline_temperature}",
            "--temperature-fit-receipt",
            f"{selected_name}={paths.selected_temperature}",
            "--slice-registry",
            args.slice_registry.resolve(),
            "--slice-id",
            args.confirmation_slice_id,
            "--offset",
            args.confirmation_offset,
            "--samples",
            48,
            "--sequence-length",
            512,
            "--c4-file",
            args.c4_file,
            "--device",
            args.device,
        )
    )
    if args.local_files_only:
        command.append("--local-files-only")
    return [str(item) for item in command]


def _run(args: argparse.Namespace) -> int:
    if args.snapshot is None or args.c4_file is None or not args.selection_slice_id or not args.confirmation_slice_id:
        raise ValueError("campaign requires snapshot, local C4 data, and both immutable slice identities")
    args.c4_file = args.c4_file.resolve()
    if not args.c4_file.is_file():
        raise ValueError("campaign C4 data file is missing")
    paths = _campaign_paths(args.output_root)
    if args.dry_run:
        print(
            json.dumps(
                {"schema_version": 1, "paths": asdict(paths), "stages": _static_plan(paths)},
                indent=2,
                default=str,
            )
        )
        return 0
    definition = _load_definition()
    if definition.workflow.task_limit != 1000:
        raise ValueError("Experiment 048 task guardrail is not 1,000 examples")
    _run_stage(
        "resident",
        _python("run_experiment048_campaign.py", "--worker", "resident", "--snapshot", args.snapshot.resolve()),
        lambda: _completed_json(paths.resident / "manifest.json"),
        paths.root,
    )
    _run_stage(
        "validate-resident",
        _python(
            "validate_resident_run.py",
            "--run-output",
            paths.resident,
            "--require-complete",
            "--output",
            paths.strict_validation,
        ),
        lambda: (
            paths.strict_validation.is_file()
            and json.loads(paths.strict_validation.read_text(encoding="utf-8")).get("complete") is True
        ),
        paths.root,
    )
    receipt_command = _python(
        "prepare_experiment048_campaign_receipt.py",
        "--run-output",
        paths.resident,
        "--launcher",
        LAUNCHER,
        "--strict-validation",
        paths.strict_validation,
        "--slice-registry",
        args.slice_registry.resolve(),
        "--selection-slice-id",
        args.selection_slice_id,
        "--confirmation-slice-id",
        args.confirmation_slice_id,
        "--c4-file",
        args.c4_file,
        *_reference_arguments(),
        "--output",
        paths.receipt,
    )
    _run_stage(
        "campaign-receipt",
        receipt_command,
        lambda: _completed_json(paths.receipt, status="ready_for_selection_evaluation"),
        paths.root,
    )
    _run_stage(
        "open-selection",
        _python(
            "open_experiment048_c4_slice.py",
            "--campaign-receipt",
            paths.receipt,
            "--slice-registry",
            args.slice_registry.resolve(),
            "--role",
            "selection",
        ),
        lambda: _slice_retired(args.slice_registry.resolve(), args.selection_slice_id, "selection"),
        paths.root,
    )
    _run_stage(
        "selection-c4",
        _selection_evaluation_command(args, paths),
        lambda: _completed_json(paths.selection_quality),
        paths.root,
    )
    decision_command = _python(
        "select_c4_capability_correction_checkpoint.py",
        "--quality-output",
        paths.selection_quality,
        "--checkpoint-output",
        paths.selection_checkpoint,
        "--output",
        paths.decision,
        "--baseline",
        "uncorrected=256",
    )
    for epoch in range(1, 5):
        decision_command.extend(("--ordered-arm", f"correction{epoch}={epoch * 32}"))
    decision_command.extend(("--tolerance", 0.01, "--resamples", 10_000, "--seed", 0))
    _run_stage(
        "select-checkpoint",
        decision_command,
        lambda: (
            paths.decision.is_file()
            and json.loads(paths.decision.read_text(encoding="utf-8")).get("schema_version") == 1
        ),
        paths.root,
    )
    decision = cast(dict[str, Any], json.loads(paths.decision.read_text(encoding="utf-8")))
    selected_name, selected_steps, epoch, namespace, correction_applied = _selected(decision)
    _run_stage(
        "materialize-selected",
        _python(
            "materialize_topk_tail_checkpoint.py",
            "--run-output",
            paths.resident,
            "--snapshot",
            args.snapshot.resolve(),
            "--checkpoint-output",
            paths.resident,
            "--epoch",
            epoch,
            "--state-namespace",
            namespace,
            "--derived-run-output",
            paths.derived,
        ),
        lambda: _materialized(paths),
        paths.root,
    )
    _run_stage(
        "validate-selected",
        _python(
            "validate_resident_run.py",
            "--run-output",
            paths.derived,
            "--require-complete",
            "--output",
            paths.derived_validation,
        ),
        lambda: (
            paths.derived_validation.is_file()
            and json.loads(paths.derived_validation.read_text(encoding="utf-8")).get("complete") is True
        ),
        paths.root,
    )
    if correction_applied:
        baseline_arm = _arm(
            "uncorrected",
            "tuning",
            paths.resident,
            paths.resident / "global-distillation-result.json",
        )
        selected_arm = _arm(
            selected_name,
            "checkpoint",
            paths.resident,
            paths.resident,
            epoch,
            CORRECTION_NAMESPACE,
        )
        _run_stage(
            "fit-temperature-uncorrected",
            _fit_command(
                args,
                paths,
                name="uncorrected",
                arm=baseline_arm,
                role="baseline",
                steps=256,
                output=paths.baseline_temperature,
            ),
            lambda: _completed_json(paths.baseline_temperature),
            paths.root,
        )
        _run_stage(
            "fit-temperature-selected",
            _fit_command(
                args,
                paths,
                name=selected_name,
                arm=selected_arm,
                role="selected",
                steps=selected_steps,
                output=paths.selected_temperature,
            ),
            lambda: _completed_json(paths.selected_temperature),
            paths.root,
        )
        _run_stage(
            "open-confirmation",
            _python(
                "open_experiment048_c4_slice.py",
                "--campaign-receipt",
                paths.receipt,
                "--slice-registry",
                args.slice_registry.resolve(),
                "--role",
                "confirmation",
            ),
            lambda: _slice_retired(args.slice_registry.resolve(), args.confirmation_slice_id, "confirmation"),
            paths.root,
        )
        _run_stage(
            "confirmation-c4",
            _confirmation_command(
                args,
                paths,
                selected_name=selected_name,
                selected_steps=selected_steps,
            ),
            lambda: _completed_json(paths.confirmation_quality),
            paths.root,
        )
    confirmation_passed = (
        not correction_applied
        or json.loads(paths.confirmation_quality.read_text(encoding="utf-8"))
        .get("primary_comparison", {})
        .get("passes")
        is True
    )
    deployed_run = paths.derived
    deployed_validation = paths.derived_validation
    if correction_applied and not confirmation_passed:
        _run_stage(
            "materialize-fallback",
            _python(
                "materialize_topk_tail_checkpoint.py",
                "--run-output",
                paths.resident,
                "--snapshot",
                args.snapshot.resolve(),
                "--checkpoint-output",
                paths.resident,
                "--epoch",
                8,
                "--state-namespace",
                "global-distillation",
                "--derived-run-output",
                paths.fallback,
            ),
            lambda: _materialized_run(paths.fallback),
            paths.root,
        )
        _run_stage(
            "validate-fallback",
            _python(
                "validate_resident_run.py",
                "--run-output",
                paths.fallback,
                "--require-complete",
                "--output",
                paths.fallback_validation,
            ),
            lambda: (
                paths.fallback_validation.is_file()
                and json.loads(paths.fallback_validation.read_text(encoding="utf-8")).get("complete") is True
            ),
            paths.root,
        )
        deployed_run = paths.fallback
        deployed_validation = paths.fallback_validation
    final_summary = ROOT / definition.workflow.summary_output
    _run_stage(
        "complete-compression-quality",
        _python(
            "run_experiment048_campaign.py",
            "--worker",
            "complete",
            "--snapshot",
            args.snapshot.resolve(),
            "--derived-run",
            deployed_run,
        ),
        lambda: _complete_summary(final_summary, deployed_run),
        paths.root,
    )
    deployment_quality_passed = json.loads(final_summary.read_text(encoding="utf-8")).get("passed") is True
    summary = {
        "schema_version": 1,
        "status": "completed",
        "correction_applied": correction_applied,
        "confirmation_passed": confirmation_passed,
        "deployment_quality_passed": deployment_quality_passed,
        "policy_accepted": (correction_applied and confirmation_passed and deployment_quality_passed),
        "selected_arm": selected_name,
        "selected_steps": selected_steps,
        "selection_decision": str(paths.decision),
        "selection_slice_retired": True,
        "confirmation_slice_opened": correction_applied,
        "confirmation_quality": str(paths.confirmation_quality) if correction_applied else None,
        "selected_run": str(paths.derived),
        "deployed_run": str(deployed_run),
        "strict_validation": str(deployed_validation),
        "complete_compression_summary": str(final_summary),
    }
    if paths.summary.is_file():
        if json.loads(paths.summary.read_text(encoding="utf-8")) != summary:
            raise FileExistsError("Experiment 048 campaign summary differs")
    else:
        atomic_write_json(paths.summary, summary)
    return 0


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    return _worker(args) if args.worker else _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
