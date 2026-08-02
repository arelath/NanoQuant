"""Fit one identity-bound logit temperature on a retired C4 selection slice."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch

from nanoquant.application.temperature_calibration import (
    TEMPERATURE_CALIBRATION_VERSION,
    TemperatureFitIteration,
    TemperatureNllStatistics,
    combine_temperature_nll_statistics,
    fit_logit_temperature,
    temperature_nll_statistics,
)
from nanoquant.config.codec import from_dict, semantic_hash, to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.global_tuning import load_global_tuning
from nanoquant.infrastructure.io_utils import hash_file
from nanoquant.infrastructure.reproducibility import deterministic_torch_execution
from nanoquant.infrastructure.temperature_fit_checkpoint import (
    complete_temperature_fit_receipt,
    load_temperature_fit_iterations,
    load_temperature_fit_receipt,
    write_temperature_fit_progress,
)

_QUALITY_MODULE_NAME = (
    "tools.probe_non_wikitext_kd_quality" if __package__ else "probe_non_wikitext_kd_quality"
)
_quality = importlib.import_module(_QUALITY_MODULE_NAME)

C4_DATASET = cast(str, _quality.C4_DATASET)
C4_REVISION = cast(str, _quality.C4_REVISION)
C4_VALIDATION_FILE = cast(str, _quality.C4_VALIDATION_FILE)
MODEL_SOURCE = cast(str, _quality.MODEL_SOURCE)
PINNED_MODEL_REVISION = cast(str, _quality.PINNED_MODEL_REVISION)

INITIAL_LOGIT_SCALE = 1.0
MINIMUM_LOGIT_SCALE = 0.5
MAXIMUM_LOGIT_SCALE = 1.5
MAXIMUM_UPDATE_PASSES = 4
CONVERGENCE_TOLERANCE = 1e-4
HESSIAN_FLOOR = 1e-12
SOLVER_SEED = 0

Arm = tuple[str, str, Path, Path | None, int | None, str]


def _arm(value: str) -> Arm:
    return cast(Arm, _quality._parse_arm(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", type=_arm, required=True)
    parser.add_argument("--role", choices=("baseline", "selected"), required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--selection-decision", type=Path, required=True)
    parser.add_argument("--slice-registry", type=Path, required=True)
    parser.add_argument("--slice-id", required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--c4-revision", default=C4_REVISION)
    parser.add_argument("--c4-file", default=C4_VALIDATION_FILE)
    parser.add_argument("--c4-documents", type=int, default=1_100)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--interrupt-after-iterations", type=int, help=argparse.SUPPRESS)
    return parser


def _sha256(path: Path) -> str:
    return "sha256:" + hash_file(path)


def _retired_slice(
    path: Path,
    slice_id: str,
    *,
    offset: int,
    samples: int,
    sequence_length: int,
    token_hash: str,
) -> tuple[dict[str, object], str]:
    encoded = path.read_bytes()
    registry = json.loads(encoded)
    entries = registry.get("slices")
    if registry.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError("evaluation slice registry is invalid")
    matches = [item for item in entries if item.get("id") == slice_id]
    if len(matches) != 1:
        raise ValueError("temperature-fit slice is missing or ambiguous")
    selected = matches[0]
    expected = {
        "dataset": C4_DATASET,
        "split": "validation",
        "offset": offset,
        "samples": samples,
        "sequence_length": sequence_length,
        "token_start": offset * sequence_length,
        "token_end": (offset + samples) * sequence_length,
        "token_hash": token_hash,
        "status": "retired",
    }
    if any(selected.get(key) != value for key, value in expected.items()):
        raise ValueError("temperature fit requires the exact retired selection slice")
    return cast(dict[str, object], selected), "sha256:" + hashlib.sha256(encoded).hexdigest()


def _selection_arm_identity(
    decision_path: Path,
    *,
    role: str,
    name: str,
    expected_steps: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("schema_version") != 1 or not isinstance(decision.get("protocol"), dict):
        raise ValueError("temperature fit selection decision is invalid")
    if role == "selected":
        if decision.get("selected_arm") != name or decision.get("selected_steps") != expected_steps:
            raise ValueError("temperature fit arm differs from the selected checkpoint")
        selected_identity = decision.get("selected_identity")
        if not isinstance(selected_identity, dict):
            raise ValueError("temperature fit selection decision has no selected identity")
    else:
        baseline = decision.get("baseline")
        if not isinstance(baseline, dict) or baseline.get("name") != name or baseline.get("steps") != expected_steps:
            raise ValueError("temperature fit arm differs from the selection baseline")

    selection_protocol = cast(dict[str, object], decision["protocol"])
    quality_path_value = selection_protocol.get("quality_output")
    quality_hash = selection_protocol.get("quality_sha256")
    if not isinstance(quality_path_value, str) or not isinstance(quality_hash, str):
        raise ValueError("temperature fit decision does not bind its quality evidence")
    quality_path = Path(quality_path_value)
    if hash_file(quality_path) != quality_hash.removeprefix("sha256:"):
        raise ValueError("temperature fit selection quality evidence hash differs")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    arms = quality.get("arms")
    quality_protocol = quality.get("protocol")
    if (
        quality.get("status") != "completed"
        or not isinstance(arms, dict)
        or not isinstance(quality_protocol, dict)
    ):
        raise ValueError("temperature fit selection quality evidence is incomplete")
    expected_identity = arms.get(name)
    if not isinstance(expected_identity, dict):
        raise ValueError("temperature fit arm is absent from selection quality evidence")
    if role == "selected" and decision.get("selected_identity") != expected_identity:
        raise ValueError("temperature fit selected identity differs across decision evidence")
    return (
        cast(dict[str, object], decision),
        cast(dict[str, object], expected_identity),
        cast(dict[str, object], quality_protocol),
    )


def _load_arm(
    arm: Arm,
    *,
    snapshot: Path,
    model_revision: str,
    expected_steps: int,
    device: str,
) -> tuple[Any, dict[str, object], dict[str, object]]:
    name, mode, run_output, tuning_pointer, epoch, state_namespace = arm
    global_tuning_override = None
    if mode == "tuning" and tuning_pointer is not None:
        global_tuning_override = from_dict(
            ArtifactRef,
            json.loads(tuning_pointer.read_text(encoding="utf-8")),
            path=f"arm[{name}].global_tuning",
        )
    load_device = "cpu" if mode == "checkpoint" else device
    loaded = load_frozen_run(
        run_output,
        snapshot,
        source_name=MODEL_SOURCE,
        revision=model_revision,
        device=load_device,
        verify_hashes=False,
        backend="factorized",
        use_global_tuning=mode not in {"prekd", "checkpoint"},
        global_tuning_override=global_tuning_override,
    )
    checkpoint_receipt = None
    if mode == "checkpoint" and tuning_pointer is not None and epoch is not None:
        checkpoint_receipt = _quality._apply_checkpoint(
            loaded,
            run_output,
            tuning_pointer,
            epoch,
            state_namespace,
        )
        loaded.model.to(device)
    if checkpoint_receipt is not None:
        observed_steps = int(checkpoint_receipt["steps"])
    elif loaded.global_tuning is None and mode == "prekd":
        observed_steps = 0
    else:
        observed_steps = load_global_tuning(
            cast(ArtifactRef, loaded.global_tuning),
            LocalArtifactStore(run_output / "artifacts"),
        ).result.steps_completed
    if observed_steps != expected_steps:
        raise ValueError(
            f"temperature fit arm {name} has {observed_steps} steps; expected {expected_steps}"
        )
    manifest: dict[str, object] = {
        "mode": mode,
        "run_output": str(run_output.resolve()),
        "global_tuning": None if loaded.global_tuning is None else to_dict(loaded.global_tuning),
        "global_tuning_pointer": (
            str(cast(Path, tuning_pointer).resolve()) if mode == "tuning" else None
        ),
        "checkpoint": checkpoint_receipt,
        "checkpoint_state_namespace": (
            state_namespace if mode == "checkpoint" else None
        ),
        "steps_completed": observed_steps,
    }
    frozen_identity = {
        "model_hash": loaded.identity.model_hash,
        "config_hash": loaded.identity.config_hash,
        "plan_hash": loaded.identity.plan_hash,
    }
    return loaded, manifest, cast(dict[str, object], frozen_identity)


@torch.no_grad()
def _evaluate_temperature_statistics(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    logit_scale: float,
    device: str,
    token_chunk_size: int,
) -> TemperatureNllStatistics:
    model.eval()
    values = []
    for index in range(tokens.shape[0]):
        batch = tokens[index : index + 1].to(device)
        logits = cast(Any, model)(input_ids=batch, use_cache=False).logits
        values.append(
            temperature_nll_statistics(
                logits,
                batch,
                logit_scale=logit_scale,
                token_chunk_size=token_chunk_size,
            )
        )
        print(
            f"temperature scale {logit_scale:.9g}: {index + 1}/{tokens.shape[0]} sequences",
            flush=True,
        )
        del batch, logits
    return combine_temperature_nll_statistics(values)


def run(args: argparse.Namespace) -> int:
    if (
        args.expected_steps < 0
        or args.offset < 0
        or min(args.c4_documents, args.samples, args.sequence_length - 1, args.token_chunk_size) <= 0
        or (
            args.interrupt_after_iterations is not None
            and args.interrupt_after_iterations <= 0
        )
    ):
        raise ValueError("temperature-fit protocol is invalid")
    name, _mode, _run_output, _pointer, _epoch, _namespace = cast(Arm, args.arm)
    decision, expected_arm_identity, selection_quality_protocol = _selection_arm_identity(
        args.selection_decision,
        role=args.role,
        name=name,
        expected_steps=args.expected_steps,
    )
    tokens, fingerprint, bos_token_id = _quality._load_c4_tokens(
        args.snapshot,
        revision=args.c4_revision,
        data_file=args.c4_file,
        documents=args.c4_documents,
        offset=args.offset,
        samples=args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    token_hash = _quality._token_hash(tokens)
    reservation, registry_hash = _retired_slice(
        args.slice_registry,
        args.slice_id,
        offset=args.offset,
        samples=args.samples,
        sequence_length=args.sequence_length,
        token_hash=token_hash,
    )
    selection_reservation = selection_quality_protocol.get("slice_reservation")
    expected_selection_protocol = {
        "dataset": C4_DATASET,
        "dataset_revision": args.c4_revision,
        "data_file": args.c4_file,
        "documents": args.c4_documents,
        "document_join": "single-space",
        "tokenizer_policy": "source-model-add-special-tokens",
        "offset": args.offset,
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "token_hash": token_hash,
        "model_revision": args.model_revision,
    }
    if any(
        selection_quality_protocol.get(key) != value
        for key, value in expected_selection_protocol.items()
    ) or not isinstance(selection_reservation, dict) or any(
        selection_reservation.get(key) != value
        for key, value in {
            "id": args.slice_id,
            "offset": args.offset,
            "samples": args.samples,
            "sequence_length": args.sequence_length,
            "token_hash": token_hash,
        }.items()
    ):
        raise ValueError("temperature fit does not use the selector's exact C4 slice protocol")
    checkpoint_path = args.output.with_name(args.output.stem + ".checkpoint.json")
    with acquire_device_lease(args.device), deterministic_torch_execution(SOLVER_SEED, args.device):
        loaded, observed_arm_identity, frozen_identity = _load_arm(
            cast(Arm, args.arm),
            snapshot=args.snapshot,
            model_revision=args.model_revision,
            expected_steps=args.expected_steps,
            device=args.device,
        )
        if observed_arm_identity != expected_arm_identity:
            raise ValueError("temperature fit loaded arm differs from the selection decision")
        protocol: dict[str, object] = {
            "producer": {"name": "non-wikitext-temperature-fit", "version": 1},
            "solver": {
                "version": TEMPERATURE_CALIBRATION_VERSION,
                "seed": SOLVER_SEED,
                "initial_logit_scale": INITIAL_LOGIT_SCALE,
                "minimum_logit_scale": MINIMUM_LOGIT_SCALE,
                "maximum_logit_scale": MAXIMUM_LOGIT_SCALE,
                "maximum_update_passes": MAXIMUM_UPDATE_PASSES,
                "convergence_tolerance": CONVERGENCE_TOLERANCE,
                "hessian_floor": HESSIAN_FLOOR,
                "accumulation": "fixed-sequence-order-fsum-float32-softmax-v1",
            },
            "selection": {
                "decision": str(args.selection_decision.resolve()),
                "decision_sha256": _sha256(args.selection_decision),
                "rule": decision.get("rule"),
                "selected_arm": decision.get("selected_arm"),
                "role": args.role,
            },
            "dataset": {
                "name": C4_DATASET,
                "revision": args.c4_revision,
                "data_file": args.c4_file,
                "fingerprint": fingerprint,
                "document_join": "single-space",
                "tokenizer_policy": "source-model-add-special-tokens",
                "bos_token_id": bos_token_id,
                "documents": args.c4_documents,
            },
            "slice": {
                "registry": str(args.slice_registry.resolve()),
                "registry_sha256": registry_hash,
                "reservation": reservation,
                "token_hash": token_hash,
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "token_chunk_size": args.token_chunk_size,
            },
            "model": {
                "source": MODEL_SOURCE,
                "revision": args.model_revision,
                "snapshot": str(args.snapshot.resolve()),
                "config_sha256": _sha256(args.snapshot / "config.json"),
                "frozen_identity": frozen_identity,
            },
            "arm": {"name": name, **observed_arm_identity},
        }
        if args.output.is_file():
            completed_protocol, completed_result = load_temperature_fit_receipt(args.output)
            if completed_protocol != protocol:
                raise ValueError("existing temperature-fit receipt protocol differs")
            print(
                json.dumps(
                    {
                        "output": str(args.output.resolve()),
                        "protocol_hash": semantic_hash(protocol),
                        "logit_scale": completed_result.final_logit_scale,
                        "temperature": completed_result.equivalent_temperature,
                        "reused": True,
                    },
                    indent=2,
                )
            )
            return 0
        resume = load_temperature_fit_iterations(checkpoint_path, protocol)
        completed_before = len(resume)

        def evaluate(scale: float) -> TemperatureNllStatistics:
            return _evaluate_temperature_statistics(
                loaded.model,
                tokens,
                logit_scale=scale,
                device=args.device,
                token_chunk_size=args.token_chunk_size,
            )

        def checkpoint(iterations: tuple[TemperatureFitIteration, ...]) -> None:
            write_temperature_fit_progress(checkpoint_path, protocol, iterations)
            if (
                args.interrupt_after_iterations is not None
                and len(iterations) - completed_before >= args.interrupt_after_iterations
            ):
                raise InterruptedError("requested interruption after temperature-fit iteration")

        result = fit_logit_temperature(
            evaluate,
            initial_logit_scale=INITIAL_LOGIT_SCALE,
            minimum_logit_scale=MINIMUM_LOGIT_SCALE,
            maximum_logit_scale=MAXIMUM_LOGIT_SCALE,
            maximum_update_passes=MAXIMUM_UPDATE_PASSES,
            convergence_tolerance=CONVERGENCE_TOLERANCE,
            hessian_floor=HESSIAN_FLOOR,
            resume=resume,
            checkpoint=checkpoint,
        )
        complete_temperature_fit_receipt(args.output, checkpoint_path, protocol, result)
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "protocol_hash": semantic_hash(protocol),
                "logit_scale": result.final_logit_scale,
                "temperature": result.equivalent_temperature,
            },
            indent=2,
        )
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
