"""Compare independently materialized KD arms on a held-out WikiText slice."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_distillation_checkpoint_tail_mass import discover_checkpoints
from probe_factorized_component_overlays_kl import _dtype
from probe_mlp_overlays_kl import _split_tokens
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION

from nanoquant.config.codec import from_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.global_distillation import _selected_parameters, _thaw_frozen_layers
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.distillation_checkpoint import load_distillation_checkpoint
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.global_tuning import load_global_tuning
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_model_protocol import HuggingFaceModel
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.kl_budget_workflow import _token_hash


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    negative_log_likelihood: float
    full_kl: float
    conditional_topk_kl: float
    topk_plus_tail_kl: float
    teacher_topk_mass: float
    student_teacher_topk_mass: float
    tail_mass_absolute_error: float
    token_count: int


def _parse_arm(value: str) -> tuple[str, str, Path, Path | None, int | None]:
    name, separator, specification = value.partition("=")
    parts = specification.split(";")
    if not separator or not name.strip() or not specification.strip():
        raise argparse.ArgumentTypeError(
            "arm must use name=materialized-run-output, "
            "name=checkpoint;frozen-run-output;checkpoint-output;epoch, or "
            "name=tuning;run-output;artifact-reference-json"
        )
    if parts[0] == "tuning":
        if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
            raise argparse.ArgumentTypeError(
                "tuning arm must use name=tuning;run-output;artifact-reference-json"
            )
        return name.strip(), "tuning", Path(parts[1].strip()), Path(parts[2].strip()), None
    if parts[0] != "checkpoint":
        return name.strip(), "postkd", Path(specification.strip()), None, None
    if len(parts) != 4 or not parts[1].strip() or not parts[2].strip():
        raise argparse.ArgumentTypeError(
            "checkpoint arm must use "
            "name=checkpoint;frozen-run-output;checkpoint-output;epoch"
        )
    try:
        epoch = int(parts[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint arm epoch must be an integer") from exc
    if epoch <= 0:
        raise argparse.ArgumentTypeError("checkpoint arm epoch must be positive")
    return (
        name.strip(),
        "checkpoint",
        Path(parts[1].strip()),
        Path(parts[2].strip()),
        epoch,
    )


def _parse_expected_steps(value: str) -> tuple[str, int]:
    name, separator, steps = value.partition("=")
    try:
        parsed = int(steps)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected steps must use arm=positive-integer") from exc
    if not separator or not name.strip() or parsed <= 0:
        raise argparse.ArgumentTypeError("expected steps must use arm=positive-integer")
    return name.strip(), parsed


def _apply_checkpoint(
    loaded: Any,
    frozen_run_output: Path,
    checkpoint_output: Path,
    epoch: int,
) -> dict[str, object]:
    frozen_artifacts = LocalArtifactStore(frozen_run_output / "artifacts")
    trainable = _thaw_frozen_layers(loaded, LocalTensorStore(frozen_artifacts))
    selected_ids, _auxiliary = _selected_parameters(loaded.model, trainable)
    selected = {
        name: parameter
        for name, parameter in loaded.model.named_parameters()
        if id(parameter) in selected_ids
    }
    candidate = discover_checkpoints(checkpoint_output, {epoch})[0]
    checkpoint = load_distillation_checkpoint(
        candidate.reference,
        candidate.identity,
        LocalArtifactStore(checkpoint_output / "artifacts"),
    )
    values = dict(checkpoint.state.parameter_values)
    if set(values) != set(selected):
        raise ValueError(f"checkpoint epoch {epoch} differs from the KD selector")
    with torch.no_grad():
        for name, parameter in selected.items():
            parameter.copy_(values[name].to(device=parameter.device, dtype=parameter.dtype))
    return {
        "epoch": epoch,
        "steps": candidate.steps,
        "reference": asdict(candidate.reference),
        "checkpoint_output": str(checkpoint_output.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", type=_parse_arm, action="append", required=True)
    parser.add_argument("--primary-baseline", required=True)
    parser.add_argument("--primary-candidate", required=True)
    parser.add_argument("--expected-steps", type=_parse_expected_steps, action="append", required=True)
    parser.add_argument("--slice-registry", type=Path, required=True)
    parser.add_argument("--slice-id", required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--wikitext-split", choices=("test", "validation"), default="validation")
    parser.add_argument("--wikitext-offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--minimum-primary-delta", type=float, default=0.02)
    parser.add_argument("--selected-mass-floor", type=float)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _sequence_metrics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int,
) -> SequenceMetrics:
    if (
        teacher_logits.shape != student_logits.shape
        or teacher_logits.ndim != 3
        or labels.shape != teacher_logits.shape[:2]
        or not 0 < top_k < teacher_logits.shape[-1]
    ):
        raise ValueError("WikiText KD metric geometry is invalid")
    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_p = teacher_logp.exp()
    indices = torch.topk(teacher_logits, top_k, dim=-1).indices
    teacher_top_p = teacher_p.gather(-1, indices)
    student_top_logp = student_logp.gather(-1, indices)
    student_top_p = student_top_logp.exp()
    teacher_mass = teacher_top_p.sum(dim=-1)
    student_mass = student_top_p.sum(dim=-1)
    teacher_tail = (1.0 - teacher_mass).clamp_min(1e-12)
    student_tail = (1.0 - student_mass).clamp_min(1e-12)
    conditional_teacher = teacher_top_p / teacher_mass.unsqueeze(-1)
    conditional_student_logp = torch.log_softmax(
        student_logits.float().gather(-1, indices), dim=-1
    )
    token_count = labels.numel()
    denominator = float(token_count)
    return SequenceMetrics(
        negative_log_likelihood=float(
            -student_logp.gather(-1, labels.unsqueeze(-1)).sum()
        )
        / denominator,
        full_kl=float((teacher_p * (teacher_logp - student_logp)).sum()) / denominator,
        conditional_topk_kl=float(
            (
                conditional_teacher
                * (conditional_teacher.clamp_min(1e-30).log() - conditional_student_logp)
            ).sum()
        )
        / denominator,
        topk_plus_tail_kl=float(
            (
                teacher_top_p
                * (teacher_logp.gather(-1, indices) - student_top_logp)
            ).sum()
            + (teacher_tail * (teacher_tail.log() - student_tail.log())).sum()
        )
        / denominator,
        teacher_topk_mass=float(teacher_mass.sum()) / denominator,
        student_teacher_topk_mass=float(student_mass.sum()) / denominator,
        tail_mass_absolute_error=float((student_tail - teacher_tail).abs().sum())
        / denominator,
        token_count=token_count,
    )


def _mean(sequences: tuple[SequenceMetrics, ...], attribute: str) -> float:
    if not sequences:
        raise ValueError("WikiText KD arm produced no sequence metrics")
    tokens = math.fsum(item.token_count for item in sequences)
    return math.fsum(
        float(getattr(item, attribute)) * item.token_count for item in sequences
    ) / tokens


def _aggregate(sequences: tuple[SequenceMetrics, ...]) -> dict[str, float | int]:
    attributes = tuple(
        name for name in SequenceMetrics.__dataclass_fields__ if name != "token_count"
    )
    result: dict[str, float | int] = {
        name: _mean(sequences, name) for name in attributes
    }
    result["perplexity"] = math.exp(float(result["negative_log_likelihood"]))
    result["token_count"] = sum(item.token_count for item in sequences)
    return result


def _paired_interval(
    baseline: tuple[SequenceMetrics, ...],
    candidate: tuple[SequenceMetrics, ...],
    attribute: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int | bool]:
    if len(baseline) != len(candidate) or not baseline or resamples <= 0:
        raise ValueError("paired WikiText KD comparison requires aligned sequences")

    def sampled_mean(values: tuple[SequenceMetrics, ...], indices: list[int]) -> float:
        tokens = math.fsum(values[index].token_count for index in indices)
        return math.fsum(
            float(getattr(values[index], attribute)) * values[index].token_count
            for index in indices
        ) / tokens

    generator = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        indices = [generator.randrange(len(baseline)) for _item in baseline]
        deltas.append(sampled_mean(candidate, indices) - sampled_mean(baseline, indices))
    deltas.sort()
    baseline_mean = _mean(baseline, attribute)
    point = _mean(candidate, attribute) - baseline_mean
    return {
        "point_delta": point,
        "relative_delta": point / baseline_mean,
        "lower_delta": deltas[int(0.025 * resamples)],
        "upper_delta": deltas[int(0.975 * resamples) - 1],
        "confidence": 0.95,
        "resamples": resamples,
        "improved_with_confidence": point < 0.0 and deltas[int(0.975 * resamples) - 1] < 0.0,
    }


def _bootstrap_interval(
    values: tuple[SequenceMetrics, ...],
    attribute: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if not values or resamples <= 0:
        raise ValueError("WikiText KD bootstrap requires sequence metrics")

    def sampled_mean(indices: list[int]) -> float:
        tokens = math.fsum(values[index].token_count for index in indices)
        return math.fsum(
            float(getattr(values[index], attribute)) * values[index].token_count
            for index in indices
        ) / tokens

    generator = random.Random(seed)
    samples = []
    for _ in range(resamples):
        indices = [generator.randrange(len(values)) for _item in values]
        samples.append(sampled_mean(indices))
    samples.sort()
    return {
        "point": _mean(values, attribute),
        "lower": samples[int(0.025 * resamples)],
        "upper": samples[int(0.975 * resamples) - 1],
        "confidence": 0.95,
        "resamples": resamples,
    }


def _slice_reservation(
    path: Path,
    slice_id: str,
    *,
    split: str,
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
    selected = [item for item in entries if item.get("id") == slice_id]
    if len(selected) != 1:
        raise ValueError("evaluation slice reservation is missing or ambiguous")
    reservation = selected[0]
    expected = {
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
        "split": split,
        "offset": offset,
        "samples": samples,
        "sequence_length": sequence_length,
        "token_hash": token_hash,
    }
    if any(reservation.get(key) != value for key, value in expected.items()):
        raise ValueError("evaluation slice reservation differs from the requested protocol")
    if reservation.get("status") != "reserved":
        raise ValueError("evaluation slice must be reserved before its gate is opened")
    payload = sequence_length - 1
    start = offset * payload
    end = (offset + samples) * payload
    for item in entries:
        if (
            item is reservation
            or item.get("dataset") != expected["dataset"]
            or item.get("split") != split
            or item.get("status") not in {"reserved", "retired"}
        ):
            continue
        other_payload = int(item["sequence_length"]) - 1
        other_start = int(item["offset"]) * other_payload
        other_end = (int(item["offset"]) + int(item["samples"])) * other_payload
        if max(start, other_start) < min(end, other_end):
            raise ValueError(f"evaluation slice overlaps reserved/retired slice {item['id']}")
    return cast(dict[str, object], reservation), "sha256:" + hashlib.sha256(encoded).hexdigest()


def _restore(payload: object) -> tuple[SequenceMetrics, ...]:
    if not isinstance(payload, list):
        raise ValueError("WikiText KD checkpoint arm is invalid")
    return tuple(SequenceMetrics(**item) for item in payload)


def run(args: argparse.Namespace) -> int:
    arms = tuple(args.arm)
    names = tuple(name for name, _mode, _path, _checkpoint, _epoch in arms)
    expected_steps = dict(args.expected_steps)
    if (
        len(arms) < 2
        or len(set(names)) != len(names)
        or args.primary_baseline not in names
        or args.primary_candidate not in names
        or args.primary_baseline == args.primary_candidate
        or min(
            args.samples,
            args.sequence_length - 1,
            args.top_k,
            args.bootstrap_resamples,
        )
        <= 0
        or args.wikitext_offset < 0
        or args.minimum_primary_delta < 0.0
        or set(expected_steps) != set(names)
        or len(expected_steps) != len(args.expected_steps)
        or (
            args.selected_mass_floor is not None
            and not 0.0 < args.selected_mass_floor < 1.0
        )
    ):
        raise ValueError("WikiText KD comparison protocol is invalid")
    all_tokens, fingerprint, bos_token_id = _split_tokens(
        args.snapshot,
        split=args.wikitext_split,
        samples=args.wikitext_offset + args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    tokens = all_tokens[args.wikitext_offset : args.wikitext_offset + args.samples]
    token_hash = _token_hash(tokens)
    reservation, registry_hash = _slice_reservation(
        args.slice_registry,
        args.slice_id,
        split=args.wikitext_split,
        offset=args.wikitext_offset,
        samples=args.samples,
        sequence_length=args.sequence_length,
        token_hash=token_hash,
    )
    protocol = {
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
        "wikitext_split": args.wikitext_split,
        "wikitext_offset": args.wikitext_offset,
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "top_k": args.top_k,
        "dataset_fingerprint": fingerprint,
        "bos_token_id": bos_token_id,
        "token_hash": token_hash,
        "model_revision": args.model_revision,
        "arms": [
            {
                "name": name,
                "mode": mode,
                "run_output": str(path.resolve()),
                "checkpoint_output": (
                    None if checkpoint is None else str(checkpoint.resolve())
                ),
                "epoch": epoch,
                "expected_steps": expected_steps[name],
            }
            for name, mode, path, checkpoint, epoch in arms
        ],
        "primary_baseline": args.primary_baseline,
        "primary_candidate": args.primary_candidate,
        "minimum_primary_delta": args.minimum_primary_delta,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "selected_mass_floor": args.selected_mass_floor,
        "slice_registry": str(args.slice_registry.resolve()),
        "slice_registry_sha256": registry_hash,
        "slice_reservation": reservation,
    }
    checkpoint_path = args.output.with_name(args.output.stem + ".checkpoint.json")
    if args.output.is_file():
        completed = json.loads(args.output.read_text(encoding="utf-8"))
        if completed.get("status") != "completed" or completed.get("protocol") != protocol:
            raise ValueError("existing WikiText KD output protocol differs")
        return 0
    checkpoint: dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "protocol": protocol,
        "sequences": {},
    }
    if checkpoint_path.is_file():
        restored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if restored.get("protocol") != protocol or not isinstance(restored.get("sequences"), dict):
            raise ValueError("WikiText KD checkpoint protocol differs")
        checkpoint = restored

    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    arm_sequences: dict[str, tuple[SequenceMetrics, ...]] = {}
    manifests: dict[str, object] = {}
    frozen_identity: dict[str, str] | None = None
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        cast(Any, teacher).config.use_cache = False
        teacher.eval()
        for name, mode, run_output, checkpoint_output, epoch in arms:
            global_tuning_override = None
            if mode == "tuning":
                assert checkpoint_output is not None
                global_tuning_override = from_dict(
                    ArtifactRef,
                    json.loads(checkpoint_output.read_text(encoding="utf-8")),
                    path=f"arm[{name}].global_tuning",
                )
            load_device = "cpu" if mode == "checkpoint" else args.device
            loaded = load_frozen_run(
                run_output,
                args.snapshot,
                source_name=MODEL_SOURCE,
                revision=args.model_revision,
                device=load_device,
                verify_hashes=False,
                backend="factorized",
                use_global_tuning=mode != "checkpoint",
                global_tuning_override=global_tuning_override,
            )
            checkpoint_receipt = None
            if mode == "checkpoint" and checkpoint_output is not None and epoch is not None:
                checkpoint_receipt = _apply_checkpoint(
                    loaded,
                    run_output,
                    checkpoint_output,
                    epoch,
                )
                loaded.model.to(args.device)
            if checkpoint_receipt is not None:
                observed_steps = int(checkpoint_receipt["steps"])
            else:
                if loaded.global_tuning is None:
                    raise ValueError(f"WikiText KD arm {name} has no global tuning")
                observed_steps = load_global_tuning(
                    loaded.global_tuning,
                    LocalArtifactStore(run_output / "artifacts"),
                ).result.steps_completed
            if observed_steps != expected_steps[name]:
                raise ValueError(
                    f"WikiText KD arm {name} has {observed_steps} steps; "
                    f"expected {expected_steps[name]}"
                )
            observed_identity = {
                "model_hash": loaded.identity.model_hash,
                "config_hash": loaded.identity.config_hash,
                "plan_hash": loaded.identity.plan_hash,
            }
            if frozen_identity is not None and frozen_identity != observed_identity:
                raise ValueError("WikiText KD arms have different frozen identities")
            frozen_identity = observed_identity
            manifests[name] = {
                "run_output": str(run_output.resolve()),
                "global_tuning": asdict(loaded.global_tuning) if loaded.global_tuning else None,
                "checkpoint": checkpoint_receipt,
                "global_tuning_pointer": (
                    str(checkpoint_output.resolve()) if mode == "tuning" else None
                ),
                "steps_completed": observed_steps,
            }
            completed = _restore(cast(dict[str, object], checkpoint["sequences"]).get(name, []))
            if len(completed) > len(tokens):
                raise ValueError("WikiText KD checkpoint exceeds token inventory")
            sequences = list(completed)
            loaded.model.eval()
            with torch.no_grad():
                for index in range(len(sequences), len(tokens)):
                    batch = tokens[index : index + 1].to(args.device)
                    teacher_logits = cast(HuggingFaceModel, teacher)(
                        input_ids=batch, use_cache=False
                    ).logits[:, :-1]
                    student_logits = cast(HuggingFaceModel, loaded.model)(
                        input_ids=batch, use_cache=False
                    ).logits[:, :-1]
                    sequences.append(
                        _sequence_metrics(
                            teacher_logits,
                            student_logits,
                            batch[:, 1:],
                            top_k=args.top_k,
                        )
                    )
                    cast(dict[str, object], checkpoint["sequences"])[name] = [
                        asdict(item) for item in sequences
                    ]
                    atomic_write_json(checkpoint_path, checkpoint)
                    print(f"{name}: {index + 1}/{len(tokens)} sequences", flush=True)
                    del batch, teacher_logits, student_logits
            arm_sequences[name] = tuple(sequences)
            del loaded
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del teacher

    attributes = (
        "negative_log_likelihood",
        "full_kl",
        "conditional_topk_kl",
        "topk_plus_tail_kl",
        "student_teacher_topk_mass",
        "tail_mass_absolute_error",
    )

    def comparison(before: str, after: str) -> dict[str, object]:
        intervals = {
            attribute: _paired_interval(
                arm_sequences[before],
                arm_sequences[after],
                attribute,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            )
            for attribute in attributes
        }
        nll = cast(dict[str, object], intervals["negative_log_likelihood"])
        full_kl = cast(dict[str, object], intervals["full_kl"])
        tail_kl = cast(dict[str, object], intervals["topk_plus_tail_kl"])
        return {
            "metrics": intervals,
            "passes_primary_gate": (
                cast(float, nll["point_delta"]) <= -args.minimum_primary_delta
                and cast(float, nll["upper_delta"]) < 0.0
                and cast(float, full_kl["point_delta"])
                <= -args.minimum_primary_delta
                and cast(float, full_kl["upper_delta"]) < 0.0
                and cast(float, tail_kl["point_delta"]) < 0.0
            ),
        }

    adjacent = {
        f"{after}_minus_{before}": comparison(before, after)
        for before, after in zip(names, names[1:], strict=False)
    }
    primary = comparison(args.primary_baseline, args.primary_candidate)
    arm_results = {}
    for name, sequences in arm_sequences.items():
        intervals = {
            attribute: _bootstrap_interval(
                sequences,
                attribute,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            )
            for attribute in attributes
        }
        mass_interval = intervals["student_teacher_topk_mass"]
        arm_results[name] = {
            "means": _aggregate(sequences),
            "confidence_intervals": intervals,
            "selected_mass_floor_gate": (
                None
                if args.selected_mass_floor is None
                else {
                    "floor": args.selected_mass_floor,
                    "passes": float(mass_interval["lower"]) >= args.selected_mass_floor,
                }
            ),
        }
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "held-out WikiText KD horizon and objective comparison",
            "protocol": protocol,
            "frozen_identity": frozen_identity,
            "arms": manifests,
            "results": arm_results,
            "paired_adjacent_arms": adjacent,
            "primary_comparison": {
                "candidate_minus_baseline": (
                    f"{args.primary_candidate}_minus_{args.primary_baseline}"
                ),
                **primary,
            },
        },
    )
    checkpoint["status"] = "completed"
    atomic_write_json(checkpoint_path, checkpoint)
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
