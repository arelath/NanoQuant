"""Compare retained KD checkpoints on a pinned non-WikiText C4 slice."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from datasets import Dataset, load_dataset  # type: ignore[import-untyped]
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import _paired_payload
from probe_factorized_component_overlays_kl import _dtype
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION
from probe_wikitext_kd_quality import _apply_checkpoint
from torch import nn
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    KlSequenceResult,
    causal_kl_nll_per_sequence_from_logits,
)
from nanoquant.application.temperature_calibration import (
    TemperatureFitResult,
    TemperatureSequenceMetrics,
    causal_raw_fitted_temperature_metrics,
)
from nanoquant.config.codec import from_dict, semantic_hash, to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.global_tuning import load_global_tuning
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_model_protocol import HuggingFaceModel
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.temperature_fit_checkpoint import load_temperature_fit_receipt
from nanoquant.kl_budget_workflow import _token_hash

C4_DATASET = "allenai/c4"
C4_REVISION = "f998d2cd8b92435980789e3ecb2f89b4c68bfe1e"
C4_VALIDATION_FILE = "en/c4-validation.00000-of-00008.json.gz"
C4_VALIDATION_FILE_SHA256 = "bc35d7c1b1d14b90cd3a394cccbcbe191935edd04bf42ee965379c6e2987a5f0"


def _parse_arm(value: str) -> tuple[str, str, Path, Path | None, int | None, str]:
    name, separator, specification = value.partition("=")
    parts = specification.split(";")
    mode = parts[0] if parts else ""
    if (
        not separator
        or not name.strip()
        or mode not in {"prekd", "postkd", "tuning", "checkpoint"}
        or (mode in {"prekd", "postkd"} and len(parts) != 2)
        or (mode == "tuning" and len(parts) != 3)
        or (mode == "checkpoint" and len(parts) not in {4, 5})
        or not parts[1].strip()
        or (mode == "tuning" and not parts[2].strip())
        or (mode == "checkpoint" and not parts[2].strip())
    ):
        raise argparse.ArgumentTypeError(
            "arm must use name=prekd;run-output, name=postkd;run-output, or "
            "name=tuning;run-output;artifact-reference-json, or "
            "name=checkpoint;run-output;checkpoint-output;epoch[;state-namespace]"
        )
    epoch = None
    if mode == "checkpoint":
        try:
            epoch = int(parts[3])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("checkpoint epoch must be an integer") from exc
        if epoch <= 0:
            raise argparse.ArgumentTypeError("checkpoint epoch must be positive")
    state_namespace = (
        parts[4].strip()
        if mode == "checkpoint" and len(parts) == 5
        else "global-distillation"
    )
    if not state_namespace or Path(state_namespace).name != state_namespace:
        raise argparse.ArgumentTypeError("checkpoint state namespace must be a safe filename stem")
    return (
        name.strip(),
        mode,
        Path(parts[1].strip()),
        None if mode not in {"tuning", "checkpoint"} else Path(parts[2].strip()),
        epoch,
        state_namespace,
    )


def _parse_expected_steps(value: str) -> tuple[str, int]:
    name, separator, steps = value.partition("=")
    try:
        parsed = int(steps)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected steps must use arm=nonnegative-integer") from exc
    if not separator or not name.strip() or parsed < 0:
        raise argparse.ArgumentTypeError("expected steps must use arm=nonnegative-integer")
    return name.strip(), parsed


def _parse_temperature_receipt(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("temperature fit receipt must use arm=path")
    return name.strip(), Path(path.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", type=_parse_arm, action="append", required=True)
    parser.add_argument("--primary-baseline", required=True)
    parser.add_argument("--primary-candidate", required=True)
    parser.add_argument(
        "--reference-arm",
        action="append",
        default=[],
        help=(
            "arm whose frozen factorization may differ from the primary same-run pair; "
            "reported only as an absolute reference and never used for promotion"
        ),
    )
    parser.add_argument("--expected-steps", type=_parse_expected_steps, action="append", required=True)
    parser.add_argument("--slice-registry", type=Path, required=True)
    parser.add_argument("--slice-id", required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--c4-revision", default=C4_REVISION)
    parser.add_argument("--c4-file", default=C4_VALIDATION_FILE)
    parser.add_argument("--c4-documents", type=int, default=1_100)
    parser.add_argument("--offset", type=int, default=104)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--temperature-fit-receipt",
        type=_parse_temperature_receipt,
        action="append",
        default=[],
        help="opt-in final reporting with one frozen arm=receipt fit for each primary arm",
    )
    parser.add_argument("--temperature-top-k", type=int, default=64)
    return parser


def _contiguous_token_windows(
    texts: Sequence[str],
    tokenizer: Any,
    *,
    offset: int,
    samples: int,
    sequence_length: int,
) -> tuple[torch.Tensor, int | None]:
    if offset < 0 or samples <= 0 or sequence_length <= 1 or not texts:
        raise ValueError("C4 token window protocol is invalid")
    encoded = tokenizer(
        " ".join(texts),
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("C4 tokenizer output must contain one contiguous token stream")
    required = (offset + samples) * sequence_length
    if input_ids.shape[1] < required:
        raise ValueError("C4 token stream is shorter than the requested held-out window")
    windows = input_ids[:, :required].reshape(offset + samples, sequence_length)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    return windows[offset : offset + samples].long().contiguous(), bos_token_id


def _load_c4_tokens(
    snapshot: Path,
    *,
    revision: str,
    data_file: str,
    documents: int,
    offset: int,
    samples: int,
    sequence_length: int,
    local_files_only: bool,
) -> tuple[torch.Tensor, str, int | None]:
    local_data_file = Path(data_file)
    if local_data_file.is_file() and local_data_file.suffix == ".arrow":
        dataset = Dataset.from_file(str(local_data_file.resolve()))
    elif local_data_file.is_file():
        dataset = load_dataset(
            "json",
            data_files={"validation": str(local_data_file.resolve())},
            split="validation",
        )
    else:
        dataset = load_dataset(
            C4_DATASET,
            data_files={"validation": data_file},
            split="validation",
            revision=revision,
        )
    if documents <= 0 or len(dataset) < documents:
        raise ValueError("C4 document inventory is shorter than the pinned protocol")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=local_files_only,
    )
    texts = dataset[:documents]["text"]
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise ValueError("C4 validation shard has an invalid text column")
    tokens, bos_token_id = _contiguous_token_windows(
        texts,
        tokenizer,
        offset=offset,
        samples=samples,
        sequence_length=sequence_length,
    )
    return tokens, str(dataset._fingerprint), bos_token_id


def _arm_result(name: str, sequences: tuple[KlSequenceResult, ...]) -> KlBudgetArmResult:
    if not sequences:
        raise ValueError("non-WikiText arm produced no sequence metrics")
    tokens = math.fsum(item.token_count for item in sequences)
    agreements = tuple(item.teacher_top1_agreement for item in sequences)
    available_agreements = tuple(value for value in agreements if value is not None)
    top1_agreement = (
        math.fsum(
            value * item.token_count
            for value, item in zip(available_agreements, sequences, strict=True)
        )
        / tokens
        if len(available_agreements) == len(agreements)
        else None
    )
    return KlBudgetArmResult(
        name,
        math.fsum(item.negative_log_likelihood * item.token_count for item in sequences)
        / tokens,
        math.fsum(item.kl_nats_per_token * item.token_count for item in sequences) / tokens,
        int(tokens),
        sequences=sequences,
        teacher_top1_agreement=top1_agreement,
    )


def _sequence_from_checkpoint(item: dict[str, object]) -> KlSequenceResult:
    agreement = item.get("teacher_top1_agreement")
    return KlSequenceResult(
        float(cast(Any, item["negative_log_likelihood"])),
        float(cast(Any, item["kl_nats_per_token"])),
        int(cast(Any, item["token_count"])),
        None if agreement is None else float(cast(Any, agreement)),
    )


def _temperature_sequence_from_checkpoint(item: dict[str, object]) -> TemperatureSequenceMetrics:
    return from_dict(TemperatureSequenceMetrics, item, path="temperature_sequence")


@torch.no_grad()
def _evaluate_arm(
    name: str,
    teacher: nn.Module,
    student: nn.Module,
    tokens: torch.Tensor,
    *,
    device: str,
    token_chunk_size: int,
    completed: tuple[KlSequenceResult, ...] = (),
    progress: Callable[[tuple[KlSequenceResult, ...]], None] | None = None,
) -> KlBudgetArmResult:
    if len(completed) > tokens.shape[0]:
        raise ValueError("non-WikiText checkpoint exceeds the token inventory")
    teacher.eval()
    student.eval()
    sequences = list(completed)
    for index in range(len(completed), tokens.shape[0]):
        batch = tokens[index : index + 1].to(device)
        teacher_logits = cast(HuggingFaceModel, teacher)(input_ids=batch, use_cache=False).logits
        student_logits = cast(HuggingFaceModel, student)(input_ids=batch, use_cache=False).logits
        sequences.extend(
            causal_kl_nll_per_sequence_from_logits(
                teacher_logits,
                student_logits,
                batch,
                token_chunk_size=token_chunk_size,
            )
        )
        if progress is not None:
            progress(tuple(sequences))
        print(f"{name}: {index + 1}/{tokens.shape[0]} sequences", flush=True)
        del batch, teacher_logits, student_logits
    return _arm_result(name, tuple(sequences))


@torch.no_grad()
def _evaluate_temperature_arm(
    name: str,
    teacher: nn.Module,
    student: nn.Module,
    tokens: torch.Tensor,
    *,
    logit_scale: float,
    top_k: int,
    device: str,
    token_chunk_size: int,
    completed: tuple[TemperatureSequenceMetrics, ...] = (),
    progress: Callable[[tuple[TemperatureSequenceMetrics, ...]], None] | None = None,
) -> tuple[TemperatureSequenceMetrics, ...]:
    if len(completed) > tokens.shape[0]:
        raise ValueError("temperature-report checkpoint exceeds the token inventory")
    teacher.eval()
    student.eval()
    sequences = list(completed)
    for index in range(len(completed), tokens.shape[0]):
        batch = tokens[index : index + 1].to(device)
        teacher_logits = cast(HuggingFaceModel, teacher)(input_ids=batch, use_cache=False).logits
        student_logits = cast(HuggingFaceModel, student)(input_ids=batch, use_cache=False).logits
        sequences.extend(
            causal_raw_fitted_temperature_metrics(
                teacher_logits,
                student_logits,
                batch,
                logit_scale=logit_scale,
                top_k=top_k,
                token_chunk_size=token_chunk_size,
            )
        )
        if progress is not None:
            progress(tuple(sequences))
        print(f"{name} raw/fitted: {index + 1}/{tokens.shape[0]} sequences", flush=True)
        del batch, teacher_logits, student_logits
    return tuple(sequences)


def _temperature_mean(
    sequences: tuple[TemperatureSequenceMetrics, ...],
    attribute: str,
) -> float:
    if not sequences:
        raise ValueError("temperature report has no sequence metrics")
    tokens = math.fsum(item.token_count for item in sequences)
    return math.fsum(
        float(getattr(item, attribute)) * item.token_count for item in sequences
    ) / tokens


def _kl_result_from_temperature(
    name: str,
    sequences: tuple[TemperatureSequenceMetrics, ...],
    *,
    fitted: bool,
) -> KlBudgetArmResult:
    converted = tuple(
        KlSequenceResult(
            (
                item.fitted_negative_log_likelihood
                if fitted
                else item.raw_negative_log_likelihood
            ),
            item.fitted_kl_nats_per_token if fitted else item.raw_kl_nats_per_token,
            item.token_count,
            item.teacher_top1_agreement,
        )
        for item in sequences
    )
    return _arm_result(name, converted)


def _temperature_bootstrap(
    sequences: tuple[TemperatureSequenceMetrics, ...],
    attribute: str,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    generator = random.Random(seed)
    sampled = []
    for _sample in range(resamples):
        selected = tuple(sequences[generator.randrange(len(sequences))] for _ in sequences)
        sampled.append(_temperature_mean(selected, attribute))
    sampled.sort()
    return {
        "mean": _temperature_mean(sequences, attribute),
        "lower": sampled[int(0.025 * resamples)],
        "upper": sampled[int(0.975 * resamples) - 1],
        "confidence": 0.95,
        "resamples": resamples,
    }


def _paired_temperature_metric(
    baseline: tuple[TemperatureSequenceMetrics, ...],
    candidate: tuple[TemperatureSequenceMetrics, ...],
    attribute: str,
    *,
    resamples: int = 10_000,
    seed: int = 0,
    higher_is_better: bool = False,
) -> dict[str, object]:
    if not baseline or len(baseline) != len(candidate) or tuple(
        item.token_count for item in baseline
    ) != tuple(item.token_count for item in candidate):
        raise ValueError("paired temperature report requires aligned sequences")
    generator = random.Random(seed)
    deltas = []
    for _sample in range(resamples):
        indices = [generator.randrange(len(baseline)) for _ in baseline]
        before = tuple(baseline[index] for index in indices)
        after = tuple(candidate[index] for index in indices)
        deltas.append(
            _temperature_mean(after, attribute) - _temperature_mean(before, attribute)
        )
    deltas.sort()
    before_mean = _temperature_mean(baseline, attribute)
    point = _temperature_mean(candidate, attribute) - before_mean
    lower = deltas[int(0.025 * resamples)]
    upper = deltas[int(0.975 * resamples) - 1]
    return {
        "point_delta": point,
        "relative_delta": None if before_mean == 0 else point / before_mean,
        "lower_delta": lower,
        "upper_delta": upper,
        "confidence": 0.95,
        "resamples": resamples,
        "improved_with_confidence": (
            point > 0 and lower > 0 if higher_is_better else point < 0 and upper < 0
        ),
    }


def _load_temperature_receipts(
    entries: tuple[tuple[str, Path], ...],
    *,
    baseline: str,
    candidate: str,
    current_token_hash: str,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, TemperatureFitResult],
    dict[str, dict[str, object]],
]:
    if len(entries) != 2 or {name for name, _path in entries} != {baseline, candidate}:
        raise ValueError("temperature reporting requires exactly the primary baseline and candidate fits")
    protocols: dict[str, dict[str, object]] = {}
    results: dict[str, TemperatureFitResult] = {}
    summaries: dict[str, dict[str, object]] = {}
    for name, path in entries:
        protocol, result = load_temperature_fit_receipt(path)
        arm = protocol.get("arm")
        selection = protocol.get("selection")
        fit_slice = protocol.get("slice")
        if (
            not isinstance(arm, dict)
            or arm.get("name") != name
            or not isinstance(selection, dict)
            or not isinstance(fit_slice, dict)
            or fit_slice.get("token_hash") == current_token_hash
        ):
            raise ValueError("temperature fit receipt identity or data role is invalid")
        expected_role = "baseline" if name == baseline else "selected"
        if selection.get("role") != expected_role or selection.get("selected_arm") != candidate:
            raise ValueError("temperature fit receipt role differs from the final comparison")
        protocols[name] = protocol
        results[name] = result
        summaries[name] = {
            "path": str(path.resolve()),
            "sha256": "sha256:" + hash_file(path),
            "protocol_hash": semantic_hash(protocol),
            "logit_scale": result.final_logit_scale,
            "equivalent_temperature": result.equivalent_temperature,
        }
    shared_keys = ("solver", "dataset", "slice", "model")
    baseline_selection = cast(dict[str, object], protocols[baseline]["selection"])
    candidate_selection = cast(dict[str, object], protocols[candidate]["selection"])
    if (
        any(
            protocols[baseline].get(key) != protocols[candidate].get(key)
            for key in shared_keys
        )
        or {key: value for key, value in baseline_selection.items() if key != "role"}
        != {key: value for key, value in candidate_selection.items() if key != "role"}
    ):
        raise ValueError("temperature fit receipts do not share one fitting protocol and slice")
    return protocols, results, summaries


def _comparison(
    baseline: KlBudgetArmResult,
    candidate: KlBudgetArmResult,
) -> dict[str, object]:
    nll = _paired_metric_payload(baseline, candidate, "negative_log_likelihood")
    kl = _paired_payload(baseline, candidate, seed=0)
    top1 = (
        _paired_metric_payload(
            baseline,
            candidate,
            "teacher_top1_agreement",
            higher_is_better=True,
        )
        if baseline.teacher_top1_agreement is not None
        and candidate.teacher_top1_agreement is not None
        else None
    )
    return {
        "nll": nll,
        "kl": kl,
        "teacher_top1_agreement": top1,
        "passes": bool(nll["improved_with_confidence"])
        and float(kl["upper_delta"]) <= 0.0,
    }


def _c4_slice_reservation(
    path: Path,
    slice_id: str,
    *,
    offset: int,
    samples: int,
    sequence_length: int,
    token_hash: str,
    allowed_statuses: tuple[str, ...] = ("retired",),
) -> tuple[dict[str, object], str]:
    encoded = path.read_bytes()
    registry = json.loads(encoded)
    entries = registry.get("slices")
    if registry.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError("evaluation slice registry is invalid")
    selected = [item for item in entries if item.get("id") == slice_id]
    if len(selected) != 1:
        raise ValueError("C4 evaluation slice reservation is missing or ambiguous")
    reservation = selected[0]
    if not allowed_statuses or any(status not in {"reserved", "retired"} for status in allowed_statuses):
        raise ValueError("C4 evaluation slice allowed statuses are invalid")
    expected = {
        "dataset": C4_DATASET,
        "split": "validation",
        "offset": offset,
        "samples": samples,
        "sequence_length": sequence_length,
        "token_start": offset * sequence_length,
        "token_end": (offset + samples) * sequence_length,
        "token_hash": token_hash,
    }
    if any(reservation.get(key) != value for key, value in expected.items()) or reservation.get(
        "status"
    ) not in allowed_statuses:
        raise ValueError("C4 evaluation slice reservation differs from the requested protocol")
    start = offset * sequence_length
    end = (offset + samples) * sequence_length
    for item in entries:
        if (
            item is reservation
            or item.get("dataset") != C4_DATASET
            or item.get("split") != "validation"
            or item.get("status") not in {"reserved", "retired"}
        ):
            continue
        other_start = int(item["token_start"])
        other_end = int(item["token_end"])
        if max(start, other_start) < min(end, other_end):
            raise ValueError(f"C4 evaluation slice overlaps reserved/retired slice {item['id']}")
    return cast(dict[str, object], reservation), "sha256:" + hashlib.sha256(encoded).hexdigest()


def _primary_frozen_identity(
    name: str,
    observed: dict[str, object],
    *,
    reference_arms: frozenset[str],
    current: dict[str, object] | None,
) -> dict[str, object] | None:
    if name in reference_arms:
        return current
    if current is not None and observed != current:
        raise ValueError("non-WikiText primary arms have different frozen identities")
    return observed


def run(args: argparse.Namespace) -> int:
    arms = tuple(args.arm)
    names = tuple(
        name for name, _mode, _path, _pointer, _epoch, _namespace in arms
    )
    expected_steps = dict(args.expected_steps)
    reference_arms = frozenset(args.reference_arm)
    temperature_receipt_items = tuple(args.temperature_fit_receipt)
    if (
        len(arms) < 2
        or len(set(names)) != len(names)
        or args.primary_baseline not in names
        or args.primary_candidate not in names
        or args.primary_baseline == args.primary_candidate
        or not reference_arms.issubset(names)
        or len(reference_arms) != len(args.reference_arm)
        or args.primary_baseline in reference_arms
        or args.primary_candidate in reference_arms
        or min(args.c4_documents, args.samples, args.sequence_length - 1, args.token_chunk_size)
        <= 0
        or args.offset < 0
        or set(expected_steps) != set(names)
        or len(expected_steps) != len(args.expected_steps)
        or args.temperature_top_k <= 0
    ):
        raise ValueError("non-WikiText KD protocol is invalid")
    tokens, fingerprint, bos_token_id = _load_c4_tokens(
        args.snapshot,
        revision=args.c4_revision,
        data_file=args.c4_file,
        documents=args.c4_documents,
        offset=args.offset,
        samples=args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    token_hash = _token_hash(tokens)
    temperature_protocols: dict[str, dict[str, object]] = {}
    temperature_fits: dict[str, TemperatureFitResult] = {}
    temperature_receipts: dict[str, dict[str, object]] = {}
    if temperature_receipt_items:
        temperature_protocols, temperature_fits, temperature_receipts = (
            _load_temperature_receipts(
                temperature_receipt_items,
                baseline=args.primary_baseline,
                candidate=args.primary_candidate,
                current_token_hash=token_hash,
            )
        )
    reservation, registry_hash = _c4_slice_reservation(
        args.slice_registry,
        args.slice_id,
        offset=args.offset,
        samples=args.samples,
        sequence_length=args.sequence_length,
        token_hash=token_hash,
    )
    protocol = {
        "dataset": C4_DATASET,
        "dataset_revision": args.c4_revision,
        "data_file": args.c4_file,
        "data_file_sha256": (
            C4_VALIDATION_FILE_SHA256
            if args.c4_revision == C4_REVISION and args.c4_file == C4_VALIDATION_FILE
            else None
        ),
        "dataset_fingerprint": fingerprint,
        "documents": args.c4_documents,
        "document_join": "single-space",
        "tokenizer_policy": "source-model-add-special-tokens",
        "bos_token_id": bos_token_id,
        "offset": args.offset,
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "token_chunk_size": args.token_chunk_size,
        "token_hash": token_hash,
        "model_revision": args.model_revision,
        "arms": [
            {
                "name": name,
                "mode": mode,
                "run_output": str(path.resolve()),
                "global_tuning_pointer": (
                    str(pointer.resolve()) if mode == "tuning" else None
                ),
                "expected_steps": expected_steps[name],
                **(
                    {
                        "checkpoint_output": str(cast(Path, pointer).resolve()),
                        "epoch": epoch,
                        "checkpoint_state_namespace": namespace,
                    }
                    if mode == "checkpoint"
                    else {}
                ),
            }
            for name, mode, path, pointer, epoch, namespace in arms
        ],
        "primary_baseline": args.primary_baseline,
        "primary_candidate": args.primary_candidate,
        "reference_arms": tuple(args.reference_arm),
        "slice_registry": str(args.slice_registry.resolve()),
        "slice_registry_sha256": registry_hash,
        "slice_reservation": reservation,
    }
    if temperature_fits:
        protocol["temperature_reporting"] = {
            "mode": "apply_identity_bound_per_arm_logit_scales",
            "protocol_document": "Docs/82-temperature-calibration-reporting-protocol.md",
            "fit_receipts": temperature_receipts,
            "top_k": args.temperature_top_k,
            "raw_metrics_remain_primary": True,
        }
    checkpoint_path = args.output.with_name(args.output.stem + ".checkpoint.json")
    if args.output.is_file():
        completed = json.loads(args.output.read_text(encoding="utf-8"))
        if completed.get("status") != "completed" or completed.get("protocol") != protocol:
            raise ValueError("existing non-WikiText output protocol differs")
        return 0
    checkpoint: dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "protocol": protocol,
        "sequences": {},
    }
    if temperature_fits:
        checkpoint["temperature_sequences"] = {}
    if checkpoint_path.is_file():
        restored_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            restored_checkpoint.get("schema_version") != 1
            or restored_checkpoint.get("protocol") != protocol
            or not isinstance(restored_checkpoint.get("sequences"), dict)
            or (
                temperature_fits
                and not isinstance(
                    restored_checkpoint.get("temperature_sequences"),
                    dict,
                )
            )
        ):
            raise ValueError("non-WikiText checkpoint protocol differs")
        checkpoint = restored_checkpoint

    def restored(name: str) -> tuple[KlSequenceResult, ...]:
        payload = cast(dict[str, object], checkpoint["sequences"]).get(name, [])
        if not isinstance(payload, list):
            raise ValueError("non-WikiText checkpoint arm is invalid")
        return tuple(_sequence_from_checkpoint(item) for item in payload)

    def restored_temperature(name: str) -> tuple[TemperatureSequenceMetrics, ...]:
        payload = cast(dict[str, object], checkpoint["temperature_sequences"]).get(name, [])
        if not isinstance(payload, list):
            raise ValueError("non-WikiText temperature checkpoint arm is invalid")
        sequences = tuple(_temperature_sequence_from_checkpoint(item) for item in payload)
        raw_payload = cast(dict[str, object], checkpoint["sequences"]).get(name, [])
        expected_raw = (
            []
            if not sequences
            else [
                to_dict(item)
                for item in _kl_result_from_temperature(
                    name,
                    sequences,
                    fitted=False,
                ).sequences
            ]
        )
        if raw_payload not in ([], expected_raw):
            raise ValueError("raw and temperature checkpoint sequences differ")
        return sequences

    def save_progress(name: str, sequences: tuple[KlSequenceResult, ...]) -> None:
        cast(dict[str, object], checkpoint["sequences"])[name] = [
            to_dict(item) for item in sequences
        ]
        atomic_write_json(checkpoint_path, checkpoint)

    def save_temperature_progress(
        name: str, sequences: tuple[TemperatureSequenceMetrics, ...]
    ) -> None:
        cast(dict[str, object], checkpoint["temperature_sequences"])[name] = [
            to_dict(item) for item in sequences
        ]
        cast(dict[str, object], checkpoint["sequences"])[name] = [
            to_dict(item)
            for item in _kl_result_from_temperature(name, sequences, fitted=False).sequences
        ]
        atomic_write_json(checkpoint_path, checkpoint)

    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    results: dict[str, KlBudgetArmResult] = {}
    temperature_sequences: dict[str, tuple[TemperatureSequenceMetrics, ...]] = {}
    manifests: dict[str, dict[str, object]] = {}
    frozen_identity: dict[str, object] | None = None
    frozen_identities: dict[str, dict[str, object]] = {}
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        cast(Any, teacher).config.use_cache = False
        for name, mode, run_output, tuning_pointer, epoch, namespace in arms:
            global_tuning_override = None
            if mode == "tuning" and tuning_pointer is not None:
                global_tuning_override = from_dict(
                    ArtifactRef,
                    json.loads(tuning_pointer.read_text(encoding="utf-8")),
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
                use_global_tuning=mode not in {"prekd", "checkpoint"},
                global_tuning_override=global_tuning_override,
            )
            checkpoint_receipt = None
            if mode == "checkpoint" and tuning_pointer is not None and epoch is not None:
                checkpoint_receipt = _apply_checkpoint(
                    loaded,
                    run_output,
                    tuning_pointer,
                    epoch,
                    namespace,
                )
                loaded.model.to(args.device)
            observed_identity: dict[str, object] = {
                "model_hash": loaded.identity.model_hash,
                "config_hash": loaded.identity.config_hash,
                "plan_hash": loaded.identity.plan_hash,
            }
            frozen_identity = _primary_frozen_identity(
                name,
                observed_identity,
                reference_arms=reference_arms,
                current=frozen_identity,
            )
            frozen_identities[name] = observed_identity
            if loaded.global_tuning is None and mode not in {"prekd", "checkpoint"}:
                raise ValueError(f"non-WikiText arm {name} has no global tuning")
            if checkpoint_receipt is not None:
                observed_steps = int(cast(Any, checkpoint_receipt["steps"]))
            elif loaded.global_tuning is None and mode == "prekd":
                observed_steps = 0
            else:
                observed_steps = load_global_tuning(
                    cast(ArtifactRef, loaded.global_tuning),
                    LocalArtifactStore(run_output / "artifacts"),
                ).result.steps_completed
            if observed_steps != expected_steps[name]:
                raise ValueError(
                    f"non-WikiText arm {name} has {observed_steps} steps; "
                    f"expected {expected_steps[name]}"
                )
            manifests[name] = {
                "mode": mode,
                "run_output": str(run_output.resolve()),
                "global_tuning": (
                    None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
                ),
                "global_tuning_pointer": (
                    str(tuning_pointer.resolve()) if mode == "tuning" else None
                ),
                "checkpoint": checkpoint_receipt,
                "checkpoint_state_namespace": (
                    namespace if mode == "checkpoint" else None
                ),
                "steps_completed": observed_steps,
            }

            if name in temperature_fits:
                fit_protocol = temperature_protocols[name]
                if fit_protocol.get("arm") != {"name": name, **manifests[name]}:
                    raise ValueError(
                        f"temperature-fit receipt for {name} differs from the loaded arm"
                    )
                fit_model = fit_protocol.get("model")
                if (
                    not isinstance(fit_model, dict)
                    or fit_model.get("revision") != args.model_revision
                    or fit_model.get("frozen_identity") != observed_identity
                ):
                    raise ValueError(
                        f"temperature-fit receipt for {name} differs from the loaded model"
                    )

                def save_arm_temperature_progress(
                    sequences: tuple[TemperatureSequenceMetrics, ...],
                    arm: str = name,
                ) -> None:
                    save_temperature_progress(arm, sequences)

                temperature_sequences[name] = _evaluate_temperature_arm(
                    name,
                    teacher,
                    loaded.model,
                    tokens,
                    logit_scale=temperature_fits[name].final_logit_scale,
                    top_k=args.temperature_top_k,
                    device=args.device,
                    token_chunk_size=args.token_chunk_size,
                    completed=restored_temperature(name),
                    progress=save_arm_temperature_progress,
                )
                results[name] = _kl_result_from_temperature(
                    name,
                    temperature_sequences[name],
                    fitted=False,
                )
            else:
                def save_arm_progress(
                    sequences: tuple[KlSequenceResult, ...],
                    arm: str = name,
                ) -> None:
                    save_progress(arm, sequences)

                results[name] = _evaluate_arm(
                    name,
                    teacher,
                    loaded.model,
                    tokens,
                    device=args.device,
                    token_chunk_size=args.token_chunk_size,
                    completed=restored(name),
                    progress=save_arm_progress,
                )
            del loaded
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del teacher

    adjacent = {
        f"{after}_minus_{before}": _comparison(results[before], results[after])
        for before, after in zip(names, names[1:], strict=False)
    }
    primary = _comparison(
        results[args.primary_baseline],
        results[args.primary_candidate],
    )
    absolute_candidate_comparisons = {
        reference: {
            "candidate_minus_reference": f"{args.primary_candidate}_minus_{reference}",
            **_comparison(results[reference], results[args.primary_candidate]),
        }
        for reference in (args.primary_baseline, *args.reference_arm)
    }
    temperature_report = None
    if temperature_fits:
        baseline_temperature = temperature_sequences[args.primary_baseline]
        candidate_temperature = temperature_sequences[args.primary_candidate]
        result_attributes = (
            "raw_negative_log_likelihood",
            "raw_kl_nats_per_token",
            "fitted_negative_log_likelihood",
            "fitted_kl_nats_per_token",
            "teacher_top1_agreement",
            "teacher_topk_mass",
            "raw_student_teacher_topk_mass",
            "fitted_student_teacher_topk_mass",
        )
        paired_attributes = tuple(
            attribute
            for attribute in result_attributes
            if attribute != "teacher_topk_mass"
        )
        paired_temperature = {
            attribute: _paired_temperature_metric(
                baseline_temperature,
                candidate_temperature,
                attribute,
                higher_is_better=attribute
                in {
                    "teacher_top1_agreement",
                    "raw_student_teacher_topk_mass",
                    "fitted_student_teacher_topk_mass",
                },
            )
            for attribute in paired_attributes
        }
        raw_nll_delta = cast(
            float,
            paired_temperature["raw_negative_log_likelihood"]["point_delta"],
        )
        fitted_nll_delta = cast(
            float,
            paired_temperature["fitted_negative_log_likelihood"]["point_delta"],
        )
        raw_kl_delta = cast(
            float,
            paired_temperature["raw_kl_nats_per_token"]["point_delta"],
        )
        fitted_kl_delta = cast(
            float,
            paired_temperature["fitted_kl_nats_per_token"]["point_delta"],
        )
        temperature_report = {
            "role": "calibration diagnostic; cannot override the raw primary gate",
            "raw_primary_gate_passes": primary["passes"],
            "fitted_metrics_are_gating": False,
            "fit_receipts": temperature_receipts,
            "results": {
                name: {
                    attribute: _temperature_bootstrap(sequences, attribute)
                    for attribute in result_attributes
                }
                for name, sequences in temperature_sequences.items()
            },
            "primary_paired_comparison": {
                "candidate_minus_baseline": (
                    f"{args.primary_candidate}_minus_{args.primary_baseline}"
                ),
                **paired_temperature,
                "calibration_removed_from_raw_marginal": {
                    "negative_log_likelihood": {
                        "raw_point_delta": raw_nll_delta,
                        "fitted_point_delta": fitted_nll_delta,
                        "removed_fraction": (
                            None
                            if raw_nll_delta == 0
                            else (raw_nll_delta - fitted_nll_delta) / raw_nll_delta
                        ),
                    },
                    "kl_nats_per_token": {
                        "raw_point_delta": raw_kl_delta,
                        "fitted_point_delta": fitted_kl_delta,
                        "removed_fraction": (
                            None
                            if raw_kl_delta == 0
                            else (raw_kl_delta - fitted_kl_delta) / raw_kl_delta
                        ),
                    },
                },
            },
        }
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only pinned non-WikiText KD quality gate",
            "protocol": protocol,
            "frozen_identity": frozen_identity,
            "frozen_identities": frozen_identities,
            "arms": manifests,
            "results": {name: to_dict(result) for name, result in results.items()},
            "paired_adjacent_arms": adjacent,
            "primary_comparison": {
                "candidate_minus_baseline": (
                    f"{args.primary_candidate}_minus_{args.primary_baseline}"
                ),
                **primary,
            },
            "absolute_candidate_comparisons": absolute_candidate_comparisons,
            **(
                {}
                if temperature_report is None
                else {"temperature_reporting": temperature_report}
            ),
        },
    )
    checkpoint["status"] = "completed"
    atomic_write_json(checkpoint_path, checkpoint)
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
