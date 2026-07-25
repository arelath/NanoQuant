"""Pinned Hugging Face calibration mixture matching legacy Gemma Experiment 018."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.config.codec import semantic_hash, to_dict
from nanoquant.config.schema import BehaviorSliceConfig, DatasetConfig, ReasoningMode
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.chat_behaviors import chat_behavior_for_snapshot
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.safetensors_io import load_tensors
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.ports.chat_behavior import ReasoningModeId, RenderedBehaviorRecord, TokenRole, tensor_sha256

ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CALIBRATION_RECEIPT_NAME = "calibration-input.json"
BehaviorWindow = tuple[
    list[int],
    list[bool],
    list[int],
    list[int],
    list[bool],
    list[float],
]


@dataclass(frozen=True, slots=True)
class PinnedCalibrationDataset:
    reference: ArtifactRef
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    fingerprint: str
    source_revisions: tuple[tuple[str, str], ...]
    token_role_ids: torch.Tensor | None = None
    reasoning_mode_ids: torch.Tensor | None = None
    distillation_target_mask: torch.Tensor | None = None
    distillation_weights: torch.Tensor | None = None
    behavior_profile: str = "mode_unaware"


@dataclass(frozen=True, slots=True)
class PreparedBehaviorEvaluation:
    mode: ReasoningMode
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_mask: torch.Tensor
    identity: str


def load_pinned_calibration(output: str | Path, reference: ArtifactRef) -> PinnedCalibrationDataset:
    artifacts = LocalArtifactStore(Path(output) / "artifacts")
    artifacts.validate(reference.artifact_id)
    root = artifacts.path_for(reference.artifact_id)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tensor_artifact_id = str(manifest["tensor_artifact"])
    artifacts.validate(tensor_artifact_id)
    tensor_path = artifacts.path_for(tensor_artifact_id) / "tensors.safetensors"
    tensor_names = tuple(
        str(name)
        for name in manifest.get(
            "tensor_names",
            ("input_ids", "attention_mask"),
        )
    )
    tensors = load_tensors(tensor_path, tensor_names)
    source_revisions = tuple((str(name), str(revision)) for name, revision in manifest["source_revisions"].items())
    return PinnedCalibrationDataset(
        reference,
        tensors["input_ids"],
        tensors["attention_mask"],
        str(manifest["fingerprint"]),
        source_revisions,
        tensors.get("token_role_ids"),
        tensors.get("reasoning_mode_ids"),
        tensors.get("distillation_target_mask"),
        tensors.get("distillation_weights"),
        str(manifest.get("behavior_profile", "mode_unaware")),
    )


def materialize_pinned_calibration(
    source_output: str | Path,
    destination_output: str | Path,
    *,
    sample_count: int,
    sequence_length: int,
    seed: int,
    preparation_id: str | None,
    tokenizer_identity: str,
) -> PinnedCalibrationDataset:
    """Copy validated deterministic tokens into a run-local artifact store."""

    source_output = Path(source_output)
    receipt = json.loads((source_output / CALIBRATION_RECEIPT_NAME).read_text(encoding="utf-8"))
    if (
        receipt.get("sample_count") != sample_count
        or receipt.get("sequence_length") != sequence_length
        or receipt.get("seed") != seed
    ):
        raise ValueError("source calibration receipt does not match the requested deterministic protocol")
    source_reference = ArtifactRef(
        "calibration-dataset-manifest",
        str(receipt["artifact_id"]),
        1,
    )
    source = load_pinned_calibration(source_output, source_reference)
    if tuple(source.input_ids.shape) != (sample_count, sequence_length):
        raise ValueError("source calibration token tensor has the wrong shape")
    if tuple(source.attention_mask.shape) != (sample_count, sequence_length):
        raise ValueError("source calibration attention mask has the wrong shape")

    destination_output = Path(destination_output)
    artifacts = LocalArtifactStore(destination_output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    tensor_values = {
        "input_ids": source.input_ids,
        "attention_mask": source.attention_mask,
    }
    for name in (
        "token_role_ids",
        "reasoning_mode_ids",
        "distillation_target_mask",
        "distillation_weights",
    ):
        value = getattr(source, name)
        if value is not None:
            tensor_values[name] = value
    refs = tensors.put(
        "calibration-token-dataset",
        tensor_values,
    )
    manifest = {
        "schema_version": 2 if source.behavior_profile != "mode_unaware" else 1,
        "producer": "run-local-calibration-materialization-v1",
        "sample_count": sample_count,
        "sequence_length": sequence_length,
        "seed": seed,
        "valid_token_count": int(source.attention_mask.sum()),
        "fingerprint": source.fingerprint,
        "source_revisions": dict(source.source_revisions),
        "tensor_artifact": refs["input_ids"].artifact.artifact_id,
        **(
            {}
            if source.behavior_profile == "mode_unaware"
            else {
                "tensor_names": sorted(tensor_values),
                "behavior_profile": source.behavior_profile,
            }
        ),
        "materialized_from": str(source_output.resolve()),
        "source_artifact": source.reference.artifact_id,
        "tokenizer_identity": tokenizer_identity,
    }
    with artifacts.begin_write("calibration-dataset-manifest") as writer:
        (writer.path / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    reference = ArtifactRef("calibration-dataset-manifest", descriptor.artifact_id, 1)
    atomic_write_json(
        destination_output / CALIBRATION_RECEIPT_NAME,
        {
            "schema_version": 2 if source.behavior_profile != "mode_unaware" else 1,
            "sample_count": sample_count,
            "sequence_length": sequence_length,
            "seed": seed,
            "preparation_id": preparation_id,
            "artifact_id": reference.artifact_id,
            "fingerprint": source.fingerprint,
            "source_revisions": dict(source.source_revisions),
            "materialized_from": str(source_output.resolve()),
            "source_artifact": source.reference.artifact_id,
            "tokenizer_identity": tokenizer_identity,
            **(
                {}
                if source.behavior_profile == "mode_unaware"
                else {"behavior_profile": source.behavior_profile}
            ),
        },
    )
    return PinnedCalibrationDataset(
        reference,
        source.input_ids,
        source.attention_mask,
        source.fingerprint,
        source.source_revisions,
        source.token_role_ids,
        source.reasoning_mode_ids,
        source.distillation_target_mask,
        source.distillation_weights,
        source.behavior_profile,
    )


def _chat_tokens(tokenizer: Any, messages: list[dict[str, object]]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=False,
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    return [int(value) for value in ids]


def _pack_chat_records(
    records: Iterable[dict[str, object]],
    tokenizer: Any,
    count: int,
    sequence_length: int,
) -> list[list[int]]:
    samples: list[list[int]] = []
    token_buffer: list[int] = []
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    attempts = 0
    maximum_attempts = max(count * 50, 100)
    for record in records:
        if len(samples) >= count or attempts >= maximum_attempts:
            break
        attempts += 1
        messages = cast(list[dict[str, object]], record.get("messages") or [])
        if not messages:
            continue
        ids = _chat_tokens(tokenizer, messages)
        if len(ids) < 8:
            continue
        token_buffer.extend(ids)
        if eos is not None and token_buffer[-1] != eos:
            token_buffer.append(int(eos))
        while len(token_buffer) >= sequence_length and len(samples) < count:
            samples.append(token_buffer[:sequence_length])
            token_buffer = token_buffer[sequence_length:]
    if len(samples) != count:
        raise ValueError(f"UltraChat produced {len(samples)} windows; expected {count}")
    return samples


def _slice_wikitext(
    text: str,
    tokenizer: Any,
    count: int,
    sequence_length: int,
    rng: random.Random,
) -> list[list[int]]:
    encoded = tokenizer(text, return_tensors="pt").input_ids
    if encoded.shape[1] <= sequence_length:
        raise ValueError("WikiText token stream is shorter than the calibration sequence length")
    samples = []
    for _ in range(count):
        start = rng.randint(0, encoded.shape[1] - sequence_length - 1)
        samples.append(encoded[0, start : start + sequence_length].tolist())
    return samples


def prepare_experiment018_calibration(
    snapshot: str | Path,
    output: str | Path,
    *,
    sample_count: int = 256,
    sequence_length: int = 2048,
    seed: int = 0,
) -> PinnedCalibrationDataset:
    if sample_count <= 0 or sample_count % 2:
        raise ValueError("Experiment 018 calibration requires a positive even sample count")
    snapshot = Path(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=False)
    per_source = sample_count // 2
    chat = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,
        revision=ULTRACHAT_REVISION,
    ).shuffle(buffer_size=10_000, seed=seed)
    chat_samples = _pack_chat_records(iter(chat), tokenizer, per_source, sequence_length)
    wiki = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        revision=WIKITEXT_REVISION,
    )
    rng = random.Random(seed + 1)
    wiki_samples = _slice_wikitext(
        "\n\n".join(wiki["text"]),
        tokenizer,
        per_source,
        sequence_length,
        rng,
    )
    samples = [*chat_samples, *wiki_samples]
    rng.shuffle(samples)
    input_ids = torch.tensor(samples, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    digest = hashlib.sha256()
    digest.update(input_ids.contiguous().view(torch.uint8).numpy().tobytes())
    digest.update(ULTRACHAT_REVISION.encode())
    digest.update(WIKITEXT_REVISION.encode())
    digest.update(str(seed).encode())
    fingerprint = "sha256:" + digest.hexdigest()
    artifacts = LocalArtifactStore(Path(output) / "artifacts")
    tensors = LocalTensorStore(artifacts)
    refs = tensors.put(
        "calibration-token-dataset",
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    tensor_artifact = refs["input_ids"].artifact
    manifest = {
        "schema_version": 1,
        "producer": "experiment018-calibration-v1",
        "sample_count": sample_count,
        "sequence_length": sequence_length,
        "seed": seed,
        "valid_token_count": int(attention_mask.sum()),
        "fingerprint": fingerprint,
        "source_revisions": {
            "HuggingFaceH4/ultrachat_200k": ULTRACHAT_REVISION,
            "Salesforce/wikitext": WIKITEXT_REVISION,
        },
        "tensor_artifact": tensor_artifact.artifact_id,
    }
    with artifacts.begin_write("calibration-dataset-manifest") as writer:
        (writer.path / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    return PinnedCalibrationDataset(
        ArtifactRef("calibration-dataset-manifest", descriptor.artifact_id, 1),
        input_ids,
        attention_mask,
        fingerprint,
        (
            ("HuggingFaceH4/ultrachat_200k", ULTRACHAT_REVISION),
            ("Salesforce/wikitext", WIKITEXT_REVISION),
        ),
    )


def _record_hash(record: RenderedBehaviorRecord) -> str:
    return tensor_sha256(torch.tensor(record.input_ids, dtype=torch.long))


def _allocate_slice_windows(slices: tuple[BehaviorSliceConfig, ...], sample_count: int) -> tuple[int, ...]:
    exact = [item.target_valid_token_fraction * sample_count for item in slices]
    counts = [int(value) for value in exact]
    remainder = sample_count - sum(counts)
    order = sorted(range(len(slices)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    if any(count <= 0 for count in counts):
        raise ValueError("behavior token budget is too small to allocate every configured slice")
    return tuple(counts)


def _pad_behavior_record(
    records: list[RenderedBehaviorRecord],
    sequence_length: int,
    pad_token_id: int,
) -> tuple[list[int], list[bool], list[int], list[int], list[bool], list[float]]:
    tokens: list[int] = []
    roles: list[int] = []
    modes: list[int] = []
    targets: list[bool] = []
    weights: list[float] = []
    for record in records:
        tokens.extend(record.input_ids)
        roles.extend(record.token_role_ids)
        modes.extend(record.reasoning_mode_ids)
        targets.extend(record.distillation_target_mask)
        weights.extend(record.distillation_weights)
    valid = len(tokens)
    padding = sequence_length - valid
    tokens.extend([pad_token_id] * padding)
    roles.extend([int(TokenRole.PADDING)] * padding)
    modes.extend([int(ReasoningModeId.PADDING)] * padding)
    targets.extend([False] * padding)
    weights.extend([0.0] * padding)
    return tokens, [True] * valid + [False] * padding, roles, modes, targets, weights


def _pack_behavior_records(
    records: Iterable[RenderedBehaviorRecord],
    *,
    count: int,
    sequence_length: int,
    pad_token_id: int,
) -> tuple[
    list[BehaviorWindow],
    tuple[str, ...],
    int,
]:
    bins: list[list[RenderedBehaviorRecord]] = [[] for _index in range(count)]
    bin_tokens = [0] * count
    bin_receipts: list[list[str]] = [[] for _index in range(count)]
    rejected_length = 0
    attempts = 0
    maximum_attempts = max(count * 100, 1_000)
    for record in records:
        if attempts >= maximum_attempts or all(
            used >= int(sequence_length * 0.98) for used in bin_tokens
        ):
            break
        attempts += 1
        if len(record.input_ids) > sequence_length:
            rejected_length += 1
            continue
        candidates = [
            (sequence_length - bin_tokens[index] - len(record.input_ids), index)
            for index in range(count)
            if bin_tokens[index] + len(record.input_ids) <= sequence_length
        ]
        if not candidates:
            continue
        _remaining, selected = min(candidates)
        bins[selected].append(record)
        bin_tokens[selected] += len(record.input_ids)
        bin_receipts[selected].append(_record_hash(record))
    if any(not values for values in bins):
        completed = sum(bool(values) for values in bins)
        raise ValueError(f"behavior slice produced {completed} complete-record windows; expected {count}")
    windows = [
        _pad_behavior_record(values, sequence_length, pad_token_id)
        for values in bins
    ]
    receipts = tuple(receipt for values in bin_receipts for receipt in values)
    return windows, receipts, rejected_length


def _split_inline_reasoning(value: str) -> tuple[str, str]:
    opening = value.find("<think>")
    closing = value.find("</think>")
    if opening < 0 or closing < 0 or closing <= opening + len("<think>"):
        raise ValueError("OpenR1 generation has no complete non-empty thinking span")
    reasoning = value[opening + len("<think>") : closing].strip()
    answer = value[closing + len("</think>") :].strip()
    if not reasoning or not answer:
        raise ValueError("OpenR1 generation must contain reasoning and a final answer")
    return reasoning, answer


def _openr1_messages(record: dict[str, object]) -> list[dict[str, object]]:
    generations = record.get("generations")
    if not isinstance(generations, list) or not generations:
        raise ValueError("OpenR1 record contains no generations")
    correctness = record.get("correctness_math_verify")
    selected = None
    if isinstance(correctness, list):
        selected = next(
            (generation for generation, correct in zip(generations, correctness, strict=False) if bool(correct)),
            None,
        )
        if selected is None:
            raise ValueError("OpenR1 record contains no generation marked correct")
    if selected is None:
        selected = generations[0]
    reasoning, answer = _split_inline_reasoning(str(selected))
    problem = str(record.get("problem") or "").strip()
    if not problem:
        raise ValueError("OpenR1 record contains no problem")
    return [
        {"role": "user", "content": problem},
        {"role": "assistant", "reasoning_content": reasoning, "content": answer},
    ]


def _load_behavior_source(item: BehaviorSliceConfig, seed: int) -> Iterable[dict[str, object]]:
    source = item.source
    arguments: dict[str, object] = {
        "split": source.split,
        "revision": source.revision,
    }
    if item.record_format != "raw_text":
        arguments["streaming"] = True
    positional = (source.name,) if source.subset is None else (source.name, source.subset)
    dataset = load_dataset(*positional, **arguments)
    if item.record_format != "raw_text":
        dataset = dataset.shuffle(buffer_size=10_000, seed=seed)
    records = cast(Iterable[dict[str, object]], dataset)
    if item.record_format == "raw_text":
        return records
    ranges = {"train": (0, 80), "quick": (80, 90), "final": (90, 100)}
    lower, upper = ranges[item.partition]

    def selected() -> Iterable[dict[str, object]]:
        for record in records:
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode()
            bucket = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % 100
            if lower <= bucket < upper:
                yield record

    return selected()


def _raw_behavior_windows(
    item: BehaviorSliceConfig,
    tokenizer: Any,
    records: Iterable[dict[str, object]],
    *,
    count: int,
    sequence_length: int,
    seed: int,
) -> tuple[
    list[BehaviorWindow],
    tuple[str, ...],
]:
    text = "\n\n".join(str(record.get("text") or "") for record in records)
    rows = _slice_wikitext(text, tokenizer, count, sequence_length, random.Random(seed))
    windows = []
    receipts = []
    for row in rows:
        roles = [int(TokenRole.RAW)] * sequence_length
        modes = [int(ReasoningModeId.RAW)] * sequence_length
        targets = [item.assistant_target_weight > 0] * (sequence_length - 1) + [False]
        weights = [item.assistant_target_weight] * (sequence_length - 1) + [0.0]
        windows.append((row, [True] * sequence_length, roles, modes, targets, weights))
        receipts.append(tensor_sha256(torch.tensor(row, dtype=torch.long)))
    return windows, tuple(receipts)


def prepare_behavior_calibration(
    snapshot: str | Path,
    output: str | Path,
    dataset_config: DatasetConfig,
    *,
    sample_count: int,
    sequence_length: int,
    seed: int,
) -> PinnedCalibrationDataset:
    """Prepare the versioned, mode-aware Qwen3 behavior artifact."""

    slices = dataset_config.behavior_slices
    if not slices:
        raise ValueError("behavior calibration requires at least one behavior slice")
    snapshot = Path(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=False)
    behavior = chat_behavior_for_snapshot(snapshot)
    unsupported = [
        item.mode
        for item in slices
        if item.mode is not ReasoningMode.RAW and item.mode not in behavior.supported_modes
    ]
    if unsupported:
        raise ValueError(f"model adapter does not support reasoning modes: {unsupported}")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if not isinstance(pad_token_id, int):
        raise ValueError("behavior preparation requires a scalar pad or EOS token ID")

    counts = _allocate_slice_windows(slices, sample_count)
    all_windows: list[tuple[int, BehaviorWindow]] = []
    ordered_receipts: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    source_revisions: list[tuple[str, str]] = []
    for slice_index, (item, count) in enumerate(zip(slices, counts, strict=True)):
        records = _load_behavior_source(item, seed + slice_index)
        source_revisions.append((item.source.name, str(item.source.revision)))
        rejected_schema = 0
        rejected_length = 0
        if item.mode is ReasoningMode.RAW:
            windows, receipts = _raw_behavior_windows(
                item,
                tokenizer,
                records,
                count=count,
                sequence_length=sequence_length,
                seed=seed + slice_index,
            )
        else:
            def rendered_records(
                source_records: Iterable[dict[str, object]] = records,
                slice_config: BehaviorSliceConfig = item,
            ) -> Iterable[RenderedBehaviorRecord]:
                nonlocal rejected_schema
                for record in source_records:
                    try:
                        if slice_config.record_format == "openr1_generations":
                            messages = _openr1_messages(record)
                        elif slice_config.record_format == "ultrachat_messages":
                            messages = cast(list[dict[str, object]], record.get("messages") or [])
                        else:
                            raise ValueError(f"unsupported chat record format: {slice_config.record_format}")
                        yield behavior.render_completed(
                            tokenizer,
                            messages,
                            slice_config.mode,
                            assistant_target_weight=slice_config.assistant_target_weight,
                            prompt_target_weight=slice_config.prompt_target_weight,
                        )
                    except (TypeError, ValueError):
                        rejected_schema += 1

            windows, receipts, rejected_length = _pack_behavior_records(
                rendered_records(),
                count=count,
                sequence_length=sequence_length,
                pad_token_id=pad_token_id,
            )
        valid_tokens = sum(sum(window[1]) for window in windows)
        target_tokens = sum(sum(window[4]) for window in windows)
        if item.minimum_valid_tokens is not None and valid_tokens < item.minimum_valid_tokens:
            raise ValueError(
                f"behavior slice {item.name!r} produced {valid_tokens} valid tokens; "
                f"required at least {item.minimum_valid_tokens}"
            )
        summaries.append(
            {
                "name": item.name,
                "mode": item.mode.value,
                "target_fraction": item.target_valid_token_fraction,
                "window_count": count,
                "valid_token_count": valid_tokens,
                "target_token_count": target_tokens,
                "accepted_record_count": len(receipts),
                "rejected_schema_count": rejected_schema,
                "rejected_length_count": rejected_length,
                "packing_utilization": valid_tokens / (count * sequence_length),
            }
        )
        ordered_receipts.extend({"slice": item.name, "content_hash": value} for value in receipts)
        all_windows.extend((slice_index, window) for window in windows)

    # Interleave slice windows round-robin so logical batches cannot collapse
    # into a single behavior merely because physical microbatching changes.
    by_slice = [
        [window for observed_slice, window in all_windows if observed_slice == slice_index]
        for slice_index in range(len(slices))
    ]
    all_windows = []
    for window_index in range(max(len(values) for values in by_slice)):
        for slice_index, slice_windows in enumerate(by_slice):
            if window_index < len(slice_windows):
                all_windows.append((slice_index, slice_windows[window_index]))
    values = list(zip(*(window for _slice_index, window in all_windows), strict=True))
    input_ids = torch.tensor(values[0], dtype=torch.long)
    attention_mask = torch.tensor(values[1], dtype=torch.bool)
    token_role_ids = torch.tensor(values[2], dtype=torch.uint8)
    reasoning_mode_ids = torch.tensor(values[3], dtype=torch.uint8)
    distillation_target_mask = torch.tensor(values[4], dtype=torch.bool)
    distillation_weights = torch.tensor(values[5], dtype=torch.float32)
    policy_identity = behavior.policy_identity(tokenizer)
    behavior_profile = semantic_hash(
        {
            "implementation": "behavior-dataset-v1",
            "slices": [to_dict(item) for item in slices],
            "chat_policy": policy_identity,
            "ordered_records": ordered_receipts,
        }
    )
    fingerprint = tensor_sha256(
        input_ids,
        attention_mask,
        token_role_ids,
        reasoning_mode_ids,
        distillation_target_mask,
        distillation_weights,
    )
    tensor_values = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_role_ids": token_role_ids,
        "reasoning_mode_ids": reasoning_mode_ids,
        "distillation_target_mask": distillation_target_mask,
        "distillation_weights": distillation_weights,
    }
    artifacts = LocalArtifactStore(Path(output) / "artifacts")
    refs = LocalTensorStore(artifacts).put("calibration-token-dataset", tensor_values)
    manifest = {
        "schema_version": 2,
        "producer": "qwen3-behavior-calibration-v1",
        "sample_count": sample_count,
        "sequence_length": sequence_length,
        "seed": seed,
        "valid_token_count": int(attention_mask.sum()),
        "target_token_count": int(distillation_target_mask.sum()),
        "fingerprint": fingerprint,
        "behavior_profile": behavior_profile,
        "chat_policy_identity": policy_identity,
        "source_revisions": dict(source_revisions),
        "tensor_artifact": refs["input_ids"].artifact.artifact_id,
        "tensor_names": sorted(tensor_values),
        "ordered_record_receipts": ordered_receipts,
        "slice_summaries": summaries,
    }
    with artifacts.begin_write("calibration-dataset-manifest") as writer:
        (writer.path / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    return PinnedCalibrationDataset(
        ArtifactRef("calibration-dataset-manifest", descriptor.artifact_id, 1),
        input_ids,
        attention_mask,
        fingerprint,
        tuple(source_revisions),
        token_role_ids,
        reasoning_mode_ids,
        distillation_target_mask,
        distillation_weights,
        behavior_profile,
    )


def prepare_behavior_evaluation(
    snapshot: str | Path,
    item: BehaviorSliceConfig,
    *,
    sample_count: int,
    sequence_length: int,
    seed: int,
) -> PreparedBehaviorEvaluation:
    """Prepare a disjoint held-out chat slice for response-token NLL."""

    if item.mode is ReasoningMode.RAW:
        raise ValueError("reasoning evaluation requires a chat behavior slice")
    snapshot = Path(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=False)
    behavior = chat_behavior_for_snapshot(snapshot)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if not isinstance(pad_token_id, int):
        raise ValueError("behavior evaluation requires a scalar pad or EOS token ID")
    records = _load_behavior_source(item, seed)

    def rendered_records() -> Iterable[RenderedBehaviorRecord]:
        for record in records:
            try:
                messages = (
                    _openr1_messages(record)
                    if item.record_format == "openr1_generations"
                    else cast(list[dict[str, object]], record.get("messages") or [])
                )
                yield behavior.render_completed(
                    tokenizer,
                    messages,
                    item.mode,
                    assistant_target_weight=1.0,
                    prompt_target_weight=0.0,
                )
            except (TypeError, ValueError):
                continue

    windows = []
    receipt_values = []
    for rendered in rendered_records():
        if len(rendered.input_ids) > sequence_length:
            continue
        windows.append(_pad_behavior_record([rendered], sequence_length, pad_token_id))
        receipt_values.append(_record_hash(rendered))
        if len(windows) == sample_count:
            break
    if len(windows) != sample_count:
        raise ValueError(
            f"behavior evaluation produced {len(windows)} complete records; expected {sample_count}"
        )
    receipts = tuple(receipt_values)
    values = list(zip(*windows, strict=True))
    input_ids = torch.tensor(values[0], dtype=torch.long)
    attention_mask = torch.tensor(values[1], dtype=torch.bool)
    target_mask = torch.tensor(values[4], dtype=torch.bool)
    identity = semantic_hash(
        {
            "implementation": "behavior-evaluation-v1",
            "slice": to_dict(item),
            "chat_policy": behavior.policy_identity(tokenizer),
            "ordered_records": receipts,
            "input_hash": tensor_sha256(input_ids, attention_mask, target_mask),
        }
    )
    return PreparedBehaviorEvaluation(item.mode, input_ids, attention_mask, target_mask, identity)


def load_or_prepare_calibration(
    snapshot: str | Path,
    output: str | Path,
    *,
    sample_count: int = 256,
    sequence_length: int = 2048,
    seed: int = 0,
    preparation_id: str | None = None,
    dataset_config: DatasetConfig | None = None,
) -> PinnedCalibrationDataset:
    """Load this run's generated calibration tokens, or create them when needed."""

    output = Path(output)
    receipt_path = output / CALIBRATION_RECEIPT_NAME
    requested = {
        "sample_count": sample_count,
        "sequence_length": sequence_length,
        "seed": seed,
        "preparation_id": preparation_id,
        **(
            {}
            if dataset_config is None or not dataset_config.behavior_slices
            else {
                "behavior_profile_request": semantic_hash(
                    [to_dict(item) for item in dataset_config.behavior_slices]
                )
            }
        ),
    }
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in requested.items()):
            raise ValueError("calibration receipt does not match this run")
        reference = ArtifactRef(
            "calibration-dataset-manifest",
            str(receipt["artifact_id"]),
            1,
        )
        calibration = load_pinned_calibration(output, reference)
        if tuple(calibration.input_ids.shape) != (sample_count, sequence_length):
            raise ValueError("generated calibration tensor has the wrong shape")
        if tuple(calibration.attention_mask.shape) != (sample_count, sequence_length):
            raise ValueError("generated calibration mask has the wrong shape")
        return calibration
    except (ArtifactCorruptionError, KeyError, OSError, TypeError, ValueError):
        pass

    if dataset_config is not None and dataset_config.behavior_slices:
        calibration = prepare_behavior_calibration(
            snapshot,
            output,
            dataset_config,
            sample_count=sample_count,
            sequence_length=sequence_length,
            seed=seed,
        )
    else:
        calibration = prepare_experiment018_calibration(
            snapshot,
            output,
            sample_count=sample_count,
            sequence_length=sequence_length,
            seed=seed,
        )
    atomic_write_json(
        receipt_path,
        {
            "schema_version": 2 if calibration.behavior_profile != "mode_unaware" else 1,
            **requested,
            "artifact_id": calibration.reference.artifact_id,
            "fingerprint": calibration.fingerprint,
            "source_revisions": dict(calibration.source_revisions),
        },
    )
    return calibration
