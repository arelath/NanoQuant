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
from safetensors import safe_open
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.tensor_store import LocalTensorStore

ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CALIBRATION_RECEIPT_NAME = "calibration-input.json"


@dataclass(frozen=True, slots=True)
class PinnedCalibrationDataset:
    reference: ArtifactRef
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    fingerprint: str
    source_revisions: tuple[tuple[str, str], ...]


def load_pinned_calibration(output: str | Path, reference: ArtifactRef) -> PinnedCalibrationDataset:
    artifacts = LocalArtifactStore(Path(output) / "artifacts")
    artifacts.validate(reference.artifact_id)
    root = artifacts.path_for(reference.artifact_id)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tensor_artifact_id = str(manifest["tensor_artifact"])
    artifacts.validate(tensor_artifact_id)
    tensor_path = artifacts.path_for(tensor_artifact_id) / "tensors.safetensors"
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        input_ids = handle.get_tensor("input_ids")
        attention_mask = handle.get_tensor("attention_mask")
    source_revisions = tuple((str(name), str(revision)) for name, revision in manifest["source_revisions"].items())
    return PinnedCalibrationDataset(
        reference,
        input_ids,
        attention_mask,
        str(manifest["fingerprint"]),
        source_revisions,
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


def load_or_prepare_calibration(
    snapshot: str | Path,
    output: str | Path,
    *,
    sample_count: int = 256,
    sequence_length: int = 2048,
    seed: int = 0,
    preparation_id: str | None = None,
) -> PinnedCalibrationDataset:
    """Load this run's generated calibration tokens, or create them when needed."""

    output = Path(output)
    receipt_path = output / CALIBRATION_RECEIPT_NAME
    requested = {
        "sample_count": sample_count,
        "sequence_length": sequence_length,
        "seed": seed,
        "preparation_id": preparation_id,
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
            "schema_version": 1,
            **requested,
            "artifact_id": calibration.reference.artifact_id,
            "fingerprint": calibration.fingerprint,
            "source_revisions": dict(calibration.source_revisions),
        },
    )
    return calibration
