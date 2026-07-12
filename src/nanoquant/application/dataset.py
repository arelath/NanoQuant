"""Versioned deterministic dataset mixture, formatting, selection, and tokenization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

import torch

from nanoquant.config.schema import DatasetConfig, DatasetSourceConfig


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    sample_id: str
    text: str | None = None
    messages: tuple[dict[str, str], ...] | None = None


class DatasetProvider(Protocol):
    def records(self, source: DatasetSourceConfig) -> tuple[DatasetRecord, ...]: ...


class CalibrationTokenizer(Protocol):
    name_or_path: str
    bos_token_id: int | None
    eos_token_id: int | list[int] | None
    pad_token_id: int | None
    chat_template: str | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...
    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    source: str
    revision: str
    chat_template_hash: str | None
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]
    pad_token_id: int


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    schema_version: int
    fingerprint: str
    source_revisions: tuple[tuple[str, str], ...]
    selected_sample_ids: tuple[str, ...]
    tokenizer: TokenizerIdentity
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    valid_token_count: int


def _counts(sources: tuple[DatasetSourceConfig, ...], sample_count: int) -> tuple[int, ...]:
    if sample_count <= 0 or not sources or any(source.weight < 0 for source in sources):
        raise ValueError("dataset mixture requires positive samples and non-negative source weights")
    total = sum(source.weight for source in sources)
    if total <= 0:
        raise ValueError("dataset source weights sum to zero")
    exact = [sample_count * source.weight / total for source in sources]
    result = [math.floor(value) for value in exact]
    for index in sorted(range(len(sources)), key=lambda i: (exact[i] - result[i], -i), reverse=True)[
        : sample_count - sum(result)
    ]:
        result[index] += 1
    return tuple(result)


def _selection_key(seed: int, source: str, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{source}\0{sample_id}".encode()).digest()


def prepare_dataset(
    config: DatasetConfig,
    sample_count: int,
    sequence_length: int,
    provider: DatasetProvider,
    tokenizer: CalibrationTokenizer,
    *,
    tokenizer_source: str,
    tokenizer_revision: str,
) -> PreparedDataset:
    if sequence_length <= 0 or not tokenizer_revision:
        raise ValueError("sequence length and pinned tokenizer revision are required")
    pad = tokenizer.pad_token_id
    if pad is None:
        eos = tokenizer.eos_token_id
        pad = eos[0] if isinstance(eos, list) else eos
    if pad is None:
        raise ValueError("tokenizer has neither pad nor EOS token")
    selected: list[tuple[DatasetSourceConfig, DatasetRecord]] = []
    for source, count in zip(config.sources, _counts(config.sources, sample_count), strict=True):
        if not source.revision:
            raise ValueError(f"dataset source revision is not pinned: {source.name}")
        records = provider.records(source)
        unique = {record.sample_id: record for record in records}
        if len(unique) != len(records):
            raise ValueError(f"duplicate sample ID in source {source.name}")
        ordered = (
            sorted(records, key=lambda record: _selection_key(config.selection_seed, source.name, record.sample_id))
            if config.shuffle
            else list(records)
        )
        if len(ordered) < count:
            raise ValueError(f"source {source.name} has {len(ordered)} records but mixture requires {count}")
        selected.extend((source, record) for record in ordered[:count])
    token_rows: list[list[int]] = []
    masks: list[list[int]] = []
    selected_ids: list[str] = []
    content_hashes: list[str] = []
    for source, record in selected:
        if record.messages is not None:
            text = tokenizer.apply_chat_template(list(record.messages), tokenize=False, add_generation_prompt=False)
        elif record.text is not None:
            text = record.text
        else:
            raise ValueError(f"sample {record.sample_id} has no content")
        tokens = tokenizer.encode(text, add_special_tokens=True)[:sequence_length]
        mask = [1] * len(tokens)
        tokens.extend([pad] * (sequence_length - len(tokens)))
        mask.extend([0] * (sequence_length - len(mask)))
        token_rows.append(tokens)
        masks.append(mask)
        selected_ids.append(f"{source.name}:{record.sample_id}")
        content_hashes.append(hashlib.sha256(text.encode()).hexdigest())
    eos = tokenizer.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, list) else (() if eos is None else (eos,))
    template_hash = (
        None if tokenizer.chat_template is None else hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    )
    identity = TokenizerIdentity(
        tokenizer_source,
        tokenizer_revision,
        None if template_hash is None else f"sha256:{template_hash}",
        tokenizer.bos_token_id,
        eos_ids,
        pad,
    )
    fingerprint_payload = {
        "schema": 1,
        "sources": [
            (source.name, source.revision, source.split, source.subset, source.weight) for source in config.sources
        ],
        "formatting": config.formatting,
        "seed": config.selection_seed,
        "samples": list(zip(selected_ids, content_hashes, strict=True)),
        "tokenizer": identity.__dict__
        if hasattr(identity, "__dict__")
        else {
            "source": identity.source,
            "revision": identity.revision,
            "template": identity.chat_template_hash,
            "bos": identity.bos_token_id,
            "eos": identity.eos_token_ids,
            "pad": identity.pad_token_id,
        },
        "sequence_length": sequence_length,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    input_ids = torch.tensor(token_rows, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.bool)
    return PreparedDataset(
        1,
        f"sha256:{fingerprint}",
        tuple((source.name, source.revision or "") for source in config.sources),
        tuple(selected_ids),
        identity,
        input_ids,
        attention_mask,
        int(attention_mask.sum()),
    )
