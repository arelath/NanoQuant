"""Pinned Hugging Face inputs for legacy-compatible multiple-choice evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from nanoquant.application.task_evaluation import (
    MultipleChoiceTaskSpec,
    MultipleChoiceTokenizerIdentity,
    PreparedMultipleChoiceInputs,
    prepare_multiple_choice_inputs,
)

PAIR_ENCODING_VERSION = "lm-eval-causal-pair-v1"
PREPROCESSING_VERSION = "nanoquant-multiple-choice-v1"
TOKENIZER_BEHAVIOR_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


class CausalTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class HFCausalPairTokenizer:
    """Reproduce lm-eval 0.4.12 causal context/continuation splitting."""

    tokenizer: CausalTokenizer
    add_special_tokens: bool = True

    def __call__(self, context: str, continuation: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not context:
            raise ValueError("causal pair tokenization requires a non-empty context")
        trailing_spaces = len(context) - len(context.rstrip())
        if trailing_spaces:
            continuation = context[-trailing_spaces:] + continuation
            context = context[:-trailing_spaces]
        whole = tuple(
            int(token)
            for token in self.tokenizer.encode(
                context + continuation,
                add_special_tokens=self.add_special_tokens,
            )
        )
        context_tokens = tuple(
            int(token)
            for token in self.tokenizer.encode(
                context,
                add_special_tokens=self.add_special_tokens,
            )
        )
        if not context_tokens:
            raise ValueError("causal pair tokenizer produced an empty context")
        continuation_tokens = whole[len(context_tokens) :]
        if not continuation_tokens:
            raise ValueError("causal pair tokenizer produced an empty continuation")
        return context_tokens, continuation_tokens


def hash_hf_tokenizer_snapshot(snapshot: str | Path) -> str:
    root = Path(snapshot).resolve()
    if not root.is_dir():
        raise ValueError(f"tokenizer snapshot is not a directory: {root}")
    files = tuple(root / name for name in TOKENIZER_BEHAVIOR_FILES if (root / name).is_file())
    if not files:
        raise ValueError("tokenizer snapshot contains no recognized behavior files")
    digest = hashlib.sha256()
    for path in files:
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def load_pinned_dataset_split(
    dataset_name: str,
    dataset_config: str | None,
    revision: str,
    split: str,
    *,
    local_files_only: bool,
) -> Any:
    """Load an exact cached Arrow split without asking the Hub to resolve it."""

    # `datasets` is an evaluation extra. Import lazily so the core package and
    # non-evaluation commands do not require it.
    from datasets import Dataset, DownloadConfig, config, load_dataset  # type: ignore[import-untyped]

    if not local_files_only:
        arguments = (dataset_name,) if dataset_config is None else (dataset_name, dataset_config)
        return load_dataset(
            *arguments,
            revision=revision,
            split=split,
            download_config=DownloadConfig(local_files_only=False),
        )

    dataset_directory = dataset_name.replace("/", "___")
    configuration_directory = "default" if dataset_config is None else dataset_config
    cache_root = Path(config.HF_DATASETS_CACHE) / dataset_directory / configuration_directory
    candidates = tuple(sorted(cache_root.glob(f"*/{revision}/*-{split}.arrow")))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "pinned local dataset split requires exactly one cached Arrow file: "
            f"dataset={dataset_name!r}, config={dataset_config!r}, revision={revision!r}, "
            f"split={split!r}, matches={tuple(str(path) for path in candidates)!r}"
        )
    return Dataset.from_file(str(candidates[0]))


def load_pinned_multiple_choice_documents(
    task: MultipleChoiceTaskSpec,
    *,
    maximum_samples: int | None = None,
    local_files_only: bool = True,
) -> tuple[Mapping[str, object], ...]:
    if maximum_samples is not None and (type(maximum_samples) is not int or maximum_samples <= 0):
        raise ValueError("multiple-choice maximum samples must be a positive integer")
    dataset = load_pinned_dataset_split(
        task.dataset_name,
        task.dataset_config,
        task.dataset_revision,
        task.split,
        local_files_only=local_files_only,
    )
    count = len(dataset) if maximum_samples is None else min(maximum_samples, len(dataset))
    return tuple(cast(Mapping[str, object], dict(dataset[index])) for index in range(count))


def prepare_pinned_hf_multiple_choice_inputs(
    task: MultipleChoiceTaskSpec,
    tokenizer: CausalTokenizer,
    *,
    tokenizer_name: str,
    tokenizer_revision: str,
    tokenizer_content_hash: str,
    maximum_samples: int | None = 200,
    local_files_only: bool = True,
) -> PreparedMultipleChoiceInputs:
    pair_tokenizer = HFCausalPairTokenizer(tokenizer, add_special_tokens=True)
    tokenizer_object = cast(Any, tokenizer)
    identity = MultipleChoiceTokenizerIdentity(
        tokenizer_name,
        tokenizer_revision,
        tokenizer_content_hash,
        (
            ("add_special_tokens", True),
            ("bos_token_id", getattr(tokenizer_object, "bos_token_id", None)),
            ("eos_token_id", getattr(tokenizer_object, "eos_token_id", None)),
            ("is_fast", bool(getattr(tokenizer_object, "is_fast", False))),
            ("pair_encoding", PAIR_ENCODING_VERSION),
            ("tokenizer_class", type(tokenizer).__name__),
        ),
    )
    documents = load_pinned_multiple_choice_documents(
        task,
        maximum_samples=maximum_samples,
        local_files_only=local_files_only,
    )
    return prepare_multiple_choice_inputs(
        task,
        documents,
        pair_tokenizer,
        identity,
        maximum_samples=maximum_samples,
        partition_version="legacy-ordered-limit-v1",
        preprocessing_version=PREPROCESSING_VERSION,
    )
